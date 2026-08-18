from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx


class ProviderError(RuntimeError):
    pass


class ModelProvider(ABC):
    @property
    def identity(self) -> dict[str, str]:
        return {"kind": type(self).__name__}

    @abstractmethod
    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        """Return parsed JSON, prompt tokens, completion tokens."""


class MockProvider(ModelProvider):
    @property
    def identity(self) -> dict[str, str]:
        return {"kind": "mock", "model": "deterministic-mock"}

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        del temperature
        if "planning node" in system:
            payload: dict[str, Any] = {
                "steps": ["inspect", "change", "verify"],
                "risks": [],
                "acceptance_tests": ["deterministic checks pass"],
            }
        elif "semantic verifier" in system:
            payload = {"verdict": "pass", "reasons": [], "confidence": 0.99}
        else:
            payload = {
                "result": "Mock candidate output",
                "changed_items": [],
                "assumptions": [],
            }
        return payload, max(1, len(user) // 4), 24


class OpenAICompatibleProvider(ModelProvider):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "local",
        timeout_seconds: float = 180.0,
        default_temperature: float = 0.1,
        json_mode: str = "auto",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        normalized_json_mode = json_mode.strip().lower()
        if normalized_json_mode not in {"auto", "strict", "off"}:
            raise ValueError("json_mode must be 'auto', 'strict', or 'off'")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.default_temperature = default_temperature
        self.json_mode = normalized_json_mode
        self.transport = transport

    @property
    def identity(self) -> dict[str, str]:
        return {
            "kind": "openai-compatible",
            "base_url": self.base_url,
            "model": self.model,
            "json_mode": self.json_mode,
        }

    @classmethod
    def from_env(cls) -> "OpenAICompatibleProvider":
        return cls(
            base_url=os.getenv("GRAPH_MODEL_BASE_URL", "http://127.0.0.1:8000/v1"),
            model=os.getenv("GRAPH_MODEL_NAME", "local-model"),
            api_key=os.getenv("GRAPH_MODEL_API_KEY", "local"),
            timeout_seconds=float(os.getenv("GRAPH_MODEL_TIMEOUT_SECONDS", "180")),
            default_temperature=float(os.getenv("GRAPH_MODEL_TEMPERATURE", "0.1")),
            json_mode=os.getenv("GRAPH_MODEL_JSON_MODE", "auto"),
        )

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.default_temperature if temperature is None else temperature,
            "stream": False,
        }
        if self.json_mode != "off":
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if (
                    self.json_mode == "auto"
                    and "response_format" in payload
                    and response.status_code in {400, 422}
                ):
                    fallback_payload = dict(payload)
                    fallback_payload.pop("response_format", None)
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        json=fallback_payload,
                        headers=headers,
                    )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"model request failed: {exc}") from exc

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError("model endpoint returned a non-JSON HTTP response") from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected model response: {body!r}") from exc
        parsed = _parse_json_object(content)
        usage = body.get("usage") or {}
        return (
            parsed,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
        )


def _parse_json_object(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                value = part.get("text", part.get("content"))
                if isinstance(value, str):
                    pieces.append(value)
        content = "".join(pieces)
    if not isinstance(content, str):
        raise ProviderError(f"model content is not text or JSON: {type(content).__name__}")
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise ProviderError(f"model did not return a JSON object: {content[:300]!r}")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ProviderError(f"invalid JSON returned by model: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProviderError("model JSON response must be an object")
    return parsed
