from __future__ import annotations

import ast
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

    async def complete_patch(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        """Return a repository patch proposal.

        Providers with a native raw-text path may override this to avoid embedding
        a multiline unified diff inside a JSON string. The default remains strict
        JSON for OpenAI-compatible endpoints and scripted test providers.
        """

        return await self.complete_json(
            system=system,
            user=user,
            temperature=temperature,
        )


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
        elif "failure diagnosis node" in system:
            payload = {
                "root_causes": ["mock verifier evidence requested a repair"],
                "repair_steps": ["apply one local correction"],
                "files_to_change": [],
                "evidence": [],
            }
        elif "repository patch proposal node" in system or "repair patch node" in system:
            payload = {
                "summary": "Mock no-op repository proposal",
                "patch": "",
                "assumptions": [],
                "no_changes_needed": True,
            }
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
        decoder = json.JSONDecoder()
        parsed = None
        best_position: tuple[int, int] | None = None
        last_error: json.JSONDecodeError | None = None
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, consumed = decoder.raw_decode(text[index:])
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(candidate, dict):
                # Prefer the object that ends latest in the response. For equal end positions,
                # prefer the earliest opening brace so an outer object wins over a nested object.
                position = (index + consumed, -index)
                if best_position is None or position > best_position:
                    parsed = candidate
                    best_position = position
        if parsed is None:
            # Some local models occasionally emit a Python-style mapping
            # (single quotes, True/False, trailing commas) instead of strict
            # JSON. Accept only a literal dictionary; ast.literal_eval cannot
            # execute arbitrary code. We still reject prose or non-mappings.
            literal_candidates = [text]
            for start, character in enumerate(text):
                if character != "{":
                    continue
                depth = 0
                quote: str | None = None
                escaped = False
                for end in range(start, len(text)):
                    ch = text[end]
                    if quote is not None:
                        if escaped:
                            escaped = False
                        elif ch == "\\":
                            escaped = True
                        elif ch == quote:
                            quote = None
                        continue
                    if ch in {"'", '\"'}:
                        quote = ch
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            literal_candidates.append(text[start : end + 1])
                            break
            for candidate_text in reversed(literal_candidates):
                try:
                    candidate = ast.literal_eval(candidate_text)
                except (SyntaxError, ValueError):
                    continue
                if isinstance(candidate, dict):
                    parsed = candidate
                    break
        if parsed is None:
            detail = f": {last_error}" if last_error is not None else ""
            raise ProviderError(f"model did not return a valid JSON object{detail}")
    if not isinstance(parsed, dict):
        raise ProviderError("model JSON response must be an object")
    return parsed

_PATCH_HEADER = "GRAPH_PATCH_V1"
_PATCH_META_BEGIN = "GRAPH_PATCH_META_BEGIN"
_PATCH_META_END = "GRAPH_PATCH_META_END"
_PATCH_DIFF_BEGIN = "GRAPH_PATCH_DIFF_BEGIN"
_PATCH_DIFF_END = "GRAPH_PATCH_DIFF_END"


def _coerce_model_text(content: Any) -> str:
    if isinstance(content, str):
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
        return "".join(pieces)
    raise ProviderError(f"model content is not text: {type(content).__name__}")


def _strip_outer_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```(?:[a-zA-Z0-9_-]+)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _validated_patch_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", "")
    patch = payload.get("patch", "")
    assumptions = payload.get("assumptions", [])
    no_changes_needed = payload.get("no_changes_needed", False)

    if not isinstance(summary, str):
        raise ProviderError("patch proposal summary must be a string")
    if not isinstance(patch, str):
        raise ProviderError("patch proposal patch must be a string")
    if not isinstance(assumptions, list) or not all(
        isinstance(item, str) for item in assumptions
    ):
        raise ProviderError("patch proposal assumptions must be an array of strings")
    if not isinstance(no_changes_needed, bool):
        raise ProviderError("patch proposal no_changes_needed must be boolean")

    normalized_patch = patch.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if no_changes_needed:
        if normalized_patch.strip():
            raise ProviderError(
                "no_changes_needed=true requires an empty patch payload"
            )
    elif not normalized_patch.strip():
        raise ProviderError(
            "no_changes_needed=false requires a non-empty unified diff"
        )

    return {
        "summary": summary.strip(),
        "patch": normalized_patch,
        "assumptions": assumptions,
        "no_changes_needed": no_changes_needed,
    }


def _parse_patch_proposal(content: Any) -> dict[str, Any]:
    """Parse a bounded raw patch envelope or a strict JSON fallback.

    Multiline unified diffs are intentionally kept outside JSON so local models
    do not need to escape every newline and quote. The envelope carries only a
    small JSON metadata object and a raw diff block. A strict JSON object remains
    accepted for providers whose API enforces JSON mode.
    """

    if isinstance(content, dict):
        return _validated_patch_mapping(content)

    try:
        return _validated_patch_mapping(_parse_json_object(content))
    except ProviderError:
        pass

    text = _coerce_model_text(content)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = _strip_outer_fence(text)

    header = text.rfind(_PATCH_HEADER)
    if header < 0:
        raise ProviderError("model did not return a GRAPH_PATCH_V1 envelope")

    meta_begin = text.find(_PATCH_META_BEGIN, header)
    meta_end = text.find(_PATCH_META_END, meta_begin + len(_PATCH_META_BEGIN))
    diff_begin = text.find(_PATCH_DIFF_BEGIN, meta_end + len(_PATCH_META_END))
    diff_end = text.rfind(_PATCH_DIFF_END)

    positions = (meta_begin, meta_end, diff_begin, diff_end)
    if any(position < 0 for position in positions):
        raise ProviderError("GRAPH_PATCH_V1 envelope is missing a required marker")
    if not (header < meta_begin < meta_end < diff_begin < diff_end):
        raise ProviderError("GRAPH_PATCH_V1 markers are out of order")

    metadata_text = text[
        meta_begin + len(_PATCH_META_BEGIN) : meta_end
    ].strip()
    patch_text = text[
        diff_begin + len(_PATCH_DIFF_BEGIN) : diff_end
    ].strip("\n")
    patch_text = _strip_outer_fence(patch_text)

    metadata = _parse_json_object(metadata_text)
    payload = dict(metadata)
    payload["patch"] = patch_text
    return _validated_patch_mapping(payload)

