import json

import httpx
import pytest

from graph_model.provider import OpenAICompatibleProvider, ProviderError, _parse_json_object


def test_parse_json_object_accepts_content_parts_and_fenced_json() -> None:
    parsed = _parse_json_object([{"type": "text", "text": "```json\n{\"ok\": true}\n```"}])
    assert parsed == {"ok": True}


@pytest.mark.asyncio
async def test_openai_provider_auto_falls_back_when_json_mode_is_unsupported() -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if "response_format" in payload:
            return httpx.Response(400, json={"error": "unsupported response_format"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{\"verdict\": \"pass\"}"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )

    provider = OpenAICompatibleProvider(
        base_url="http://local.test/v1",
        model="local",
        json_mode="auto",
        transport=httpx.MockTransport(handler),
    )
    payload, prompt_tokens, completion_tokens = await provider.complete_json(
        system="system",
        user="user",
    )
    assert payload == {"verdict": "pass"}
    assert (prompt_tokens, completion_tokens) == (5, 3)
    assert len(requests) == 2
    assert "response_format" in requests[0]
    assert "response_format" not in requests[1]


@pytest.mark.asyncio
async def test_openai_provider_strict_json_mode_does_not_fallback() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "unsupported"})

    provider = OpenAICompatibleProvider(
        base_url="http://local.test/v1",
        model="local",
        json_mode="strict",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ProviderError, match="model request failed"):
        await provider.complete_json(system="system", user="user")


def test_parse_json_object_prefers_the_last_complete_outer_object() -> None:
    parsed = _parse_json_object(
        'analysis example {"wrong": true} final {"ok": {"nested": true}} trailing'
    )
    assert parsed == {"ok": {"nested": True}}
