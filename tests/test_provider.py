import json

import httpx
import pytest

from graph_model.provider import (
    OpenAICompatibleProvider,
    ProviderError,
    _parse_json_object,
    _parse_patch_proposal,
)


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


def test_parse_json_object_accepts_safe_python_literal_mapping() -> None:
    parsed = _parse_json_object("analysis {'ok': True, 'items': [1, 2,],} trailing")
    assert parsed == {"ok": True, "items": [1, 2]}



def test_parse_patch_proposal_accepts_raw_multifile_envelope() -> None:
    content = """<think>private reasoning</think>
GRAPH_PATCH_V1
GRAPH_PATCH_META_BEGIN
{"summary":"Refactor email normalization","assumptions":["Tests are immutable"],"no_changes_needed":false}
GRAPH_PATCH_META_END
GRAPH_PATCH_DIFF_BEGIN
diff --git a/email/address.py b/email/address.py
--- a/email/address.py
+++ b/email/address.py
@@ -1,2 +1,2 @@
 def normalize(value: str) -> str:
-    return value
+    return value.strip().lower()
diff --git a/email/service.py b/email/service.py
--- a/email/service.py
+++ b/email/service.py
@@ -1,2 +1,2 @@
 def canonical(value: str) -> str:
-    return value.strip()
+    return value.strip().lower()
GRAPH_PATCH_DIFF_END
"""
    parsed = _parse_patch_proposal(content)
    assert parsed["summary"] == "Refactor email normalization"
    assert parsed["assumptions"] == ["Tests are immutable"]
    assert parsed["no_changes_needed"] is False
    assert parsed["patch"].count("diff --git") == 2
    assert "value.strip().lower()" in parsed["patch"]


def test_parse_patch_proposal_accepts_strict_json_fallback() -> None:
    parsed = _parse_patch_proposal(
        {
            "summary": "No change",
            "patch": "",
            "assumptions": [],
            "no_changes_needed": True,
        }
    )
    assert parsed == {
        "summary": "No change",
        "patch": "",
        "assumptions": [],
        "no_changes_needed": True,
    }


def test_parse_patch_proposal_rejects_incomplete_envelope() -> None:
    with pytest.raises(ProviderError, match="missing a required marker"):
        _parse_patch_proposal(
            """GRAPH_PATCH_V1
GRAPH_PATCH_META_BEGIN
{"summary":"x","assumptions":[],"no_changes_needed":false}
GRAPH_PATCH_META_END
GRAPH_PATCH_DIFF_BEGIN
diff --git a/a.py b/a.py
"""
        )


def test_parse_patch_proposal_rejects_conflicting_no_change_payload() -> None:
    with pytest.raises(ProviderError, match="requires an empty patch"):
        _parse_patch_proposal(
            {
                "summary": "No change",
                "patch": "diff --git a/a.py b/a.py\n",
                "assumptions": [],
                "no_changes_needed": True,
            }
        )
