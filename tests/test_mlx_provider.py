from dataclasses import dataclass
import hashlib
from pathlib import Path
from threading import get_ident
import sys
import types

import pytest

from graph_model.graph import load_default_graph
from graph_model.mlx_native.provider import MLXLocalProvider, _load_call_kwargs
from graph_model.mlx_native.qwen_hidden import RawHiddenState
from graph_model.models import RunState
from graph_model.provider import ProviderError


class FakeTokenizer:
    def __init__(self, *, template_works: bool = True) -> None:
        self.template_works = template_works
        self.messages = None
        self.template_calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=None,
    ):
        assert tokenize is False
        self.messages = messages
        self.template_calls.append(
            (messages, add_generation_prompt, get_ident(), enable_thinking)
        )
        if not self.template_works:
            raise ValueError("missing template")
        if add_generation_prompt:
            return "CHAT-TEMPLATE-PROMPT"
        return "HIDDEN-STATE-PROMPT\n" + messages[-1]["content"]

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return list(range(max(1, len(text) // 3)))


@dataclass
class FakeResponse:
    text: str
    prompt_tokens: int
    generation_tokens: int


class FakeMLXLMBackend:
    identity = "fake-mlx-lm"

    def __init__(self, *, template_works: bool = True, empty: bool = False) -> None:
        self.load_calls = 0
        self.stream_calls = 0
        self.tokenizer = FakeTokenizer(template_works=template_works)
        self.empty = empty
        self.last_prompt = None
        self.last_sampler = None
        self.thread_ids = []

    def load_model(self, model_path, *, adapter_path, revision, lazy, trust_remote_code):
        self.thread_ids.append(get_ident())
        self.load_calls += 1
        assert model_path == "local/model"
        assert adapter_path is None
        assert revision is None
        assert lazy is False
        assert trust_remote_code is False
        return object(), self.tokenizer

    def make_sampler(self, *, temperature, top_p, min_p, top_k):
        self.last_sampler = (temperature, top_p, min_p, top_k)
        return self.last_sampler

    def stream_generate(self, model, tokenizer, prompt, *, max_tokens, sampler):
        del model, tokenizer
        self.thread_ids.append(get_ident())
        self.stream_calls += 1
        self.last_prompt = prompt
        assert max_tokens == 64
        assert sampler == self.last_sampler
        if self.empty:
            return []
        return [
            FakeResponse("Reasoning omitted.\n", 11, 1),
            FakeResponse('{"ok": ', 11, 3),
            FakeResponse("true}", 11, 4),
        ]


@pytest.mark.asyncio
async def test_mlx_provider_loads_once_and_streams_json_in_process() -> None:
    backend = FakeMLXLMBackend()
    provider = MLXLocalProvider(
        model_path="local/model",
        max_tokens=64,
        top_p=0.9,
        min_p=0.05,
        top_k=20,
        backend=backend,
    )
    first = await provider.complete_json(system="system", user="user", temperature=0.0)
    second = await provider.complete_json(system="system", user="user", temperature=0.2)
    assert first == ({"ok": True}, 11, 4)
    assert second == ({"ok": True}, 11, 4)
    assert backend.load_calls == 1
    assert backend.stream_calls == 2
    assert backend.last_prompt == "CHAT-TEMPLATE-PROMPT"
    assert backend.tokenizer.messages[0]["role"] == "system"
    assert backend.tokenizer.template_calls[-1][3] is False
    assert backend.last_sampler == (0.2, 0.9, 0.05, 20)
    assert provider.loaded is True
    provider.close()


@pytest.mark.asyncio
async def test_mlx_provider_has_a_role_delimited_template_fallback() -> None:
    backend = FakeMLXLMBackend(template_works=False)
    provider = MLXLocalProvider(model_path="local/model", max_tokens=64, backend=backend)
    payload, _, _ = await provider.complete_json(system="SYSTEM", user="USER")
    assert payload == {"ok": True}
    assert "SYSTEM INSTRUCTIONS:\nSYSTEM" in backend.last_prompt
    assert "USER INPUT:\nUSER" in backend.last_prompt
    provider.close()


@pytest.mark.asyncio
async def test_mlx_provider_retries_chat_template_without_thinking_kwarg() -> None:
    class LegacyTokenizer:
        def __init__(self) -> None:
            self.messages = None

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            self.messages = messages
            return "LEGACY-TEMPLATE" if add_generation_prompt else "LEGACY-HIDDEN"

        def encode(self, text, add_special_tokens=True):
            del add_special_tokens
            return list(range(max(1, len(text) // 3)))

    backend = FakeMLXLMBackend()
    backend.tokenizer = LegacyTokenizer()
    provider = MLXLocalProvider(model_path="local/model", max_tokens=64, backend=backend)
    payload, _, _ = await provider.complete_json(system="SYSTEM", user="USER")
    assert payload == {"ok": True}
    assert backend.last_prompt == "LEGACY-TEMPLATE"
    provider.close()


@pytest.mark.asyncio
async def test_mlx_provider_rejects_empty_generation() -> None:
    provider = MLXLocalProvider(
        model_path="local/model",
        max_tokens=64,
        backend=FakeMLXLMBackend(empty=True),
    )
    with pytest.raises(ProviderError, match="empty response"):
        await provider.complete_json(system="system", user="user")
    provider.close()


@pytest.mark.asyncio
async def test_mlx_provider_falls_back_for_backend_specific_template_errors() -> None:
    class BackendSpecificTemplateError(RuntimeError):
        pass

    backend = FakeMLXLMBackend()

    def broken_template(*args, **kwargs):
        del args, kwargs
        raise BackendSpecificTemplateError("jinja failure")

    backend.tokenizer.apply_chat_template = broken_template
    provider = MLXLocalProvider(model_path="local/model", max_tokens=64, backend=backend)
    payload, _, _ = await provider.complete_json(system="SYSTEM", user="USER")
    assert payload == {"ok": True}
    assert "SYSTEM INSTRUCTIONS:\nSYSTEM" in backend.last_prompt
    provider.close()


def test_load_kwargs_adapt_to_mlx_lm_0313_signature() -> None:
    def legacy_load(
        path_or_hf_repo,
        tokenizer_config=None,
        model_config=None,
        adapter_path=None,
        lazy=False,
        return_config=False,
        revision=None,
    ):
        del (
            path_or_hf_repo, tokenizer_config, model_config, adapter_path, lazy,
            return_config, revision
        )

    kwargs = _load_call_kwargs(
        legacy_load,
        adapter_path=None,
        revision="abc123",
        lazy=False,
        trust_remote_code=True,
    )
    assert kwargs["revision"] == "abc123"
    assert kwargs["tokenizer_config"] == {"trust_remote_code": True}
    assert "trust_remote_code" not in kwargs


def test_load_kwargs_use_new_upstream_remote_code_argument() -> None:
    def current_load(
        path_or_hf_repo,
        adapter_path=None,
        lazy=False,
        revision=None,
        trust_remote_code=False,
    ):
        del path_or_hf_repo, adapter_path, lazy, revision, trust_remote_code

    kwargs = _load_call_kwargs(
        current_load,
        adapter_path="adapter",
        revision="main",
        lazy=True,
        trust_remote_code=True,
    )
    assert kwargs == {
        "adapter_path": "adapter",
        "lazy": True,
        "revision": "main",
        "trust_remote_code": True,
    }



class FakeHiddenBackend:
    identity = "fake-qwen-hidden"

    def __init__(self) -> None:
        self.calls = []
        self.thread_ids = []

    def extract_hidden_state(
        self,
        model,
        tokenizer,
        prompt,
        *,
        max_tokens,
        layer_specs,
        pooling,
    ):
        del model, tokenizer
        self.thread_ids.append(get_ident())
        self.calls.append((prompt, max_tokens, tuple(layer_specs), pooling))
        digest = hashlib.sha256(prompt.encode()).digest()
        values = tuple((byte - 127.5) / 127.5 for byte in digest[:8])
        return RawHiddenState(
            values=values,
            source="model.model",
            layer_labels=("final",),
            pooling=pooling,
            token_count=32,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            model_hidden_size=8,
        )


def test_mlx_provider_hidden_capture_is_state_dependent_and_hash_addressed(
    tmp_path: Path,
) -> None:
    backend = FakeMLXLMBackend()
    hidden_backend = FakeHiddenBackend()
    provider = MLXLocalProvider(
        model_path="local/model",
        max_tokens=64,
        backend=backend,
        hidden_backend=hidden_backend,
        hidden_capture_enabled=True,
        hidden_artifact_root=tmp_path / "hidden",
        hidden_feature_size=16,
    )
    state = RunState.new(
        graph=load_default_graph(),
        task="Fix the failing test with the smallest patch",
        run_id="hidden-provider",
    )
    first = provider.capture_policy_hidden(
        state=state,
        node_id="intake",
        decision_type="route",
    )
    repeated = provider.capture_policy_hidden(
        state=state,
        node_id="intake",
        decision_type="route",
    )
    assert repeated.reference.sha256 == first.reference.sha256
    assert first.cache_hit is False
    assert repeated.cache_hit is True
    assert len(hidden_backend.calls) == 1
    assert Path(first.reference.path).is_file()

    state.current_node = "tests"
    state.step_count = 4
    state.data.update({"verdict": "fail", "test_report": {"stderr": "expected 5, got 6"}})
    state.updated_at += 1.0
    second = provider.capture_policy_hidden(
        state=state,
        node_id="tests",
        decision_type="transition",
    )
    assert second.reference.prompt_sha256 != first.reference.prompt_sha256
    assert second.reference.sha256 != first.reference.sha256
    assert len(hidden_backend.calls) == 2
    assert set(backend.thread_ids + hidden_backend.thread_ids) == {provider._affinity_thread_id}
    provider.close()


@pytest.mark.asyncio
async def test_generation_and_hidden_capture_share_one_affinity_worker(tmp_path: Path) -> None:
    main_thread = get_ident()
    backend = FakeMLXLMBackend()
    hidden_backend = FakeHiddenBackend()
    provider = MLXLocalProvider(
        model_path="local/model",
        max_tokens=64,
        backend=backend,
        hidden_backend=hidden_backend,
        hidden_artifact_root=tmp_path / "hidden",
        hidden_feature_size=16,
    )
    await provider.complete_json(system="system", user="user")
    state = RunState.new(
        graph=load_default_graph(),
        task="Inspect and repair",
        run_id="affinity",
    )
    provider.capture_policy_hidden(
        state=state,
        node_id="intake",
        decision_type="route",
    )
    worker_ids = set(backend.thread_ids + hidden_backend.thread_ids)
    assert len(worker_ids) == 1
    assert main_thread not in worker_ids
    assert provider.identity["execution"] == "dedicated-single-worker"
    provider.close()


def test_mlx_provider_close_releases_state_and_prevents_reuse() -> None:
    backend = FakeMLXLMBackend()
    provider = MLXLocalProvider(
        model_path="local/model",
        max_tokens=64,
        backend=backend,
    )
    provider.load()
    assert provider.loaded is True
    provider.close()
    assert provider.loaded is False
    assert not provider._hidden_cache
    with pytest.raises(RuntimeError, match="closed"):
        provider.load()


def test_structured_thinking_toggle_does_not_change_resume_model_identity() -> None:
    disabled = MLXLocalProvider(
        model_path="local/model",
        max_tokens=64,
        structured_thinking_enabled=False,
        backend=FakeMLXLMBackend(),
    )
    enabled = MLXLocalProvider(
        model_path="local/model",
        max_tokens=64,
        structured_thinking_enabled=True,
        backend=FakeMLXLMBackend(),
    )
    assert disabled.identity == enabled.identity
    disabled.close()
    enabled.close()


@pytest.mark.asyncio
async def test_mlx_provider_retries_invalid_json_once_with_strict_recovery() -> None:
    class RecoveryBackend(FakeMLXLMBackend):
        def stream_generate(self, model, tokenizer, prompt, *, max_tokens, sampler):
            del model, tokenizer
            self.thread_ids.append(get_ident())
            self.stream_calls += 1
            self.last_prompt = prompt
            assert max_tokens == 64
            self.last_sampler = sampler
            if self.stream_calls == 1:
                return [FakeResponse('{"ok": ', 9, 2)]
            return [FakeResponse('{"ok": true}', 13, 4)]

    backend = RecoveryBackend()
    provider = MLXLocalProvider(model_path="local/model", max_tokens=64, backend=backend)
    payload, prompt_tokens, completion_tokens = await provider.complete_json(
        system="system", user="user", temperature=0.2
    )
    assert payload == {"ok": True}
    assert backend.stream_calls == 2
    assert prompt_tokens == 22
    assert completion_tokens == 6
    assert backend.last_sampler[0] == 0.0
    provider.close()


@pytest.mark.asyncio
async def test_mlx_provider_uses_raw_patch_envelope_without_json_escaping() -> None:
    class PatchBackend(FakeMLXLMBackend):
        def stream_generate(self, model, tokenizer, prompt, *, max_tokens, sampler):
            del model, tokenizer
            self.thread_ids.append(get_ident())
            self.stream_calls += 1
            self.last_prompt = prompt
            self.last_sampler = sampler
            assert max_tokens == 64
            return [
                FakeResponse(
                    """GRAPH_PATCH_V1
GRAPH_PATCH_META_BEGIN
{"summary":"Two-file fix","assumptions":[],"no_changes_needed":false}
GRAPH_PATCH_META_END
GRAPH_PATCH_DIFF_BEGIN
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-old
+new
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1 +1 @@
-before
+after
GRAPH_PATCH_DIFF_END
""",
                    21,
                    31,
                )
            ]

    backend = PatchBackend()
    provider = MLXLocalProvider(model_path="local/model", max_tokens=64, backend=backend)
    payload, prompt_tokens, completion_tokens = await provider.complete_patch(
        system="patch system",
        user="patch user",
        temperature=0.2,
    )
    assert payload["summary"] == "Two-file fix"
    assert payload["patch"].count("diff --git") == 2
    assert (prompt_tokens, completion_tokens) == (21, 31)
    assert backend.stream_calls == 1
    assert backend.last_sampler[0] == 0.2
    provider.close()


@pytest.mark.asyncio
async def test_mlx_provider_retries_invalid_patch_once_with_envelope_recovery() -> None:
    class PatchRecoveryBackend(FakeMLXLMBackend):
        def __init__(self) -> None:
            super().__init__()
            self.temperatures: list[float] = []

        def stream_generate(self, model, tokenizer, prompt, *, max_tokens, sampler):
            del model, tokenizer
            self.thread_ids.append(get_ident())
            self.stream_calls += 1
            self.last_prompt = prompt
            self.last_sampler = sampler
            self.temperatures.append(float(sampler[0]))
            assert max_tokens == 64
            if self.stream_calls == 1:
                return [FakeResponse("Here is the patch, but no envelope", 9, 8)]
            return [
                FakeResponse(
                    """GRAPH_PATCH_V1
GRAPH_PATCH_META_BEGIN
{"summary":"Recovered","assumptions":[],"no_changes_needed":false}
GRAPH_PATCH_META_END
GRAPH_PATCH_DIFF_BEGIN
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-old
+new
GRAPH_PATCH_DIFF_END
""",
                    13,
                    20,
                )
            ]

    backend = PatchRecoveryBackend()
    provider = MLXLocalProvider(model_path="local/model", max_tokens=64, backend=backend)
    payload, prompt_tokens, completion_tokens = await provider.complete_patch(
        system="patch system",
        user="patch user",
        temperature=0.3,
    )
    assert payload["summary"] == "Recovered"
    assert backend.stream_calls == 2
    assert backend.temperatures == [0.3, 0.0]
    assert prompt_tokens == 22
    assert completion_tokens == 28
    assert "PATCH ENVELOPE RECOVERY" in backend.tokenizer.messages[0]["content"]
    provider.close()


@pytest.mark.asyncio
async def test_mlx_provider_continues_one_truncated_patch_envelope() -> None:
    class TruncatedPatchBackend(FakeMLXLMBackend):
        def __init__(self) -> None:
            super().__init__()
            self.temperatures: list[float] = []

        def stream_generate(self, model, tokenizer, prompt, *, max_tokens, sampler):
            del model, tokenizer
            self.thread_ids.append(get_ident())
            self.stream_calls += 1
            self.last_prompt = prompt
            self.last_sampler = sampler
            self.temperatures.append(float(sampler[0]))
            assert max_tokens == 64
            if self.stream_calls == 1:
                return [FakeResponse("Malformed initial proposal", 9, 8)]
            if self.stream_calls == 2:
                return [
                    FakeResponse(
                        "<think>bounded private work</think>\nGRAPH_PATCH_V1",
                        13,
                        64,
                    )
                ]
            return [
                FakeResponse(
                    """GRAPH_PATCH_META_BEGIN
{"summary":"Continued","assumptions":[],"no_changes_needed":false}
GRAPH_PATCH_META_END
GRAPH_PATCH_DIFF_BEGIN
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-old
+new
GRAPH_PATCH_DIFF_END
""",
                    17,
                    24,
                )
            ]

    backend = TruncatedPatchBackend()
    provider = MLXLocalProvider(model_path="local/model", max_tokens=64, backend=backend)
    payload, prompt_tokens, completion_tokens = await provider.complete_patch(
        system="patch system",
        user="patch user",
        temperature=0.3,
    )
    assert payload["summary"] == "Continued"
    assert "+new" in payload["patch"]
    assert backend.stream_calls == 3
    assert backend.temperatures == [0.3, 0.0, 0.0]
    assert prompt_tokens == 39
    assert completion_tokens == 96
    assert "PATCH ENVELOPE TRUNCATION CONTINUATION" in backend.tokenizer.messages[0][
        "content"
    ]
    assert backend.tokenizer.template_calls[-1][3] is False
    provider.close()


def test_patch_continuation_plan_discards_partial_metadata() -> None:
    content = """analysis
GRAPH_PATCH_V1
GRAPH_PATCH_META_BEGIN
{"summary":"cut off"
"""
    prefix, instruction = MLXLocalProvider._patch_continuation_plan(content) or ("", "")
    assert prefix.endswith("GRAPH_PATCH_V1\n")
    assert "cut off" not in prefix
    assert instruction.startswith("Start with GRAPH_PATCH_META_BEGIN")


def test_patch_continuation_plan_keeps_complete_partial_diff_lines() -> None:
    content = """GRAPH_PATCH_V1
GRAPH_PATCH_META_BEGIN
{"summary":"Partial","assumptions":[],"no_changes_needed":false}
GRAPH_PATCH_META_END
GRAPH_PATCH_DIFF_BEGIN
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-old
+ne"""
    prefix, instruction = MLXLocalProvider._patch_continuation_plan(content) or ("", "")
    assert prefix.endswith("-old\n")
    assert "+ne" not in prefix
    assert "Continue the raw unified diff" in instruction


@pytest.mark.asyncio
async def test_mlx_provider_patch_failure_reports_bounded_diagnostics_only() -> None:
    class BrokenPatchBackend(FakeMLXLMBackend):
        def stream_generate(self, model, tokenizer, prompt, *, max_tokens, sampler):
            del model, tokenizer, prompt, sampler
            self.stream_calls += 1
            return [FakeResponse("TOP-SECRET-RAW-BUT-MALFORMED", 7, max_tokens)]

    backend = BrokenPatchBackend()
    provider = MLXLocalProvider(model_path="local/model", max_tokens=64, backend=backend)
    with pytest.raises(ProviderError) as exc_info:
        await provider.complete_patch(system="patch", user="user")
    message = str(exc_info.value)
    assert "after one bounded recovery attempt" in message
    assert "likely_truncated=true" in message
    assert "completion_tokens=64" in message
    assert "TOP-SECRET" not in message
    assert backend.stream_calls == 2
    provider.close()


@pytest.mark.asyncio
async def test_mlx_seeded_generation_resets_mlx_rng(monkeypatch) -> None:
    seeds: list[int] = []
    core = types.ModuleType("mlx.core")
    core.random = types.SimpleNamespace(seed=lambda value: seeds.append(int(value)))
    mlx = types.ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    backend = FakeMLXLMBackend()
    provider = MLXLocalProvider(
        model_path="local/model",
        max_tokens=64,
        backend=backend,
    )
    payload, _, _ = await provider.complete_json_seeded(
        system="system",
        user="user",
        temperature=0.0,
        seed=123456,
    )
    assert payload == {"ok": True}
    assert seeds == [123456]
    provider.close()
