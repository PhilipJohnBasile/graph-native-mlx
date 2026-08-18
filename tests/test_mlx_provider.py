from dataclasses import dataclass

import pytest

from graph_model.mlx_native.provider import MLXLocalProvider, _load_call_kwargs
from graph_model.provider import ProviderError


class FakeTokenizer:
    def __init__(self, *, template_works: bool = True) -> None:
        self.template_works = template_works
        self.messages = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        if not self.template_works:
            raise ValueError("missing template")
        return "CHAT-TEMPLATE-PROMPT"

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

    def load_model(self, model_path, *, adapter_path, revision, lazy, trust_remote_code):
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
    assert backend.last_sampler == (0.2, 0.9, 0.05, 20)
    assert provider.loaded is True


@pytest.mark.asyncio
async def test_mlx_provider_has_a_role_delimited_template_fallback() -> None:
    backend = FakeMLXLMBackend(template_works=False)
    provider = MLXLocalProvider(model_path="local/model", max_tokens=64, backend=backend)
    payload, _, _ = await provider.complete_json(system="SYSTEM", user="USER")
    assert payload == {"ok": True}
    assert "SYSTEM INSTRUCTIONS:\nSYSTEM" in backend.last_prompt
    assert "USER INPUT:\nUSER" in backend.last_prompt


@pytest.mark.asyncio
async def test_mlx_provider_rejects_empty_generation() -> None:
    provider = MLXLocalProvider(
        model_path="local/model",
        max_tokens=64,
        backend=FakeMLXLMBackend(empty=True),
    )
    with pytest.raises(ProviderError, match="empty response"):
        await provider.complete_json(system="system", user="user")


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
