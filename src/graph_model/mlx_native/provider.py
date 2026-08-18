from __future__ import annotations

import asyncio
import inspect
from importlib import metadata
import os
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Protocol

from graph_model.provider import ModelProvider, ProviderError, _parse_json_object


class MLXLMBackend(Protocol):
    @property
    def identity(self) -> str: ...

    def load_model(
        self,
        model_path: str,
        *,
        adapter_path: str | None,
        revision: str | None,
        lazy: bool,
        trust_remote_code: bool,
    ) -> tuple[Any, Any]: ...

    def make_sampler(
        self,
        *,
        temperature: float,
        top_p: float,
        min_p: float,
        top_k: int,
    ) -> Any: ...

    def stream_generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        *,
        max_tokens: int,
        sampler: Any,
    ) -> Iterable[Any]: ...


def _load_call_kwargs(
    load_function: Any,
    *,
    adapter_path: str | None,
    revision: str | None,
    lazy: bool,
    trust_remote_code: bool,
) -> dict[str, Any]:
    """Build kwargs accepted by both MLX-LM 0.31.3 and newer upstream APIs."""

    try:
        parameters = inspect.signature(load_function).parameters
    except (TypeError, ValueError):
        parameters = {}
    kwargs: dict[str, Any] = {
        "adapter_path": adapter_path,
        "lazy": lazy,
    }
    if not parameters or "revision" in parameters:
        kwargs["revision"] = revision
    if not parameters or "trust_remote_code" in parameters:
        kwargs["trust_remote_code"] = trust_remote_code
    elif trust_remote_code and "tokenizer_config" in parameters:
        # MLX-LM 0.31.3 exposes remote-code trust through tokenizer kwargs only.
        kwargs["tokenizer_config"] = {"trust_remote_code": True}
    return kwargs


class NativeMLXLMBackend:
    def __init__(self) -> None:
        try:
            import mlx_lm
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:  # pragma: no cover - executed on Apple Silicon
            raise ProviderError(
                "mlx-lm is not installed. Install the MLX extra on Apple Silicon: "
                "python -m pip install -e '.[mlx]'"
            ) from exc
        self._version = str(getattr(mlx_lm, "__version__", "unknown"))
        if self._version == "unknown":
            try:
                self._version = metadata.version("mlx-lm")
            except metadata.PackageNotFoundError:
                pass
        self._load = load
        self._stream_generate = stream_generate
        self._make_sampler = make_sampler

    @property
    def identity(self) -> str:
        return f"mlx-lm-{self._version}"

    def load_model(
        self,
        model_path: str,
        *,
        adapter_path: str | None,
        revision: str | None,
        lazy: bool,
        trust_remote_code: bool,
    ) -> tuple[Any, Any]:
        kwargs = _load_call_kwargs(
            self._load,
            adapter_path=adapter_path,
            revision=revision,
            lazy=lazy,
            trust_remote_code=trust_remote_code,
        )
        return self._load(model_path, **kwargs)

    def make_sampler(
        self,
        *,
        temperature: float,
        top_p: float,
        min_p: float,
        top_k: int,
    ) -> Any:
        return self._make_sampler(
            temp=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        )

    def stream_generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        *,
        max_tokens: int,
        sampler: Any,
    ) -> Iterable[Any]:
        return self._stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _token_count(tokenizer: Any, text: str, *, add_special_tokens: bool) -> int:
    try:
        tokens = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        tokens = tokenizer.encode(text)
    except Exception:
        return max(1, len(text) // 4)
    try:
        return int(len(tokens))
    except TypeError:
        return max(1, len(text) // 4)


def _response_value(response: Any, name: str, default: Any) -> Any:
    if isinstance(response, dict):
        return response.get(name, default)
    return getattr(response, name, default)


class MLXLocalProvider(ModelProvider):
    """In-process MLX-LM provider that keeps one model/tokenizer resident.

    Generation is serialized by default because MLX-LM model/cache objects are mutable during
    decoding. Async graph execution remains responsive by running the blocking generation in a
    worker thread.
    """

    def __init__(
        self,
        *,
        model_path: str,
        adapter_path: str | None = None,
        revision: str | None = None,
        max_tokens: int = 8_192,
        default_temperature: float = 0.1,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        lazy_load_weights: bool = False,
        trust_remote_code: bool = False,
        backend: MLXLMBackend | None = None,
    ) -> None:
        if not model_path.strip():
            raise ValueError("model_path must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if default_temperature < 0:
            raise ValueError("default_temperature must be >= 0")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be in [0, 1]")
        if not 0.0 <= min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.revision = revision
        self.max_tokens = int(max_tokens)
        self.default_temperature = float(default_temperature)
        self.top_p = float(top_p)
        self.min_p = float(min_p)
        self.top_k = int(top_k)
        self.lazy_load_weights = bool(lazy_load_weights)
        self.trust_remote_code = bool(trust_remote_code)
        self.backend = backend or NativeMLXLMBackend()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._lock = RLock()

    @classmethod
    def from_env(cls) -> "MLXLocalProvider":
        model_path = os.getenv("GRAPH_MODEL_MLX_MODEL")
        if not model_path:
            raise ProviderError(
                "GRAPH_MODEL_MLX_MODEL is required for --provider mlx; set it to a local "
                "MLX model directory or Hugging Face repository ID"
            )
        return cls(
            model_path=model_path,
            adapter_path=os.getenv("GRAPH_MODEL_MLX_ADAPTER_PATH") or None,
            revision=os.getenv("GRAPH_MODEL_MLX_REVISION") or None,
            max_tokens=int(os.getenv("GRAPH_MODEL_MLX_MAX_TOKENS", "8192")),
            default_temperature=float(
                os.getenv("GRAPH_MODEL_MLX_TEMPERATURE", "0.1")
            ),
            top_p=float(os.getenv("GRAPH_MODEL_MLX_TOP_P", "1.0")),
            min_p=float(os.getenv("GRAPH_MODEL_MLX_MIN_P", "0.0")),
            top_k=int(os.getenv("GRAPH_MODEL_MLX_TOP_K", "0")),
            lazy_load_weights=_env_bool("GRAPH_MODEL_MLX_LAZY_LOAD", False),
            trust_remote_code=_env_bool("GRAPH_MODEL_MLX_TRUST_REMOTE_CODE", False),
        )

    @property
    def identity(self) -> dict[str, str]:
        model_identity = self.model_path
        model_path = Path(self.model_path).expanduser()
        if model_path.exists():
            model_identity = str(model_path.resolve())
        adapter_identity = self.adapter_path or ""
        if self.adapter_path:
            adapter_path = Path(self.adapter_path).expanduser()
            if adapter_path.exists():
                adapter_identity = str(adapter_path.resolve())
        return {
            "kind": "mlx-local",
            "backend": self.backend.identity,
            "model": model_identity,
            "adapter": adapter_identity,
            "revision": self.revision or "",
            "max_tokens": str(self.max_tokens),
            "temperature": f"{self.default_temperature:g}",
            "top_p": f"{self.top_p:g}",
            "min_p": f"{self.min_p:g}",
            "top_k": str(self.top_k),
            "lazy_load": str(self.lazy_load_weights).lower(),
            "trust_remote_code": str(self.trust_remote_code).lower(),
        }

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        with self._lock:
            self._ensure_loaded()

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            try:
                self._model, self._tokenizer = self.backend.load_model(
                    self.model_path,
                    adapter_path=self.adapter_path,
                    revision=self.revision,
                    lazy=self.lazy_load_weights,
                    trust_remote_code=self.trust_remote_code,
                )
            except Exception as exc:
                raise ProviderError(
                    f"failed to load MLX model {self.model_path!r}: {exc}"
                ) from exc
        return self._model, self._tokenizer

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        return await asyncio.to_thread(
            self._complete_json_sync,
            system,
            user,
            self.default_temperature if temperature is None else float(temperature),
        )

    def _complete_json_sync(
        self,
        system: str,
        user: str,
        temperature: float,
    ) -> tuple[dict[str, Any], int, int]:
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        with self._lock:
            model, tokenizer = self._ensure_loaded()
            prompt = self._render_prompt(tokenizer, system=system, user=user)
            sampler = self.backend.make_sampler(
                temperature=temperature,
                top_p=self.top_p,
                min_p=self.min_p,
                top_k=self.top_k,
            )
            pieces: list[str] = []
            prompt_tokens = 0
            completion_tokens = 0
            try:
                for response in self.backend.stream_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=self.max_tokens,
                    sampler=sampler,
                ):
                    text = _response_value(response, "text", "")
                    if isinstance(text, str):
                        pieces.append(text)
                    prompt_tokens = int(
                        _response_value(response, "prompt_tokens", prompt_tokens) or prompt_tokens
                    )
                    completion_tokens = int(
                        _response_value(
                            response,
                            "generation_tokens",
                            completion_tokens,
                        )
                        or completion_tokens
                    )
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(f"MLX generation failed: {exc}") from exc

            content = "".join(pieces).strip()
            if not content:
                raise ProviderError("MLX model generated an empty response")
            parsed = _parse_json_object(content)
            if prompt_tokens <= 0:
                prompt_tokens = _token_count(tokenizer, prompt, add_special_tokens=True)
            if completion_tokens <= 0:
                completion_tokens = _token_count(
                    tokenizer,
                    content,
                    add_special_tokens=False,
                )
            return parsed, prompt_tokens, completion_tokens

    @staticmethod
    def _render_prompt(tokenizer: Any, *, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                rendered = apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str) and rendered:
                    return rendered
            except Exception:  # noqa: BLE001 - tokenizer templates raise backend-specific errors
                # Some community tokenizers ship incomplete templates. The explicit fallback keeps
                # role boundaries visible without depending on model-specific special tokens.
                pass
        return (
            "SYSTEM INSTRUCTIONS:\n"
            f"{system}\n\n"
            "USER INPUT:\n"
            f"{user}\n\n"
            "ASSISTANT JSON:\n"
        )
