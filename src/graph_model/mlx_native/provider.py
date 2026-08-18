from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from importlib import metadata
import os
from pathlib import Path
from threading import RLock, get_ident
from typing import Any, Callable, Iterable, Protocol, Sequence, TypeVar

from graph_model.models import RunState
from graph_model.provider import ModelProvider, ProviderError, _parse_json_object

from .hidden_state import (
    DEFAULT_HIDDEN_FEATURE_SIZE,
    DEFAULT_HIDDEN_LAYER_SPECS,
    DEFAULT_HIDDEN_MAX_INPUT_TOKENS,
    DEFAULT_HIDDEN_POOLING,
    DEFAULT_HIDDEN_PROJECTION_SEED,
    HiddenStateArtifactStore,
    HiddenStateCaptureConfig,
    HiddenStateObservation,
    capture_from_raw_hidden,
    model_fingerprint,
    normalize_layer_specs,
    policy_state_prompt,
)
from .qwen_hidden import RawHiddenState, extract_qwen_hidden_state


_T = TypeVar("_T")


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


class QwenHiddenStateBackend(Protocol):
    @property
    def identity(self) -> str: ...

    def extract_hidden_state(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        *,
        max_tokens: int,
        layer_specs: Sequence[str],
        pooling: str,
    ) -> RawHiddenState: ...


def _load_call_kwargs(
    load_function: Any,
    *,
    adapter_path: str | None,
    revision: str | None,
    lazy: bool,
    trust_remote_code: bool,
) -> dict[str, Any]:
    """Build kwargs accepted by MLX-LM 0.31.3 and newer upstream APIs."""

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
        kwargs["tokenizer_config"] = {"trust_remote_code": True}
    return kwargs


class NativeMLXLMBackend:
    def __init__(self) -> None:
        try:
            import mlx_lm
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:  # pragma: no cover - Apple Silicon only
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


class NativeQwenHiddenStateBackend:
    @property
    def identity(self) -> str:
        return "mlx-qwen-selected-hidden-v1"

    def extract_hidden_state(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        *,
        max_tokens: int,
        layer_specs: Sequence[str],
        pooling: str,
    ) -> RawHiddenState:
        return extract_qwen_hidden_state(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            layer_specs=layer_specs,
            pooling=pooling,
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
    """In-process MLX-LM provider with optional Qwen hidden-state capture.

    Generation and hidden-state extraction are serialized because model/cache objects are mutable
    during decoding. Projected hidden features are hash-addressed; raw prompts and raw hidden
    tensors are not persisted.
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
        hidden_backend: QwenHiddenStateBackend | None = None,
        hidden_capture_enabled: bool = False,
        hidden_artifact_root: str | Path = ".graph-model/hidden-states",
        hidden_feature_size: int = DEFAULT_HIDDEN_FEATURE_SIZE,
        hidden_max_input_tokens: int = DEFAULT_HIDDEN_MAX_INPUT_TOKENS,
        hidden_layer_specs: Sequence[str] | str = DEFAULT_HIDDEN_LAYER_SPECS,
        hidden_pooling: str = DEFAULT_HIDDEN_POOLING,
        hidden_projection_seed: int = DEFAULT_HIDDEN_PROJECTION_SEED,
        hidden_cache_max_entries: int = 1_024,
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
        if hidden_cache_max_entries < 0:
            raise ValueError("hidden_cache_max_entries must be >= 0")
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
        self.hidden_capture_enabled = bool(hidden_capture_enabled)
        self.hidden_config = HiddenStateCaptureConfig(
            feature_size=int(hidden_feature_size),
            max_input_tokens=int(hidden_max_input_tokens),
            layer_specs=normalize_layer_specs(hidden_layer_specs),
            pooling=str(hidden_pooling).strip().lower(),
            projection_seed=int(hidden_projection_seed),
        )
        self.hidden_store = HiddenStateArtifactStore(hidden_artifact_root)
        self.hidden_cache_max_entries = int(hidden_cache_max_entries)
        self._hidden_backend = hidden_backend
        self._hidden_cache: OrderedDict[
            tuple[str, str, str], HiddenStateObservation
        ] = OrderedDict()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._lock = RLock()
        self._lifecycle_lock = RLock()
        self._closed = False
        self._affinity_thread_id: int | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="graph-model-mlx-affinity",
        )

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
            default_temperature=float(os.getenv("GRAPH_MODEL_MLX_TEMPERATURE", "0.1")),
            top_p=float(os.getenv("GRAPH_MODEL_MLX_TOP_P", "1.0")),
            min_p=float(os.getenv("GRAPH_MODEL_MLX_MIN_P", "0.0")),
            top_k=int(os.getenv("GRAPH_MODEL_MLX_TOP_K", "0")),
            lazy_load_weights=_env_bool("GRAPH_MODEL_MLX_LAZY_LOAD", False),
            trust_remote_code=_env_bool("GRAPH_MODEL_MLX_TRUST_REMOTE_CODE", False),
            hidden_capture_enabled=_env_bool("GRAPH_MODEL_MLX_CAPTURE_HIDDEN", False),
            hidden_artifact_root=os.getenv(
                "GRAPH_MODEL_MLX_HIDDEN_ROOT", ".graph-model/hidden-states"
            ),
            hidden_feature_size=int(
                os.getenv(
                    "GRAPH_MODEL_MLX_HIDDEN_FEATURE_SIZE",
                    str(DEFAULT_HIDDEN_FEATURE_SIZE),
                )
            ),
            hidden_max_input_tokens=int(
                os.getenv(
                    "GRAPH_MODEL_MLX_HIDDEN_MAX_INPUT_TOKENS",
                    str(DEFAULT_HIDDEN_MAX_INPUT_TOKENS),
                )
            ),
            hidden_layer_specs=os.getenv(
                "GRAPH_MODEL_MLX_POLICY_LAYERS",
                ",".join(DEFAULT_HIDDEN_LAYER_SPECS),
            ),
            hidden_pooling=os.getenv(
                "GRAPH_MODEL_MLX_POLICY_POOLING", DEFAULT_HIDDEN_POOLING
            ),
            hidden_projection_seed=int(
                os.getenv(
                    "GRAPH_MODEL_MLX_HIDDEN_PROJECTION_SEED",
                    str(DEFAULT_HIDDEN_PROJECTION_SEED),
                )
            ),
            hidden_cache_max_entries=int(
                os.getenv("GRAPH_MODEL_MLX_HIDDEN_CACHE_ENTRIES", "1024")
            ),
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
            "execution": "dedicated-single-worker",
            "hidden_capture": str(self.hidden_capture_enabled).lower(),
            "hidden_schema_hash": self.hidden_config.schema_hash,
            "hidden_feature_size": str(self.hidden_config.feature_size),
            "hidden_layers": ",".join(self.hidden_config.layer_specs),
            "hidden_pooling": self.hidden_config.pooling,
            "hidden_cache_entries": str(self.hidden_cache_max_entries),
        }

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _execute_on_affinity(
        self,
        function: Callable[..., _T],
        args: tuple[Any, ...],
    ) -> _T:
        self._affinity_thread_id = get_ident()
        return function(*args)

    def run_on_affinity(self, function: Callable[..., _T], *args: Any) -> _T:
        """Run every MLX model and policy operation on one stable worker thread."""

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MLX provider is closed")
            executor = self._executor
        if get_ident() == self._affinity_thread_id:
            return function(*args)
        return executor.submit(self._execute_on_affinity, function, tuple(args)).result()

    async def _run_on_affinity_async(
        self,
        function: Callable[..., _T],
        *args: Any,
    ) -> _T:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MLX provider is closed")
            executor = self._executor
        if get_ident() == self._affinity_thread_id:
            return function(*args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            self._execute_on_affinity,
            function,
            tuple(args),
        )

    def load(self) -> None:
        self.run_on_affinity(self._load_sync)

    def _load_sync(self) -> None:
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

    @property
    def hidden_state_identity(self) -> str:
        backend_identity = (
            self._hidden_backend.identity
            if self._hidden_backend is not None
            else "mlx-qwen-selected-hidden-v1"
        )
        return (
            f"{backend_identity}:{model_fingerprint(self.identity)[:16]}:"
            f"{self.hidden_config.schema_hash[:16]}"
        )

    def _ensure_hidden_backend(self) -> QwenHiddenStateBackend:
        if self._hidden_backend is None:
            self._hidden_backend = NativeQwenHiddenStateBackend()
        return self._hidden_backend

    def capture_policy_hidden(
        self,
        *,
        state: RunState,
        node_id: str,
        decision_type: str,
    ) -> HiddenStateObservation:
        if not state.run_id.strip():
            raise ValueError("run_id is required for hidden-state capture")
        if not state.task.strip():
            raise ValueError("task is required for hidden-state capture")
        snapshot = state.model_copy(deep=True)
        return self.run_on_affinity(
            self._capture_policy_hidden_sync,
            snapshot,
            node_id,
            decision_type,
        )

    def _capture_policy_hidden_sync(
        self,
        state: RunState,
        node_id: str,
        decision_type: str,
    ) -> HiddenStateObservation:
        with self._lock:
            model, tokenizer = self._ensure_loaded()
            system, user = policy_state_prompt(
                state,
                node_id=node_id,
                decision_type=decision_type,
            )
            prompt = self._render_hidden_prompt(
                tokenizer,
                system=system,
                user=user,
            )
            prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            fingerprint = model_fingerprint(self.identity)
            cache_key = (prompt_digest, fingerprint, self.hidden_config.schema_hash)
            cached = self._hidden_cache.get(cache_key)
            if cached is not None:
                self._hidden_cache.move_to_end(cache_key)
                return HiddenStateObservation(
                    features=cached.features,
                    reference=cached.reference,
                    cache_hit=True,
                )
            raw = self._ensure_hidden_backend().extract_hidden_state(
                model,
                tokenizer,
                prompt,
                max_tokens=self.hidden_config.max_input_tokens,
                layer_specs=self.hidden_config.layer_specs,
                pooling=self.hidden_config.pooling,
            )
            capture = capture_from_raw_hidden(
                raw,
                task=state.task,
                model_identity=self.identity,
                config=self.hidden_config,
            )
            reference = self.hidden_store.write(capture)
            observation = HiddenStateObservation(
                features=capture.features,
                reference=reference,
            )
            if self.hidden_cache_max_entries > 0:
                self._hidden_cache[cache_key] = observation
                self._hidden_cache.move_to_end(cache_key)
                while len(self._hidden_cache) > self.hidden_cache_max_entries:
                    self._hidden_cache.popitem(last=False)
            return observation

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        return await self._run_on_affinity_async(
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
                        _response_value(response, "generation_tokens", completion_tokens)
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

    def _release_sync(self) -> None:
        with self._lock:
            self._hidden_cache.clear()
            self._hidden_backend = None
            self._model = None
            self._tokenizer = None

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
        if get_ident() == self._affinity_thread_id:
            self._release_sync()
        else:
            executor.submit(self._execute_on_affinity, self._release_sync, ()).result()
        executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "MLXLocalProvider":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()

    @staticmethod
    def _render_messages(
        tokenizer: Any,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> str | None:
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            return None
        try:
            rendered = apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            return rendered if isinstance(rendered, str) and rendered else None
        except Exception:  # noqa: BLE001 - tokenizer templates raise backend-specific errors
            return None

    @classmethod
    def _render_hidden_prompt(
        cls,
        tokenizer: Any,
        *,
        system: str,
        user: str,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        rendered = cls._render_messages(
            tokenizer,
            messages,
            add_generation_prompt=False,
        )
        if rendered:
            return rendered
        return f"SYSTEM INSTRUCTIONS:\n{system}\n\nGRAPH STATE:\n{user}\n"

    @classmethod
    def _render_prompt(cls, tokenizer: Any, *, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        rendered = cls._render_messages(
            tokenizer,
            messages,
            add_generation_prompt=True,
        )
        if rendered:
            return rendered
        return (
            "SYSTEM INSTRUCTIONS:\n"
            f"{system}\n\n"
            "USER INPUT:\n"
            f"{user}\n\n"
            "ASSISTANT JSON:\n"
        )
