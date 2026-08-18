from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

from graph_model.provider import ProviderError


@dataclass(frozen=True)
class RawHiddenState:
    values: tuple[float, ...]
    source: str
    layer_labels: tuple[str, ...]
    pooling: str
    token_count: int
    prompt_sha256: str
    model_hidden_size: int


def parse_layer_specs(specs: Sequence[str], layer_count: int) -> tuple[tuple[str, int | None], ...]:
    """Resolve human layer selectors to zero-based decoder-layer indices.

    ``final`` selects the normalized backbone output. Integer values select the raw output of a
    decoder layer, with negative values following Python indexing. Percentages select the decoder
    layer at that depth. Duplicate resolved selections are removed while preserving order.
    """

    if layer_count < 1:
        raise ValueError("layer_count must be positive")
    if not specs:
        raise ValueError("at least one hidden-state layer selector is required")
    resolved: list[tuple[str, int | None]] = []
    seen: set[int | None] = set()
    for raw in specs:
        token = str(raw).strip().lower()
        if token in {"final", "normalized-final"}:
            label, index = "final", None
        elif token.endswith("%"):
            try:
                percent = float(token[:-1])
            except ValueError as exc:
                raise ValueError(f"invalid hidden-state percentage selector {raw!r}") from exc
            if not 0 < percent <= 100:
                raise ValueError(f"hidden-state percentage selector must be in (0, 100]: {raw!r}")
            index = min(layer_count - 1, max(0, math.ceil(layer_count * percent / 100.0) - 1))
            label = f"layer:{index}"
        else:
            try:
                index = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"hidden-state selector must be 'final', an integer, or a percentage: {raw!r}"
                ) from exc
            if index < 0:
                index += layer_count
            if not 0 <= index < layer_count:
                raise ValueError(
                    f"hidden-state layer index {raw!r} is outside 0..{layer_count - 1}"
                )
            label = f"layer:{index}"
        if index not in seen:
            resolved.append((label, index))
            seen.add(index)
    return tuple(resolved)


def _resolve_backbone(model: Any) -> tuple[Any, str]:
    candidates: list[tuple[str, Any]] = []
    for name in ("model", "language_model", "transformer"):
        try:
            value = getattr(model, name, None)
        except Exception:  # noqa: BLE001 - model properties may be backend-specific
            value = None
        if value is not None and value is not model:
            candidates.append((name, value))
    try:
        language_model = getattr(model, "language_model", None)
        nested = getattr(language_model, "model", None) if language_model is not None else None
    except Exception:  # noqa: BLE001
        nested = None
    if nested is not None and nested is not model:
        candidates.insert(0, ("language_model.model", nested))

    candidates.append(("self", model))
    visited: set[int] = set()
    for source, candidate in candidates:
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if callable(candidate) and hasattr(candidate, "embed_tokens") and hasattr(candidate, "norm"):
            layers = getattr(candidate, "pipeline_layers", None) or getattr(candidate, "layers", None)
            if isinstance(layers, (list, tuple)) and layers:
                return candidate, source
    raise ProviderError(
        "the loaded MLX-LM model does not expose a supported hidden-state backbone; "
        "expected a callable model/model.language_model with embed_tokens, layers, and norm"
    )


def _token_ids(tokenizer: Any, prompt: str, max_tokens: int) -> list[int]:
    if max_tokens < 8:
        raise ValueError("hidden-state max_tokens must be >= 8")
    try:
        encoded = tokenizer.encode(prompt, add_special_tokens=True)
    except TypeError:
        encoded = tokenizer.encode(prompt)
    try:
        raw = encoded.tolist()
    except AttributeError:
        raw = list(encoded)
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    ids = [int(item) for item in raw]
    if not ids:
        raise ProviderError("tokenizer produced no tokens for the policy-state prompt")
    if len(ids) <= max_tokens:
        return ids
    keep_head = max(1, max_tokens // 4)
    keep_tail = max_tokens - keep_head
    return ids[:keep_head] + ids[-keep_tail:]


def _pool_and_normalize(mx: Any, hidden: Any, pooling: str) -> Any:
    if len(hidden.shape) != 3 or int(hidden.shape[0]) != 1:
        raise ProviderError(
            f"hidden-state backbone returned shape {tuple(hidden.shape)}, expected [1, tokens, hidden]"
        )
    token_matrix = hidden[0].astype(mx.float32)
    if pooling == "last-token":
        vector = token_matrix[-1]
    elif pooling == "mean":
        vector = mx.mean(token_matrix, axis=0)
    elif pooling == "mean-last":
        vector = mx.concatenate([mx.mean(token_matrix, axis=0), token_matrix[-1]], axis=-1)
    else:
        raise ValueError(f"unsupported hidden-state pooling {pooling!r}")
    denominator = mx.sqrt(mx.sum(vector * vector) + mx.array(1e-12, dtype=mx.float32))
    return vector / denominator


def _call_layer(layer: Any, hidden: Any, mask: Any) -> Any:
    try:
        return layer(hidden, mask=mask, cache=None)
    except TypeError:
        try:
            return layer(hidden, mask, None)
        except TypeError as exc:
            raise ProviderError(
                f"decoder layer {type(layer).__name__} does not support the MLX-LM layer call contract"
            ) from exc


def _manual_selected_forward(
    *,
    mx: Any,
    backbone: Any,
    inputs: Any,
    selections: Sequence[tuple[str, int | None]],
) -> list[tuple[str, Any]]:
    try:
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    except ImportError as exc:  # pragma: no cover - runs only on MLX hosts
        raise ProviderError("mlx-lm model mask helpers are unavailable") from exc

    layers = getattr(backbone, "pipeline_layers", None) or getattr(backbone, "layers", None)
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ProviderError("hidden-state backbone does not expose decoder layers")
    pipeline_size = int(getattr(backbone, "pipeline_size", 1) or 1)
    if pipeline_size != 1:
        raise ProviderError(
            "selected intermediate hidden states are not supported with pipeline-parallel MLX-LM; "
            "use GRAPH_MODEL_MLX_POLICY_LAYERS=final"
        )

    hidden = backbone.embed_tokens(inputs)
    has_linear = any(bool(getattr(layer, "is_linear", False)) for layer in layers)
    has_full_attention = any(not bool(getattr(layer, "is_linear", False)) for layer in layers)
    attention_mask = create_attention_mask(hidden, None) if has_full_attention else None
    ssm_mask = create_ssm_mask(hidden, None) if has_linear else None

    wanted = {index for _, index in selections if index is not None}
    captured: dict[int, Any] = {}
    for index, layer in enumerate(layers):
        mask = ssm_mask if bool(getattr(layer, "is_linear", False)) else attention_mask
        hidden = _call_layer(layer, hidden, mask)
        if index in wanted:
            captured[index] = hidden

    final_hidden = backbone.norm(hidden)
    output: list[tuple[str, Any]] = []
    for label, index in selections:
        if index is None:
            output.append((label, final_hidden))
        elif index in captured:
            output.append((label, captured[index]))
        else:  # pragma: no cover - defensive
            raise ProviderError(f"selected hidden-state layer {index} was not captured")
    return output


def extract_qwen_hidden_state(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int = 512,
    layer_specs: Sequence[str] = ("final",),
    pooling: str = "last-token",
) -> RawHiddenState:
    """Extract selected Qwen/MLX backbone representations without changing model weights.

    The final-only path calls the model's own normalized backbone directly. Intermediate layers use
    the same public layer/mask call structure employed by current MLX-LM Qwen3.5/Qwen3-Next model
    implementations, with no cache and pipeline size one.
    """

    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - executed on Apple Silicon
        raise ProviderError("hidden-state extraction requires the mlx package") from exc

    backbone, source = _resolve_backbone(model)
    layers = getattr(backbone, "pipeline_layers", None) or getattr(backbone, "layers", None)
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ProviderError("hidden-state backbone exposes no decoder layers")
    selections = parse_layer_specs(layer_specs, len(layers))
    ids = _token_ids(tokenizer, prompt, max_tokens)
    inputs = mx.array([ids], dtype=mx.int32)

    try:
        if len(selections) == 1 and selections[0][1] is None:
            try:
                hidden_outputs = [(selections[0][0], backbone(inputs, cache=None))]
            except TypeError:
                hidden_outputs = [(selections[0][0], backbone(inputs))]
        else:
            hidden_outputs = _manual_selected_forward(
                mx=mx,
                backbone=backbone,
                inputs=inputs,
                selections=selections,
            )
        pooled = [_pool_and_normalize(mx, hidden, pooling) for _, hidden in hidden_outputs]
        model_hidden_size = int(hidden_outputs[0][1].shape[-1])
        vector = pooled[0] if len(pooled) == 1 else mx.concatenate(pooled, axis=-1)
        mx.eval(vector)
        values = tuple(float(item) for item in vector.tolist())
    except ProviderError:
        raise
    except Exception as exc:  # noqa: BLE001 - backend exceptions differ by MLX/model version
        raise ProviderError(f"MLX hidden-state extraction failed: {type(exc).__name__}: {exc}") from exc

    if not values or any(not math.isfinite(item) for item in values):
        raise ProviderError("MLX hidden-state extraction produced an empty or non-finite vector")
    return RawHiddenState(
        values=values,
        source=source,
        layer_labels=tuple(label for label, _ in selections),
        pooling=pooling,
        token_count=len(ids),
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        model_hidden_size=model_hidden_size,
    )
