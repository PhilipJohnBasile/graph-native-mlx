from __future__ import annotations

import math
from importlib import metadata
from dataclasses import dataclass
from typing import Protocol, Sequence


class MLXUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaskedDecision:
    selected_index: int
    probabilities: tuple[float, ...]


class DecisionBackend(Protocol):
    @property
    def identity(self) -> str: ...

    def masked_softmax_argmax(
        self,
        logits: Sequence[float],
        mask: Sequence[bool],
    ) -> MaskedDecision: ...


def _validate(logits: Sequence[float], mask: Sequence[bool]) -> None:
    if not logits:
        raise ValueError("masked decision requires at least one logit")
    if len(logits) != len(mask):
        raise ValueError("logits and mask must have the same length")
    if not any(mask):
        raise ValueError("masked decision requires at least one allowed choice")


class PythonDecisionBackend:
    """Numerically equivalent test backend. Production `mlx` mode does not select this silently."""

    @property
    def identity(self) -> str:
        return "python-reference"

    def masked_softmax_argmax(
        self,
        logits: Sequence[float],
        mask: Sequence[bool],
    ) -> MaskedDecision:
        _validate(logits, mask)
        masked = [float(value) if allowed else -math.inf for value, allowed in zip(logits, mask)]
        maximum = max(masked)
        exponentials = [math.exp(value - maximum) if math.isfinite(value) else 0.0 for value in masked]
        denominator = sum(exponentials)
        probabilities = tuple(value / denominator for value in exponentials)
        selected = max(range(len(probabilities)), key=probabilities.__getitem__)
        return MaskedDecision(selected_index=selected, probabilities=probabilities)


class MLXDecisionBackend:
    """Runs hard masking, softmax, and argmax as native MLX tensor operations."""

    def __init__(self) -> None:
        try:
            import mlx
            import mlx.core as mx
        except ImportError as exc:  # pragma: no cover - exercised on Apple Silicon
            raise MLXUnavailableError(
                "MLX is not installed. On Apple Silicon install the `mlx` extra: "
                "python -m pip install -e '.[mlx]'"
            ) from exc
        self.mx = mx
        self._version = str(getattr(mlx, "__version__", "unknown"))
        if self._version == "unknown":
            try:
                self._version = metadata.version("mlx")
            except metadata.PackageNotFoundError:
                pass

    @property
    def identity(self) -> str:
        return f"mlx-core-{self._version}"

    def masked_softmax_argmax(
        self,
        logits: Sequence[float],
        mask: Sequence[bool],
    ) -> MaskedDecision:
        _validate(logits, mask)
        mx = self.mx
        values = mx.array(list(logits), dtype=mx.float32)
        allowed = mx.array(list(mask), dtype=mx.bool_)
        masked = mx.where(allowed, values, mx.full_like(values, -1e9))
        probabilities = mx.softmax(masked, axis=-1)
        selected = mx.argmax(probabilities, axis=-1)
        mx.eval(probabilities, selected)
        return MaskedDecision(
            selected_index=int(selected.item()),
            probabilities=tuple(float(value) for value in probabilities.tolist()),
        )
