from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Protocol, Sequence

from graph_model.models import RunState
from graph_model.paired_eval import PairedEvaluationConfig, canonicalize_prompt_value
from graph_model.provider import ProviderError

HIDDEN_STATE_FORMAT_VERSION = 1
HIDDEN_STATE_FORMAT = "graph-native-hidden-state-v1"
HIDDEN_STATE_EXTRACTOR_VERSION = "qwen-selected-layers-countsketch-v1"
DEFAULT_HIDDEN_FEATURE_SIZE = 256
DEFAULT_HIDDEN_MAX_INPUT_TOKENS = 2_048
DEFAULT_HIDDEN_LAYER_SPECS: tuple[str, ...] = ("final",)
DEFAULT_HIDDEN_POOLING = "last-token"
DEFAULT_HIDDEN_PROJECTION_SEED = 47_261_993
MAX_HIDDEN_ARTIFACT_BYTES = 2_000_000
POLICY_STATE_PROMPT_VERSION = "graph-policy-state-v1"
DEFAULT_POLICY_STATE_PROMPT_MAX_CHARS = 32_000


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _finite_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("hidden-state features must contain only finite values")
    return number


def _l2_normalize(values: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(_finite_float(value) for value in values)
    magnitude = math.sqrt(sum(value * value for value in normalized))
    if magnitude <= 1e-12:
        return tuple(0.0 for _ in normalized)
    return tuple(value / magnitude for value in normalized)


def _countsketch_block(
    values: Sequence[float],
    *,
    output_size: int,
    seed: int,
) -> tuple[float, ...]:
    if output_size < 1:
        raise ValueError("hidden-state projection block must be non-empty")
    source = _l2_normalize(values)
    projected = [0.0] * output_size
    # Fixed 64-bit hashes make the projection reproducible across Python processes.
    mask = (1 << 64) - 1
    for index, value in enumerate(source):
        bucket_hash = (
            ((index + 1) * 0x9E3779B185EBCA87)
            ^ (seed * 0xC2B2AE3D27D4EB4F)
        ) & mask
        sign_hash = (
            ((index + 1) * 0x165667B19E3779F9)
            ^ (seed * 0x85EBCA77C2B2AE63)
        ) & mask
        bucket = bucket_hash % output_size
        sign = 1.0 if sign_hash & 1 else -1.0
        projected[bucket] += sign * value
    return _l2_normalize(projected)


def project_hidden_views(
    views: Mapping[str, Sequence[float]],
    *,
    output_size: int = DEFAULT_HIDDEN_FEATURE_SIZE,
    seed: int = DEFAULT_HIDDEN_PROJECTION_SEED,
) -> tuple[float, ...]:
    """Project named Qwen hidden views to one stable fixed-size feature vector.

    Each selected layer receives a disjoint block. This preserves layer identity while keeping
    traces and sidecar weights small enough to inspect, checkpoint, and train repeatedly.
    """

    if output_size < 8:
        raise ValueError("hidden-state feature size must be >= 8")
    if not views:
        raise ValueError("at least one hidden-state view is required")
    names = tuple(sorted(str(name) for name in views))
    base, remainder = divmod(output_size, len(names))
    if base < 1:
        raise ValueError("hidden-state feature size is smaller than the number of views")
    blocks: list[float] = []
    for position, name in enumerate(names):
        block_size = base + (1 if position < remainder else 0)
        values = tuple(views[name])
        if not values:
            raise ValueError(f"hidden-state view {name!r} is empty")
        view_seed = seed ^ int(_sha256_text(name)[:16], 16)
        blocks.extend(_countsketch_block(values, output_size=block_size, seed=view_seed))
    if len(blocks) != output_size:
        raise AssertionError("hidden-state projection produced an invalid feature size")
    return tuple(blocks)


def normalize_layer_specs(values: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = values.split(",")
    normalized = tuple(str(value).strip().lower() for value in values if str(value).strip())
    if not normalized:
        raise ValueError("at least one hidden-state layer selector is required")
    return normalized


def _trim_control_text(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    marker = "\n... graph-control evidence truncated ...\n"
    remaining = max(2, limit - len(marker))
    head = max(1, remaining // 3)
    tail = max(1, remaining - head)
    return text[:head] + marker + text[-tail:]


def _compact_control_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        return _trim_control_text(value, 700)
    if isinstance(value, Mapping):
        selected: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item))[:20]:
            item = value[key]
            name = str(key)
            if name in {"patch", "diff", "stdout", "stderr", "content", "source"}:
                selected[name] = _trim_control_text(item, 1_200)
            else:
                selected[name] = _compact_control_value(item, depth=depth + 1)
        return selected
    if isinstance(value, (list, tuple)):
        return [_compact_control_value(item, depth=depth + 1) for item in value[:12]]
    if isinstance(value, str):
        return _trim_control_text(value, 1_200)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _trim_control_text(value, 700)


def _bounded_control_value(value: Any, *, limit: int = 1_800) -> Any:
    compacted = _compact_control_value(value)
    encoded = json.dumps(
        compacted,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    if len(encoded) <= limit:
        return compacted
    return {
        "sha256": _sha256_text(encoded),
        "excerpt": _trim_control_text(encoded, max(128, limit - 120)),
    }


def policy_state_prompt(
    state: RunState,
    *,
    node_id: str,
    decision_type: str,
    max_chars: int = DEFAULT_POLICY_STATE_PROMPT_MAX_CHARS,
) -> tuple[str, str]:
    """Render a bounded deterministic state prompt for the Qwen policy forward pass.

    The prompt includes the current checkpoint, verifier evidence, progress, and remaining budgets.
    It intentionally excludes runtime identity metadata, secrets, and full repository artifacts.
    """

    if not node_id.strip():
        raise ValueError("node_id is required for a policy-state prompt")
    if not decision_type.strip():
        raise ValueError("decision_type is required for a policy-state prompt")
    if max_chars < 4_096:
        raise ValueError("policy-state prompt max_chars must be >= 4096")

    metrics = state.metrics.model_dump()
    budget = state.budget.model_dump()
    paired = PairedEvaluationConfig.from_state(state)
    if paired.enabled and paired.normalize_timing:
        metrics["elapsed_seconds"] = 0.0
    remaining = {
        "steps": max(0, int(budget["max_steps"]) - state.step_count),
        "llm_calls": max(0, int(budget["max_llm_calls"]) - metrics["llm_calls"]),
        "tool_calls": max(0, int(budget["max_tool_calls"]) - metrics["tool_calls"]),
        "tokens": max(
            0,
            int(budget["max_tokens"])
            - metrics["prompt_tokens"]
            - metrics["completion_tokens"],
        ),
        "seconds": (
            float(budget["max_seconds"])
            if paired.enabled and paired.normalize_timing
            else max(
                0.0,
                float(budget["max_seconds"]) - float(metrics["elapsed_seconds"]),
            )
        ),
    }
    selected_data_keys = (
        "route",
        "difficulty",
        "verdict",
        "repair_count",
        "plan_revision_count",
        "context_ready",
        "plan",
        "apply_report",
        "test_report",
        "review",
        "diagnosis",
        "pending_patch",
        "candidate",
    )
    selected_data = {
        key: _bounded_control_value(state.data[key])
        for key in selected_data_keys
        if key in state.data
    }
    payload: dict[str, Any] = {
        "format": POLICY_STATE_PROMPT_VERSION,
        "task": _trim_control_text(state.task, 6_000),
        "task_sha256": _sha256_text(state.task),
        "graph": {"name": state.graph_name, "version": state.graph_version},
        "decision": {"type": decision_type, "node": node_id},
        "execution": {
            "status": state.status,
            "current_node": state.current_node,
            "step_count": state.step_count,
            "no_progress_count": state.no_progress_count,
            "attempts_at_node": int(state.attempts.get(node_id, 0)),
            "completed_nodes": state.completed_nodes[-12:],
            "edge_counts": dict(sorted(state.edge_counts.items())),
            "metrics": metrics,
            "budget_remaining": remaining,
        },
        "state": selected_data,
        "artifact_keys": sorted(str(key) for key in state.artifacts)[:48],
    }
    payload = canonicalize_prompt_value(payload, state)
    user = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    if len(user) > max_chars:
        # Fail closed to a smaller structured prompt instead of slicing invalid JSON.
        payload = {
            "format": POLICY_STATE_PROMPT_VERSION,
            "task": _trim_control_text(state.task, 3_000),
            "task_sha256": _sha256_text(state.task),
            "graph": payload["graph"],
            "decision": payload["decision"],
            "execution": {
                "status": state.status,
                "current_node": state.current_node,
                "step_count": state.step_count,
                "no_progress_count": state.no_progress_count,
                "attempts_at_node": int(state.attempts.get(node_id, 0)),
                "completed_nodes": state.completed_nodes[-8:],
                "metrics": metrics,
                "budget_remaining": remaining,
            },
            "state": {
                key: _bounded_control_value(value, limit=800)
                for key, value in selected_data.items()
            },
            "artifact_keys": sorted(str(key) for key in state.artifacts)[:24],
            "full_state_sha256": _sha256_text(user),
        }
        payload = canonicalize_prompt_value(payload, state)
        user = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
    if len(user) > max_chars:
        minimal = {
            "format": POLICY_STATE_PROMPT_VERSION,
            "task": _trim_control_text(state.task, 2_000),
            "task_sha256": _sha256_text(state.task),
            "graph": {"name": state.graph_name, "version": state.graph_version},
            "decision": {"type": decision_type, "node": node_id},
            "execution": {
                "status": state.status,
                "current_node": state.current_node,
                "step_count": state.step_count,
                "no_progress_count": state.no_progress_count,
                "budget_remaining": remaining,
            },
            "state_sha256": _sha256_text(user),
        }
        minimal = canonicalize_prompt_value(minimal, state)
        user = json.dumps(
            minimal,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
    if len(user) > max_chars:
        raise ValueError("policy-state prompt could not be bounded safely")

    system = (
        "Encode the graph-controlled software-engineering state for a constrained route, "
        "transition, stopping, value, and cost policy. Preserve task semantics, verifier "
        "evidence, progress, and remaining budget. This is a representation-only forward pass; "
        "do not answer or solve the task."
    )
    return system, user


@dataclass(frozen=True)
class HiddenStateCaptureConfig:
    feature_size: int = DEFAULT_HIDDEN_FEATURE_SIZE
    max_input_tokens: int = DEFAULT_HIDDEN_MAX_INPUT_TOKENS
    layer_specs: tuple[str, ...] = DEFAULT_HIDDEN_LAYER_SPECS
    pooling: str = DEFAULT_HIDDEN_POOLING
    projection_seed: int = DEFAULT_HIDDEN_PROJECTION_SEED
    extractor_version: str = HIDDEN_STATE_EXTRACTOR_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_specs", normalize_layer_specs(self.layer_specs))
        self.validate()

    def validate(self) -> "HiddenStateCaptureConfig":
        if self.feature_size < 8:
            raise ValueError("hidden-state feature_size must be >= 8")
        if self.max_input_tokens < 16:
            raise ValueError("hidden-state max_input_tokens must be >= 16")
        if self.pooling not in {"last-token", "mean", "mean-last"}:
            raise ValueError(
                "hidden-state pooling must be one of: last-token, mean, mean-last"
            )
        if not self.extractor_version.strip():
            raise ValueError("hidden-state extractor_version must not be empty")
        return self

    @property
    def schema_hash(self) -> str:
        return _sha256_bytes(
            _canonical_json_bytes(
                {
                    "format_version": HIDDEN_STATE_FORMAT_VERSION,
                    "format": HIDDEN_STATE_FORMAT,
                    "feature_size": self.feature_size,
                    "max_input_tokens": self.max_input_tokens,
                    "layer_specs": list(self.layer_specs),
                    "pooling": self.pooling,
                    "projection_seed": self.projection_seed,
                    "extractor_version": self.extractor_version,
                    "policy_state_prompt_version": POLICY_STATE_PROMPT_VERSION,
                }
            )
        )


class RawHiddenStateLike(Protocol):
    values: tuple[float, ...]
    source: str
    layer_labels: tuple[str, ...]
    pooling: str
    token_count: int
    prompt_sha256: str
    model_hidden_size: int


@dataclass(frozen=True)
class HiddenStateCapture:
    features: tuple[float, ...]
    model_fingerprint: str
    extractor_schema_hash: str
    raw_hidden_size: int
    raw_vector_size: int
    prompt_tokens: int
    task_sha256: str
    prompt_sha256: str
    core_path: str
    layer_labels: tuple[str, ...]
    pooling: str
    extractor_version: str = HIDDEN_STATE_EXTRACTOR_VERSION

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("hidden-state capture must contain features")
        for value in self.features:
            _finite_float(value)
        for name, value in (
            ("model_fingerprint", self.model_fingerprint),
            ("extractor_schema_hash", self.extractor_schema_hash),
            ("task_sha256", self.task_sha256),
            ("prompt_sha256", self.prompt_sha256),
        ):
            if not _is_digest(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.raw_hidden_size < 1 or self.raw_vector_size < 1:
            raise ValueError("raw hidden-state dimensions must be positive")
        if self.prompt_tokens < 1:
            raise ValueError("prompt_tokens must be positive")
        if not self.core_path or not self.layer_labels or not self.extractor_version.strip():
            raise ValueError("hidden-state source metadata is incomplete")

    def payload(self) -> dict[str, Any]:
        return {
            "format_version": HIDDEN_STATE_FORMAT_VERSION,
            "format": HIDDEN_STATE_FORMAT,
            "extractor_version": self.extractor_version,
            "feature_size": len(self.features),
            "features": list(self.features),
            "model_fingerprint": self.model_fingerprint,
            "extractor_schema_hash": self.extractor_schema_hash,
            "raw_hidden_size": self.raw_hidden_size,
            "raw_vector_size": self.raw_vector_size,
            "prompt_tokens": self.prompt_tokens,
            "task_sha256": self.task_sha256,
            "prompt_sha256": self.prompt_sha256,
            "core_path": self.core_path,
            "layer_labels": list(self.layer_labels),
            "pooling": self.pooling,
        }


@dataclass(frozen=True)
class HiddenStateReference:
    path: str
    sha256: str
    format: str
    feature_size: int
    model_fingerprint: str
    extractor_schema_hash: str
    raw_hidden_size: int
    raw_vector_size: int
    prompt_tokens: int
    task_sha256: str
    prompt_sha256: str
    core_path: str
    layer_labels: tuple[str, ...]
    pooling: str
    extractor_version: str = HIDDEN_STATE_EXTRACTOR_VERSION

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HiddenStateReference":
        try:
            reference = cls(
                path=str(payload["path"]),
                sha256=str(payload["sha256"]),
                format=str(payload.get("format", HIDDEN_STATE_FORMAT)),
                feature_size=int(payload["feature_size"]),
                model_fingerprint=str(payload["model_fingerprint"]),
                extractor_schema_hash=str(payload["extractor_schema_hash"]),
                raw_hidden_size=int(payload["raw_hidden_size"]),
                raw_vector_size=int(payload.get("raw_vector_size", payload["raw_hidden_size"])),
                prompt_tokens=int(payload["prompt_tokens"]),
                task_sha256=str(payload["task_sha256"]),
                prompt_sha256=str(payload.get("prompt_sha256", "")),
                core_path=str(payload["core_path"]),
                layer_labels=tuple(str(value) for value in payload.get("layer_labels", ["final"])),
                pooling=str(payload.get("pooling", DEFAULT_HIDDEN_POOLING)),
                extractor_version=str(
                    payload.get("extractor_version", HIDDEN_STATE_EXTRACTOR_VERSION)
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid hidden-state reference: {exc}") from exc
        reference.validate()
        return reference

    def validate(self) -> None:
        if self.format != HIDDEN_STATE_FORMAT:
            raise ValueError(f"unsupported hidden-state format {self.format!r}")
        for name, value in (
            ("sha256", self.sha256),
            ("model_fingerprint", self.model_fingerprint),
            ("extractor_schema_hash", self.extractor_schema_hash),
            ("task_sha256", self.task_sha256),
            ("prompt_sha256", self.prompt_sha256),
        ):
            if not _is_digest(value):
                raise ValueError(f"invalid hidden-state {name}")
        if self.feature_size < 1 or self.raw_hidden_size < 1 or self.raw_vector_size < 1:
            raise ValueError("hidden-state reference dimensions must be positive")
        if self.prompt_tokens < 1 or not self.layer_labels or not self.extractor_version.strip():
            raise ValueError("hidden-state reference metadata is incomplete")

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "format": self.format,
            "feature_size": self.feature_size,
            "model_fingerprint": self.model_fingerprint,
            "extractor_schema_hash": self.extractor_schema_hash,
            "raw_hidden_size": self.raw_hidden_size,
            "raw_vector_size": self.raw_vector_size,
            "prompt_tokens": self.prompt_tokens,
            "task_sha256": self.task_sha256,
            "prompt_sha256": self.prompt_sha256,
            "core_path": self.core_path,
            "layer_labels": list(self.layer_labels),
            "pooling": self.pooling,
            "extractor_version": self.extractor_version,
        }


@dataclass(frozen=True)
class HiddenStateObservation:
    features: tuple[float, ...]
    reference: HiddenStateReference
    cache_hit: bool = False


class HiddenStateSource(Protocol):
    @property
    def hidden_state_identity(self) -> str: ...

    def capture_policy_hidden(
        self,
        *,
        state: RunState,
        node_id: str,
        decision_type: str,
    ) -> HiddenStateObservation: ...


class HiddenStateArtifactStore:
    """Immutable hash-addressed storage for projected features only.

    Raw prompts and raw Qwen hidden tensors are never written by this store.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self._lock = RLock()

    def write(self, capture: HiddenStateCapture) -> HiddenStateReference:
        payload_bytes = _canonical_json_bytes(capture.payload())
        if len(payload_bytes) > MAX_HIDDEN_ARTIFACT_BYTES:
            raise ValueError("hidden-state artifact exceeds the configured safety limit")
        digest = _sha256_bytes(payload_bytes)
        destination = self.root / digest[:2] / f"{digest}.json"
        with self._lock:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                existing = destination.read_bytes()
                if _sha256_bytes(existing) != digest:
                    raise OSError("existing hidden-state artifact failed hash verification")
            else:
                temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
                try:
                    with temporary.open("xb") as handle:
                        handle.write(payload_bytes)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.chmod(temporary, 0o600)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
        return HiddenStateReference(
            path=str(destination),
            sha256=digest,
            format=HIDDEN_STATE_FORMAT,
            feature_size=len(capture.features),
            model_fingerprint=capture.model_fingerprint,
            extractor_schema_hash=capture.extractor_schema_hash,
            raw_hidden_size=capture.raw_hidden_size,
            raw_vector_size=capture.raw_vector_size,
            prompt_tokens=capture.prompt_tokens,
            task_sha256=capture.task_sha256,
            prompt_sha256=capture.prompt_sha256,
            core_path=capture.core_path,
            layer_labels=capture.layer_labels,
            pooling=capture.pooling,
            extractor_version=capture.extractor_version,
        )

    @staticmethod
    def load(reference: HiddenStateReference | Mapping[str, Any]) -> HiddenStateObservation:
        ref = reference if isinstance(reference, HiddenStateReference) else HiddenStateReference.from_dict(reference)
        ref.validate()
        path = Path(ref.path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"hidden-state artifact does not exist: {path}")
        size = path.stat().st_size
        if size < 2 or size > MAX_HIDDEN_ARTIFACT_BYTES:
            raise ValueError("hidden-state artifact has an invalid size")
        payload_bytes = path.read_bytes()
        actual_hash = _sha256_bytes(payload_bytes)
        if actual_hash != ref.sha256:
            raise ValueError(
                "hidden-state artifact hash mismatch: "
                f"expected={ref.sha256}, actual={actual_hash}"
            )
        try:
            payload = json.loads(payload_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid hidden-state artifact JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("hidden-state artifact must contain a JSON object")
        if int(payload.get("format_version", -1)) != HIDDEN_STATE_FORMAT_VERSION:
            raise ValueError("unsupported hidden-state artifact format version")
        if payload.get("format") != HIDDEN_STATE_FORMAT:
            raise ValueError("unsupported hidden-state artifact format")
        try:
            features = tuple(_finite_float(value) for value in payload["features"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid hidden-state feature vector: {exc}") from exc
        checks = {
            "feature_size": (len(features), ref.feature_size),
            "model_fingerprint": (payload.get("model_fingerprint"), ref.model_fingerprint),
            "extractor_schema_hash": (payload.get("extractor_schema_hash"), ref.extractor_schema_hash),
            "raw_hidden_size": (payload.get("raw_hidden_size"), ref.raw_hidden_size),
            "raw_vector_size": (payload.get("raw_vector_size"), ref.raw_vector_size),
            "prompt_tokens": (payload.get("prompt_tokens"), ref.prompt_tokens),
            "task_sha256": (payload.get("task_sha256"), ref.task_sha256),
            "prompt_sha256": (payload.get("prompt_sha256"), ref.prompt_sha256),
            "core_path": (payload.get("core_path"), ref.core_path),
            "layer_labels": (tuple(payload.get("layer_labels", [])), ref.layer_labels),
            "pooling": (payload.get("pooling"), ref.pooling),
            "extractor_version": (
                payload.get("extractor_version", HIDDEN_STATE_EXTRACTOR_VERSION),
                ref.extractor_version,
            ),
        }
        mismatches = {
            name: {"artifact": artifact, "reference": expected}
            for name, (artifact, expected) in checks.items()
            if artifact != expected
        }
        if mismatches:
            raise ValueError(f"hidden-state artifact metadata mismatch: {mismatches}")
        return HiddenStateObservation(features=features, reference=ref)


def model_fingerprint(identity: Mapping[str, Any]) -> str:
    stable = {
        key: value
        for key, value in identity.items()
        if key in {"kind", "backend", "model", "adapter", "revision", "trust_remote_code"}
    }
    return _sha256_bytes(_canonical_json_bytes(stable))


def capture_from_raw_hidden(
    raw: RawHiddenStateLike,
    *,
    task: str,
    model_identity: Mapping[str, Any],
    config: HiddenStateCaptureConfig,
) -> HiddenStateCapture:
    labels = tuple(str(label) for label in raw.layer_labels)
    if not labels:
        raise ValueError("raw hidden state contains no layer labels")
    values = tuple(_finite_float(value) for value in raw.values)
    if not values or len(values) % len(labels) != 0:
        raise ValueError("raw hidden-state vector cannot be split across selected layers")
    block_size = len(values) // len(labels)
    views = {
        label: values[index * block_size : (index + 1) * block_size]
        for index, label in enumerate(labels)
    }
    features = project_hidden_views(
        views,
        output_size=config.feature_size,
        seed=config.projection_seed,
    )
    return HiddenStateCapture(
        features=features,
        model_fingerprint=model_fingerprint(model_identity),
        extractor_schema_hash=config.schema_hash,
        raw_hidden_size=int(raw.model_hidden_size),
        raw_vector_size=len(values),
        prompt_tokens=int(raw.token_count),
        task_sha256=_sha256_text(task),
        prompt_sha256=str(raw.prompt_sha256),
        core_path=str(raw.source),
        layer_labels=labels,
        pooling=str(raw.pooling),
        extractor_version=config.extractor_version,
    )


def require_hidden_state_source(value: Any) -> HiddenStateSource:
    if not callable(getattr(value, "capture_policy_hidden", None)):
        raise ProviderError(
            "the selected provider cannot capture MLX hidden states required by this policy"
        )
    if not isinstance(getattr(value, "hidden_state_identity", None), str):
        raise ProviderError("hidden-state provider identity is unavailable")
    return value
