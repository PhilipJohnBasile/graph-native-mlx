from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision import MLXUnavailableError
from .features import controller_input_size
from .graph_tables import CompiledGraphTables

try:  # pragma: no cover - import availability is platform dependent
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover - normal in non-MLX CI
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class GraphPolicyConfig:
    format_version: int
    graph_name: str
    graph_version: str
    graph_schema_hash: str
    input_size: int
    hidden_size: int
    route_count: int
    edge_count: int
    stop_count: int
    backbone_feature_size: int = 0
    hidden_state_schema_hash: str = ""
    model_fingerprint: str = ""
    fusion_version: str = "explicit-only-v1"

    @classmethod
    def for_graph(
        cls,
        tables: CompiledGraphTables,
        *,
        hidden_size: int = 128,
        backbone_feature_size: int = 0,
        hidden_state_schema_hash: str = "",
        model_fingerprint: str = "",
    ) -> "GraphPolicyConfig":
        if backbone_feature_size < 0:
            raise ValueError("backbone_feature_size must be non-negative")
        using_backbone = backbone_feature_size > 0
        return cls(
            format_version=2,
            graph_name=tables.name,
            graph_version=tables.version,
            graph_schema_hash=tables.schema_hash,
            input_size=controller_input_size(tables),
            hidden_size=hidden_size,
            route_count=3,
            edge_count=len(tables.edge_keys),
            stop_count=4,
            backbone_feature_size=backbone_feature_size,
            hidden_state_schema_hash=(hidden_state_schema_hash if using_backbone else ""),
            model_fingerprint=(model_fingerprint if using_backbone else ""),
            fusion_version=("gated-residual-v1" if using_backbone else "explicit-only-v1"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GraphPolicyConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("graph policy config must be a JSON object")
        # v0.3 sidecars are explicit-feature-only and remain loadable.
        payload.setdefault("backbone_feature_size", 0)
        payload.setdefault("hidden_state_schema_hash", "")
        payload.setdefault("model_fingerprint", "")
        payload.setdefault("fusion_version", "explicit-only-v1")
        return cls(**payload)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.__dict__, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output

    @property
    def requires_backbone_features(self) -> bool:
        return self.backbone_feature_size > 0

    def validate(self, tables: CompiledGraphTables) -> None:
        if self.format_version not in {1, 2}:
            raise ValueError(f"unsupported graph policy format_version={self.format_version}")
        if self.hidden_size < 8:
            raise ValueError("graph policy hidden_size must be >= 8")
        if self.backbone_feature_size < 0:
            raise ValueError("graph policy backbone_feature_size must be non-negative")
        expected = GraphPolicyConfig.for_graph(
            tables,
            hidden_size=self.hidden_size,
            backbone_feature_size=self.backbone_feature_size,
            hidden_state_schema_hash=self.hidden_state_schema_hash,
            model_fingerprint=self.model_fingerprint,
        )
        mismatches = {
            field: (getattr(self, field), getattr(expected, field))
            for field in (
                "graph_name",
                "graph_version",
                "graph_schema_hash",
                "input_size",
                "route_count",
                "edge_count",
                "stop_count",
            )
            if getattr(self, field) != getattr(expected, field)
        }
        if mismatches:
            raise ValueError(
                f"graph policy config is incompatible with compiled graph: {mismatches}"
            )
        if self.requires_backbone_features:
            if self.format_version < 2:
                raise ValueError("backbone features require graph policy format_version >= 2")
            if self.backbone_feature_size < 8:
                raise ValueError("backbone_feature_size must be 0 or >= 8")
            for name, value in (
                ("hidden_state_schema_hash", self.hidden_state_schema_hash),
                ("model_fingerprint", self.model_fingerprint),
            ):
                if len(value) != 64 or any(
                    character not in "0123456789abcdef" for character in value
                ):
                    raise ValueError(f"{name} must be a lowercase SHA-256 digest")
            if self.fusion_version != "gated-residual-v1":
                raise ValueError(
                    f"unsupported hidden-state fusion_version={self.fusion_version!r}"
                )
        elif self.fusion_version != "explicit-only-v1":
            raise ValueError("explicit-only policy must use fusion_version='explicit-only-v1'")


@dataclass(frozen=True)
class PolicyOutput:
    route_logits: tuple[float, ...]
    edge_logits: tuple[float, ...]
    stop_logits: tuple[float, ...]
    success_value: float
    cost: tuple[float, ...]
    hidden_used: bool = False


if nn is not None:

    class GraphPolicyHeads(nn.Module):  # pragma: no cover - executed on MLX hosts
        """Small graph policy that can fuse explicit state with Qwen hidden features.

        The policy predicts residual logits only. Runtime predicates and graph masks remain the
        authority for route, edge, and stop validity.
        """

        def __init__(self, config: GraphPolicyConfig) -> None:
            super().__init__()
            self.config = config
            self.input_norm = nn.LayerNorm(config.input_size)
            self.input_projection = nn.Linear(config.input_size, config.hidden_size)
            self.hidden_projection = nn.Linear(config.hidden_size, config.hidden_size)
            if config.requires_backbone_features:
                self.backbone_norm = nn.LayerNorm(config.backbone_feature_size)
                self.backbone_projection = nn.Linear(
                    config.backbone_feature_size,
                    config.hidden_size,
                )
                self.fusion_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
                self.fusion_projection = nn.Linear(
                    config.hidden_size * 2,
                    config.hidden_size,
                )
            self.route_head = nn.Linear(config.hidden_size, config.route_count)
            self.edge_head = nn.Linear(config.hidden_size, config.edge_count)
            self.stop_head = nn.Linear(config.hidden_size, config.stop_count)
            self.value_head = nn.Linear(config.hidden_size, 1)
            self.cost_head = nn.Linear(config.hidden_size, 3)

        def __call__(self, features, backbone_features=None):
            explicit = nn.silu(self.input_projection(self.input_norm(features)))
            explicit = nn.silu(self.hidden_projection(explicit))
            hidden = explicit
            if self.config.requires_backbone_features:
                if backbone_features is None:
                    raise ValueError("backbone features are required by this graph policy")
                backbone = nn.silu(
                    self.backbone_projection(self.backbone_norm(backbone_features))
                )
                joined = mx.concatenate([explicit, backbone], axis=-1)
                gate = mx.sigmoid(self.fusion_gate(joined))
                candidate = nn.silu(self.fusion_projection(joined))
                hidden = explicit + gate * candidate
            return (
                self.route_head(hidden),
                self.edge_head(hidden),
                self.stop_head(hidden),
                self.value_head(hidden),
                self.cost_head(hidden),
            )

else:

    class GraphPolicyHeads:  # type: ignore[no-redef]
        def __init__(self, config: GraphPolicyConfig) -> None:
            del config
            raise MLXUnavailableError("GraphPolicyHeads requires MLX")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MLXPolicyRunner:
    def __init__(
        self,
        *,
        tables: CompiledGraphTables,
        weights_path: str | Path,
        config_path: str | Path,
    ) -> None:
        if mx is None or nn is None:  # pragma: no cover - executed on non-MLX hosts
            raise MLXUnavailableError("MLX policy weights require the mlx package")
        self.tables = tables
        self.config = GraphPolicyConfig.load(config_path)
        self.config.validate(tables)
        self.model = GraphPolicyHeads(self.config)
        self.model.load_weights(str(weights_path), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        self.weights_path = str(Path(weights_path).expanduser().resolve())
        self.config_path = str(Path(config_path).expanduser().resolve())
        self._weights_hash = _sha256_file(self.weights_path)
        self._config_hash = _sha256_file(self.config_path)

    @property
    def identity(self) -> str:
        return (
            f"mlx-policy:{self._weights_hash[:16]}:{self._config_hash[:16]}:"
            f"{self.tables.schema_hash[:12]}"
        )

    @property
    def requires_hidden(self) -> bool:
        return self.config.requires_backbone_features

    def predict(
        self,
        features: Sequence[float],
        *,
        hidden_features: Sequence[float] | None = None,
        hidden_metadata: Mapping[str, Any] | None = None,
    ) -> PolicyOutput:
        if len(features) != self.config.input_size:
            raise ValueError(
                f"policy expected {self.config.input_size} features, got {len(features)}"
            )
        backbone_values = None
        if self.config.requires_backbone_features:
            if hidden_features is None:
                raise ValueError("policy requires Qwen hidden-state features")
            if len(hidden_features) != self.config.backbone_feature_size:
                raise ValueError(
                    "policy hidden feature size mismatch: "
                    f"expected={self.config.backbone_feature_size}, "
                    f"actual={len(hidden_features)}"
                )
            metadata = dict(hidden_metadata or {})
            checks = {
                "extractor_schema_hash": self.config.hidden_state_schema_hash,
                "model_fingerprint": self.config.model_fingerprint,
            }
            mismatches = {
                name: {"expected": expected, "actual": metadata.get(name)}
                for name, expected in checks.items()
                if metadata.get(name) != expected
            }
            if mismatches:
                raise ValueError(
                    f"hidden-state features do not match the trained policy: {mismatches}"
                )
            backbone_values = mx.array([list(hidden_features)], dtype=mx.float32)

        values = mx.array([list(features)], dtype=mx.float32)
        route, edge, stop, value, cost = self.model(values, backbone_values)
        mx.eval(route, edge, stop, value, cost)
        success = mx.sigmoid(value)
        positive_cost = nn.softplus(cost)
        mx.eval(success, positive_cost)
        return PolicyOutput(
            route_logits=tuple(float(item) for item in route[0].tolist()),
            edge_logits=tuple(float(item) for item in edge[0].tolist()),
            stop_logits=tuple(float(item) for item in stop[0].tolist()),
            success_value=float(success[0, 0].item()),
            cost=tuple(float(item) for item in positive_cost[0].tolist()),
            hidden_used=self.config.requires_backbone_features,
        )
