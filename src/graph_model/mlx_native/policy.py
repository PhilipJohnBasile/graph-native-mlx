from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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

    @classmethod
    def for_graph(
        cls,
        tables: CompiledGraphTables,
        *,
        hidden_size: int = 128,
    ) -> "GraphPolicyConfig":
        return cls(
            format_version=1,
            graph_name=tables.name,
            graph_version=tables.version,
            graph_schema_hash=tables.schema_hash,
            input_size=controller_input_size(tables),
            hidden_size=hidden_size,
            route_count=3,
            edge_count=len(tables.edge_keys),
            stop_count=4,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GraphPolicyConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("graph policy config must be a JSON object")
        return cls(**payload)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.__dict__, indent=2, sort_keys=True), encoding="utf-8")
        return output

    def validate(self, tables: CompiledGraphTables) -> None:
        if self.hidden_size < 8:
            raise ValueError("graph policy hidden_size must be >= 8")
        expected = GraphPolicyConfig.for_graph(tables, hidden_size=self.hidden_size)
        mismatches = {
            field: (getattr(self, field), getattr(expected, field))
            for field in (
                "format_version",
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
            raise ValueError(f"graph policy config is incompatible with compiled graph: {mismatches}")


@dataclass(frozen=True)
class PolicyOutput:
    route_logits: tuple[float, ...]
    edge_logits: tuple[float, ...]
    stop_logits: tuple[float, ...]
    success_value: float
    cost: tuple[float, ...]


if nn is not None:

    class GraphPolicyHeads(nn.Module):  # pragma: no cover - executed on MLX hosts
        """Small sidecar policy over explicit graph/task state.

        The backbone remains responsible for language generation. These heads predict only route,
        edge, stop, value, and cost residuals; runtime masks remain authoritative.
        """

        def __init__(self, config: GraphPolicyConfig) -> None:
            super().__init__()
            self.input_norm = nn.LayerNorm(config.input_size)
            self.input_projection = nn.Linear(config.input_size, config.hidden_size)
            self.hidden_projection = nn.Linear(config.hidden_size, config.hidden_size)
            self.route_head = nn.Linear(config.hidden_size, config.route_count)
            self.edge_head = nn.Linear(config.hidden_size, config.edge_count)
            self.stop_head = nn.Linear(config.hidden_size, config.stop_count)
            self.value_head = nn.Linear(config.hidden_size, 1)
            self.cost_head = nn.Linear(config.hidden_size, 3)

        def __call__(self, features):
            hidden = nn.silu(self.input_projection(self.input_norm(features)))
            hidden = nn.silu(self.hidden_projection(hidden))
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

    def predict(self, features: Sequence[float]) -> PolicyOutput:
        if len(features) != self.config.input_size:
            raise ValueError(
                f"policy expected {self.config.input_size} features, got {len(features)}"
            )
        values = mx.array([list(features)], dtype=mx.float32)
        route, edge, stop, value, cost = self.model(values)
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
        )
