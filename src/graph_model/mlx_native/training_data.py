from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graph_model.controller import ROUTES, STOP_ACTIONS
from graph_model.models import GraphSpec
from graph_model.store import SQLiteRunStore

from .features import controller_input_size
from .graph_tables import CompiledGraphTables, compile_graph


@dataclass(frozen=True)
class PolicyTrainingRecord:
    run_id: str
    node_id: str
    decision_type: str
    features: tuple[float, ...]
    route_label: int
    edge_label: int
    stop_label: int
    allowed_edge_mask: tuple[bool, ...]
    allowed_stop_mask: tuple[bool, ...]
    reward: float
    cost_target: tuple[float, float, float]


def _ratio(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return min(1.0, max(0.0, value / maximum))


def _features_from_decision(
    decision: dict[str, Any],
    tables: CompiledGraphTables,
) -> tuple[float, ...] | None:
    metrics = decision.get("policy_metrics")
    if not isinstance(metrics, dict):
        return None
    if metrics.get("graph_schema_hash") != tables.schema_hash:
        return None
    values = metrics.get("feature_vector")
    if not isinstance(values, list) or len(values) != controller_input_size(tables):
        return None
    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None


def export_mlx_policy_training_data(
    store: SQLiteRunStore,
    output_path: str | Path,
    *,
    graph: GraphSpec,
    success_only: bool = False,
) -> int:
    tables = compile_graph(graph)
    records: list[dict[str, Any]] = []
    for state in reversed(store.list_runs()):
        if state.graph_name != graph.name or state.graph_version != graph.version:
            continue
        if success_only and state.status != "completed":
            continue
        reward = 1.0 if state.status == "completed" else 0.0
        cost_target = [
            _ratio(state.metrics.total_tokens, state.budget.max_tokens),
            _ratio(state.metrics.elapsed_seconds, state.budget.max_seconds),
            _ratio(state.metrics.tool_calls, state.budget.max_tool_calls),
        ]
        for event in store.events(state.run_id):
            if event.get("event_type") != "node_completed":
                continue
            payload = event.get("payload") or {}
            node_id = str(event.get("node_id") or "")
            result = payload.get("result") or {}
            delta = result.get("delta") if isinstance(result, dict) else None
            router = delta.get("router") if isinstance(delta, dict) else None
            if isinstance(router, dict):
                features = _features_from_decision(router, tables)
                route = router.get("route")
                if features is not None and route in ROUTES:
                    records.append(
                        {
                            "run_id": state.run_id,
                            "node_id": node_id,
                            "decision_type": "route",
                            "features": list(features),
                            "route_label": ROUTES.index(route),
                            "edge_label": -1,
                            "stop_label": -1,
                            "allowed_edge_mask": [False] * len(tables.edge_keys),
                            "allowed_stop_mask": [False] * len(STOP_ACTIONS),
                            "reward": reward,
                            "cost_target": cost_target,
                        }
                    )

            stop = payload.get("stop_decision")
            edge = payload.get("edge_decision")
            if not isinstance(stop, dict) or not isinstance(edge, dict):
                continue
            features = _features_from_decision(edge, tables) or _features_from_decision(
                stop, tables
            )
            edge_key = edge.get("edge_key")
            stop_action = stop.get("action")
            if (
                features is None
                or edge_key not in tables.edge_index
                or stop_action not in STOP_ACTIONS
            ):
                continue
            allowed_edge_keys = edge.get("allowed_edge_keys") or []
            allowed_actions = stop.get("allowed_actions") or []
            records.append(
                {
                    "run_id": state.run_id,
                    "node_id": node_id,
                    "decision_type": "transition",
                    "features": list(features),
                    "route_label": -1,
                    "edge_label": tables.edge_index[str(edge_key)],
                    "stop_label": STOP_ACTIONS.index(str(stop_action)),
                    "allowed_edge_mask": [
                        edge_name in allowed_edge_keys for edge_name in tables.edge_keys
                    ],
                    "allowed_stop_mask": [
                        action in allowed_actions for action in STOP_ACTIONS
                    ],
                    "reward": reward,
                    "cost_target": cost_target,
                }
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return len(records)


def read_policy_training_data(
    input_path: str | Path,
    *,
    tables: CompiledGraphTables,
) -> list[PolicyTrainingRecord]:
    expected_features = controller_input_size(tables)
    records: list[PolicyTrainingRecord] = []
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid policy JSONL at line {line_number}: {exc}") from exc
            try:
                record = PolicyTrainingRecord(
                    run_id=str(payload["run_id"]),
                    node_id=str(payload["node_id"]),
                    decision_type=str(payload["decision_type"]),
                    features=tuple(float(value) for value in payload["features"]),
                    route_label=int(payload["route_label"]),
                    edge_label=int(payload["edge_label"]),
                    stop_label=int(payload["stop_label"]),
                    allowed_edge_mask=tuple(bool(value) for value in payload["allowed_edge_mask"]),
                    allowed_stop_mask=tuple(bool(value) for value in payload["allowed_stop_mask"]),
                    reward=float(payload["reward"]),
                    cost_target=tuple(float(value) for value in payload["cost_target"]),  # type: ignore[arg-type]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid policy training record at line {line_number}: {exc}"
                ) from exc
            if len(record.features) != expected_features:
                raise ValueError(
                    f"line {line_number}: expected {expected_features} features, "
                    f"got {len(record.features)}"
                )
            if len(record.allowed_edge_mask) != len(tables.edge_keys):
                raise ValueError(f"line {line_number}: invalid allowed_edge_mask length")
            if len(record.allowed_stop_mask) != len(STOP_ACTIONS):
                raise ValueError(f"line {line_number}: invalid allowed_stop_mask length")
            if len(record.cost_target) != 3:
                raise ValueError(f"line {line_number}: cost_target must contain three values")
            if not -1 <= record.route_label < len(ROUTES):
                raise ValueError(f"line {line_number}: invalid route label")
            if not -1 <= record.edge_label < len(tables.edge_keys):
                raise ValueError(f"line {line_number}: invalid edge label")
            if not -1 <= record.stop_label < len(STOP_ACTIONS):
                raise ValueError(f"line {line_number}: invalid stop label")
            records.append(record)
    if not records:
        raise ValueError("policy training input contains no usable records")
    return records
