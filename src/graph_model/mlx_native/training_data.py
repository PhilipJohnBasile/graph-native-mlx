from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from graph_model.controller import ROUTES, STOP_ACTIONS
from graph_model.models import GraphSpec
from graph_model.store import SQLiteRunStore

from .features import controller_input_size
from .graph_tables import CompiledGraphTables, compile_graph
from .hidden_state import HiddenStateArtifactStore, HiddenStateObservation


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
    hidden_features: tuple[float, ...] = ()
    hidden_state_schema_hash: str = ""
    model_fingerprint: str = ""
    hidden_artifact_sha256: str = ""


@dataclass(frozen=True)
class PolicyDatasetIdentity:
    hidden_feature_size: int
    hidden_state_schema_hash: str
    model_fingerprint: str

    @property
    def uses_hidden_states(self) -> bool:
        return self.hidden_feature_size > 0


def _ratio(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return min(1.0, max(0.0, value / maximum))


def _features_from_decision(
    decision: Mapping[str, Any],
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


def _hidden_from_decision(
    decision: Mapping[str, Any],
    cache: dict[str, HiddenStateObservation],
) -> HiddenStateObservation | None:
    metrics = decision.get("policy_metrics")
    if not isinstance(metrics, dict):
        return None
    reference = metrics.get("hidden_state")
    if not isinstance(reference, dict):
        return None
    digest = str(reference.get("sha256", ""))
    if digest in cache:
        return cache[digest]
    observation = HiddenStateArtifactStore.load(reference)
    cache[observation.reference.sha256] = observation
    return observation


def _hidden_payload(observation: HiddenStateObservation | None) -> dict[str, Any]:
    if observation is None:
        return {
            "hidden_features": [],
            "hidden_state_schema_hash": "",
            "model_fingerprint": "",
            "hidden_artifact_sha256": "",
        }
    return {
        "hidden_features": list(observation.features),
        "hidden_state_schema_hash": observation.reference.extractor_schema_hash,
        "model_fingerprint": observation.reference.model_fingerprint,
        "hidden_artifact_sha256": observation.reference.sha256,
    }


def export_mlx_policy_training_data(
    store: SQLiteRunStore,
    output_path: str | Path,
    *,
    graph: GraphSpec,
    success_only: bool = False,
    require_hidden: bool = False,
) -> int:
    tables = compile_graph(graph)
    records: list[dict[str, Any]] = []
    hidden_cache: dict[str, HiddenStateObservation] = {}
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
                hidden = _hidden_from_decision(router, hidden_cache)
                if (
                    features is not None
                    and route in ROUTES
                    and (hidden is not None or not require_hidden)
                ):
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
                            **_hidden_payload(hidden),
                        }
                    )

            stop = payload.get("stop_decision")
            edge = payload.get("edge_decision")
            if not isinstance(stop, dict) or not isinstance(edge, dict):
                continue
            features = _features_from_decision(edge, tables) or _features_from_decision(
                stop, tables
            )
            hidden = _hidden_from_decision(edge, hidden_cache) or _hidden_from_decision(
                stop, hidden_cache
            )
            edge_key = edge.get("edge_key")
            stop_action = stop.get("action")
            if (
                features is None
                or edge_key not in tables.edge_index
                or stop_action not in STOP_ACTIONS
                or (require_hidden and hidden is None)
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
                    **_hidden_payload(hidden),
                }
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    return len(records)


def _digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def read_policy_training_data(
    input_path: str | Path,
    *,
    tables: CompiledGraphTables,
    require_hidden: bool = False,
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
                    allowed_edge_mask=tuple(
                        bool(value) for value in payload["allowed_edge_mask"]
                    ),
                    allowed_stop_mask=tuple(
                        bool(value) for value in payload["allowed_stop_mask"]
                    ),
                    reward=float(payload["reward"]),
                    cost_target=tuple(
                        float(value) for value in payload["cost_target"]
                    ),  # type: ignore[arg-type]
                    hidden_features=tuple(
                        float(value) for value in payload.get("hidden_features", [])
                    ),
                    hidden_state_schema_hash=str(
                        payload.get("hidden_state_schema_hash", "")
                    ),
                    model_fingerprint=str(payload.get("model_fingerprint", "")),
                    hidden_artifact_sha256=str(
                        payload.get("hidden_artifact_sha256", "")
                    ),
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
            numeric_values = (
                *record.features,
                *record.hidden_features,
                record.reward,
                *record.cost_target,
            )
            if any(not math.isfinite(value) for value in numeric_values):
                raise ValueError(f"line {line_number}: policy numeric values must be finite")
            if not 0.0 <= record.reward <= 1.0:
                raise ValueError(f"line {line_number}: reward must be in [0, 1]")
            if any(not 0.0 <= value <= 1.0 for value in record.cost_target):
                raise ValueError(f"line {line_number}: cost targets must be in [0, 1]")
            if record.node_id not in tables.node_index:
                raise ValueError(f"line {line_number}: unknown node_id {record.node_id!r}")
            if record.decision_type == "route":
                if record.route_label < 0 or record.edge_label != -1 or record.stop_label != -1:
                    raise ValueError(f"line {line_number}: invalid route decision labels")
                if any(record.allowed_edge_mask) or any(record.allowed_stop_mask):
                    raise ValueError(f"line {line_number}: route records must not carry transition masks")
            elif record.decision_type == "transition":
                if record.route_label != -1 or record.edge_label < 0 or record.stop_label < 0:
                    raise ValueError(f"line {line_number}: invalid transition decision labels")
                if not record.allowed_edge_mask[record.edge_label]:
                    raise ValueError(f"line {line_number}: selected edge is masked")
                if not record.allowed_stop_mask[record.stop_label]:
                    raise ValueError(f"line {line_number}: selected stop action is masked")
            else:
                raise ValueError(
                    f"line {line_number}: decision_type must be 'route' or 'transition'"
                )
            if record.hidden_features:
                for name, value in (
                    ("hidden_state_schema_hash", record.hidden_state_schema_hash),
                    ("model_fingerprint", record.model_fingerprint),
                    ("hidden_artifact_sha256", record.hidden_artifact_sha256),
                ):
                    if not _digest(value):
                        raise ValueError(f"line {line_number}: invalid {name}")
            elif require_hidden:
                raise ValueError(f"line {line_number}: hidden-state features are required")
            elif any(
                (
                    record.hidden_state_schema_hash,
                    record.model_fingerprint,
                    record.hidden_artifact_sha256,
                )
            ):
                raise ValueError(
                    f"line {line_number}: hidden metadata exists without hidden features"
                )
            records.append(record)
    if not records:
        raise ValueError("policy training input contains no usable records")
    dataset_identity(records)
    return records


def dataset_identity(records: list[PolicyTrainingRecord]) -> PolicyDatasetIdentity:
    sizes = {len(record.hidden_features) for record in records if record.hidden_features}
    schemas = {
        record.hidden_state_schema_hash for record in records if record.hidden_features
    }
    models = {record.model_fingerprint for record in records if record.hidden_features}
    has_missing = any(not record.hidden_features for record in records)
    if sizes and has_missing:
        raise ValueError(
            "policy dataset mixes hidden-state and explicit-only records; export a homogeneous dataset"
        )
    if len(sizes) > 1:
        raise ValueError(f"policy dataset mixes hidden feature sizes: {sorted(sizes)}")
    if len(schemas) > 1:
        raise ValueError("policy dataset mixes hidden-state extractor schemas")
    if len(models) > 1:
        raise ValueError("policy dataset mixes model fingerprints")
    return PolicyDatasetIdentity(
        hidden_feature_size=next(iter(sizes), 0),
        hidden_state_schema_hash=next(iter(schemas), ""),
        model_fingerprint=next(iter(models), ""),
    )
