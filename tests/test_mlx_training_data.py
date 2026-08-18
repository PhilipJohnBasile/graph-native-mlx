import json
from pathlib import Path

import pytest

from graph_model.graph import load_default_graph
from graph_model.mlx_native.controller import MLXGraphController
from graph_model.mlx_native.decision import PythonDecisionBackend
from graph_model.mlx_native.graph_tables import compile_graph
from graph_model.mlx_native.features import controller_input_size
from graph_model.mlx_native.training_data import (
    export_mlx_policy_training_data,
    read_policy_training_data,
)
from graph_model.provider import MockProvider
from graph_model.runtime import GraphRuntime
from graph_model.store import SQLiteRunStore


@pytest.mark.asyncio
async def test_policy_training_export_contains_hard_masks_and_features(tmp_path: Path) -> None:
    graph = load_default_graph()
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    controller = MLXGraphController(
        graph=graph,
        decision_backend=PythonDecisionBackend(),
    )
    runtime = GraphRuntime(
        graph=graph,
        store=store,
        provider=MockProvider(),
        controller=controller,
    )
    await runtime.run("quick fix: rename one variable", run_id="training-export")

    output = tmp_path / "policy.jsonl"
    count = export_mlx_policy_training_data(store, output, graph=graph)
    assert count == 7
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["decision_type"] == "route"
    assert rows[0]["route_label"] == 0
    transition_rows = [row for row in rows if row["decision_type"] == "transition"]
    assert len(transition_rows) == 6
    assert all(sum(row["allowed_edge_mask"]) == 1 for row in transition_rows)
    assert all(sum(row["allowed_stop_mask"]) >= 1 for row in transition_rows)

    parsed = read_policy_training_data(output, tables=compile_graph(graph))
    assert len(parsed) == count
    assert all(record.reward == 1.0 for record in parsed)


def test_policy_training_data_rejects_labels_below_not_applicable(tmp_path: Path) -> None:
    tables = compile_graph(load_default_graph())
    record = {
        "run_id": "bad",
        "node_id": "intake",
        "decision_type": "route",
        "features": [0.0] * controller_input_size(tables),
        "route_label": -2,
        "edge_label": -1,
        "stop_label": -1,
        "allowed_edge_mask": [False] * len(tables.edge_keys),
        "allowed_stop_mask": [False] * 4,
        "reward": 0.0,
        "cost_target": [0.0, 0.0, 0.0],
    }
    path = tmp_path / "bad-policy.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid route label"):
        read_policy_training_data(path, tables=tables)


class _StateAwareHiddenSource:
    hidden_state_identity = "state-aware-training-source"
    hidden_capture_enabled = True

    def __init__(self, root: Path) -> None:
        import hashlib

        from graph_model.mlx_native.hidden_state import HiddenStateArtifactStore

        self._hashlib = hashlib
        self.store = HiddenStateArtifactStore(root)
        self.calls: list[tuple[int, str, str]] = []

    def run_on_affinity(self, function, *args):
        return function(*args)

    def capture_policy_hidden(self, *, state, node_id, decision_type):
        from graph_model.mlx_native.hidden_state import (
            HiddenStateCapture,
            HiddenStateObservation,
        )

        marker = json.dumps(
            {
                "run": state.run_id,
                "step": state.step_count,
                "node": node_id,
                "decision": decision_type,
                "current": state.current_node,
                "data": state.data,
            },
            sort_keys=True,
            default=str,
        )
        digest = self._hashlib.sha256(marker.encode("utf-8")).hexdigest()
        self.calls.append((state.step_count, node_id, decision_type))
        base = int(digest[:8], 16) / 0xFFFFFFFF
        features = tuple(base + index / 100.0 for index in range(8))
        capture = HiddenStateCapture(
            features=features,
            model_fingerprint="a" * 64,
            extractor_schema_hash="b" * 64,
            raw_hidden_size=8,
            raw_vector_size=8,
            prompt_tokens=16,
            task_sha256=self._hashlib.sha256(state.task.encode("utf-8")).hexdigest(),
            prompt_sha256=digest,
            core_path="model.model",
            layer_labels=("final",),
            pooling="last-token",
        )
        reference = self.store.write(capture)
        return HiddenStateObservation(features=features, reference=reference)


@pytest.mark.asyncio
async def test_policy_training_export_requires_and_preserves_state_aware_hidden_features(
    tmp_path: Path,
) -> None:
    from graph_model.mlx_native.training_data import dataset_identity

    graph = load_default_graph()
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    hidden_source = _StateAwareHiddenSource(tmp_path / "hidden")
    controller = MLXGraphController(
        graph=graph,
        decision_backend=PythonDecisionBackend(),
        hidden_state_source=hidden_source,
        capture_hidden=True,
    )
    runtime = GraphRuntime(
        graph=graph,
        store=store,
        provider=MockProvider(),
        controller=controller,
    )
    await runtime.run("quick fix: rename one variable", run_id="hidden-training-export")

    output = tmp_path / "hidden-policy.jsonl"
    count = export_mlx_policy_training_data(
        store,
        output,
        graph=graph,
        require_hidden=True,
    )
    assert count == 7

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert all(len(row["hidden_features"]) == 8 for row in rows)
    assert all(row["hidden_state_schema_hash"] == "b" * 64 for row in rows)
    assert all(row["model_fingerprint"] == "a" * 64 for row in rows)
    assert len({row["hidden_artifact_sha256"] for row in rows}) == count

    parsed = read_policy_training_data(
        output,
        tables=compile_graph(graph),
        require_hidden=True,
    )
    identity = dataset_identity(parsed)
    assert identity.uses_hidden_states is True
    assert identity.hidden_feature_size == 8
    assert identity.hidden_state_schema_hash == "b" * 64
    assert identity.model_fingerprint == "a" * 64
    assert {record.decision_type for record in parsed} == {"route", "transition"}


def test_policy_training_data_rejects_nonfinite_and_mask_inconsistent_records(
    tmp_path: Path,
) -> None:
    tables = compile_graph(load_default_graph())
    base = {
        "run_id": "bad",
        "node_id": "intake",
        "decision_type": "route",
        "features": [0.0] * controller_input_size(tables),
        "route_label": 0,
        "edge_label": -1,
        "stop_label": -1,
        "allowed_edge_mask": [False] * len(tables.edge_keys),
        "allowed_stop_mask": [False] * 4,
        "reward": 1.0,
        "cost_target": [0.0, 0.0, 0.0],
    }
    nonfinite = tmp_path / "nonfinite.jsonl"
    payload = dict(base)
    payload["features"] = list(payload["features"])
    payload["features"][0] = float("nan")
    nonfinite.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        read_policy_training_data(nonfinite, tables=tables)

    transition = dict(base)
    transition.update(
        {
            "decision_type": "transition",
            "route_label": -1,
            "edge_label": 0,
            "stop_label": 0,
        }
    )
    masked = tmp_path / "masked.jsonl"
    masked.write_text(json.dumps(transition) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="selected edge is masked"):
        read_policy_training_data(masked, tables=tables)
