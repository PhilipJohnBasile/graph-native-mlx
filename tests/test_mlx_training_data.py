import json
from pathlib import Path

import pytest

from graph_model.graph import load_default_graph
from graph_model.mlx_native.controller import MLXGraphController
from graph_model.mlx_native.decision import PythonDecisionBackend
from graph_model.mlx_native.graph_tables import compile_graph
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
    assert count == 6
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["decision_type"] == "route"
    assert rows[0]["route_label"] == 0
    transition_rows = [row for row in rows if row["decision_type"] == "transition"]
    assert len(transition_rows) == 5
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
        "features": [0.0] * 55,
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
