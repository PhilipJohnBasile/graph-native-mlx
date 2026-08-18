import json
from pathlib import Path

import pytest

from graph_model.graph import load_default_graph
from graph_model.provider import MockProvider
from graph_model.runtime import GraphRuntime
from graph_model.store import SQLiteRunStore
from graph_model.trace_export import export_router_training_data


@pytest.mark.asyncio
async def test_trace_export_contains_only_completed_node_events(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    runtime = GraphRuntime(graph=load_default_graph(), store=store, provider=MockProvider())
    state = await runtime.run("quick fix: rename one variable", run_id="trace-test")
    output = tmp_path / "traces.jsonl"
    assert export_router_training_data(store, output) == 1
    record = json.loads(output.read_text())
    assert record["path"] == state.completed_nodes
    assert record["path"].count("intake") == 1
    assert record["path"].count("finish") == 1
    assert record["provider"]["kind"] == "mock"
