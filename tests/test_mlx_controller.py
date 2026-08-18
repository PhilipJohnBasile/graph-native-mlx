from pathlib import Path

import pytest

from graph_model.controller import DeterministicGraphController
from graph_model.graph import load_default_graph
from graph_model.mlx_native.controller import MLXGraphController
from graph_model.mlx_native.decision import PythonDecisionBackend
from graph_model.mlx_native.policy import PolicyOutput
from graph_model.models import RunState
from graph_model.provider import MockProvider
from graph_model.runtime import GraphRuntime, GraphRuntimeError, valid_outgoing_edges
from graph_model.store import SQLiteRunStore


class InvalidEdgeLovingPolicy:
    identity = "invalid-edge-loving-test-policy"

    def __init__(self, edge_count: int, abort_index: int) -> None:
        self.edge_count = edge_count
        self.abort_index = abort_index

    def predict(self, features):
        del features
        edge_logits = [0.0] * self.edge_count
        edge_logits[self.abort_index] = 1_000_000.0
        return PolicyOutput(
            route_logits=(0.0, 0.0, 0.0),
            edge_logits=tuple(edge_logits),
            stop_logits=(0.0, 0.0, 0.0, 1_000_000.0),
            success_value=0.75,
            cost=(100.0, 1.0, 2.0),
        )


def _controller(graph=None, policy=None) -> MLXGraphController:
    graph = graph or load_default_graph()
    return MLXGraphController(
        graph=graph,
        decision_backend=PythonDecisionBackend(),
        policy=policy,
    )


def test_mlx_controller_routes_only_to_named_routes() -> None:
    graph = load_default_graph()
    state = RunState.new(graph=graph, task="quick fix: rename one variable")
    decision = _controller(graph).select_route(state.task, state)
    assert decision.route == "fast"
    assert set(decision.probabilities) == {"fast", "deep", "repair"}
    assert decision.source == "mlx-hardcoded-route"


def test_hard_mask_blocks_a_policy_from_selecting_an_invalid_edge() -> None:
    graph = load_default_graph()
    baseline = _controller(graph)
    abort_index = next(
        index
        for index, key in enumerate(baseline.tables.edge_keys)
        if key.startswith("tests->abort:")
    )
    controller = _controller(
        graph,
        policy=InvalidEdgeLovingPolicy(len(baseline.tables.edge_keys), abort_index),
    )
    state = RunState.new(graph=graph, task="quick fix: rename one variable")
    state.current_node = "context"
    state.data.update({"route": "fast", "context_ready": True, "verdict": "pending"})
    candidates = valid_outgoing_edges(graph, state, "context")
    stop = controller.select_stop(
        graph=graph,
        state=state,
        node_id="context",
        candidates=candidates,
    )
    edge = controller.select_edge(
        graph=graph,
        state=state,
        node_id="context",
        candidates=candidates,
        stop=stop,
    )
    assert edge.edge.target == "implement"
    assert edge.allowed_edge_keys == (candidates[0].key,)
    assert edge.probabilities[candidates[0].key] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_runtime_records_mlx_stop_and_edge_decisions(tmp_path: Path) -> None:
    graph = load_default_graph()
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    runtime = GraphRuntime(
        graph=graph,
        store=store,
        provider=MockProvider(),
        controller=_controller(graph),
    )
    state = await runtime.run("quick fix: rename one variable", run_id="mlx-trace")
    assert state.status == "completed"
    assert state.completed_nodes == ["intake", "context", "implement", "tests", "review", "finish"]
    intake = next(
        event
        for event in store.events("mlx-trace")
        if event["event_type"] == "node_completed" and event["node_id"] == "intake"
    )
    assert intake["payload"]["edge_decision"]["target_node"] == "context"
    assert intake["payload"]["edge_decision"]["source"] == "mlx-hard-masked-edge"
    assert intake["payload"]["stop_decision"]["action"] == "continue"


@pytest.mark.asyncio
async def test_resume_rejects_a_different_controller(tmp_path: Path) -> None:
    graph = load_default_graph()
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    first = GraphRuntime(
        graph=graph,
        store=store,
        provider=MockProvider(),
        controller=_controller(graph),
    )
    await first.run(
        "implement a production feature",
        run_id="controller-resume",
        stop_after_steps=2,
    )
    different = GraphRuntime(
        graph=graph,
        store=store,
        provider=MockProvider(),
        controller=DeterministicGraphController(),
    )
    with pytest.raises(GraphRuntimeError, match="resume controller"):
        await different.run(run_id="controller-resume")
