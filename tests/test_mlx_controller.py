import hashlib
from pathlib import Path

import pytest

from graph_model.controller import DeterministicGraphController
from graph_model.graph import load_default_graph
from graph_model.mlx_native.controller import MLXGraphController
from graph_model.mlx_native.decision import PythonDecisionBackend
from graph_model.mlx_native.hidden_state import (
    HiddenStateArtifactStore,
    HiddenStateCapture,
    HiddenStateObservation,
)
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


class RecordingHiddenSource:
    hidden_state_identity = "recording-hidden-source"
    hidden_capture_enabled = True

    def __init__(self, root: Path) -> None:
        self.store = HiddenStateArtifactStore(root)
        self.calls = []
        self.affinity_calls = 0

    def run_on_affinity(self, function, *args):
        self.affinity_calls += 1
        return function(*args)

    def capture_policy_hidden(self, *, state, node_id, decision_type):
        marker = f"{state.run_id}:{state.step_count}:{node_id}:{decision_type}:{state.data.get('verdict')}"
        digest = hashlib.sha256(marker.encode()).hexdigest()
        self.calls.append((state.step_count, node_id, decision_type, state.data.get("verdict")))
        offset = len(self.calls) / 10.0
        capture = HiddenStateCapture(
            features=tuple(offset + index / 100.0 for index in range(8)),
            model_fingerprint="a" * 64,
            extractor_schema_hash="b" * 64,
            raw_hidden_size=8,
            raw_vector_size=8,
            prompt_tokens=16,
            task_sha256=hashlib.sha256(state.task.encode()).hexdigest(),
            prompt_sha256=digest,
            core_path="model.model",
            layer_labels=("final",),
            pooling="last-token",
        )
        reference = self.store.write(capture)
        return HiddenStateObservation(features=capture.features, reference=reference)


class HiddenAwarePolicy:
    identity = "hidden-aware-policy"
    requires_hidden = True

    def __init__(self, edge_count: int) -> None:
        self.edge_count = edge_count
        self.calls = []

    def predict(self, features, *, hidden_features=None, hidden_metadata=None):
        assert hidden_features is not None
        assert hidden_metadata is not None
        self.calls.append((tuple(features), tuple(hidden_features), dict(hidden_metadata)))
        return PolicyOutput(
            route_logits=(0.0, 0.0, 0.0),
            edge_logits=tuple(0.0 for _ in range(self.edge_count)),
            stop_logits=(0.0, 0.0, 0.0, 0.0),
            success_value=0.5,
            cost=(0.1, 0.2, 0.3),
            hidden_used=True,
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
    assert state.completed_nodes == [
        "intake", "context", "implement", "apply", "tests", "review", "finish"
    ]
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



def test_hidden_context_is_state_specific_and_reused_for_stop_and_edge(tmp_path: Path) -> None:
    graph = load_default_graph()
    source = RecordingHiddenSource(tmp_path / "hidden")
    policy = HiddenAwarePolicy(edge_count=len(graph.edges))
    controller = MLXGraphController(
        graph=graph,
        decision_backend=PythonDecisionBackend(),
        policy=policy,
        hidden_state_source=source,
        capture_hidden=True,
    )
    state = RunState.new(graph=graph, task="Fix the failing test", run_id="hidden-context")

    route = controller.select_route(state.task, state)
    assert len(source.calls) == 1
    assert source.calls[0][2] == "route"
    assert route.policy_metrics["hidden_state"]["sha256"]

    state.current_node = "context"
    state.step_count = 1
    state.updated_at += 1.0
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
    assert len(source.calls) == 2
    assert source.calls[-1][2] == "transition"
    assert (
        stop.policy_metrics["hidden_state"]["sha256"]
        == edge.policy_metrics["hidden_state"]["sha256"]
    )

    state.step_count += 1
    state.updated_at += 1.0
    state.data["verdict"] = "fail"
    controller.select_stop(
        graph=graph,
        state=state,
        node_id="context",
        candidates=candidates,
    )
    assert len(source.calls) == 3
    assert source.calls[-1][-1] == "fail"
    assert len(policy.calls) == 3
    assert source.affinity_calls >= 6


def test_policy_requiring_hidden_state_fails_without_a_source() -> None:
    graph = load_default_graph()
    with pytest.raises(ValueError, match="requires Qwen hidden-state"):
        MLXGraphController(
            graph=graph,
            decision_backend=PythonDecisionBackend(),
            policy=HiddenAwarePolicy(edge_count=len(graph.edges)),
        )


@pytest.mark.asyncio
async def test_runtime_accounts_for_hidden_policy_prefill_work(tmp_path: Path) -> None:
    graph = load_default_graph()
    source = RecordingHiddenSource(tmp_path / "hidden-runtime")
    policy = HiddenAwarePolicy(edge_count=len(graph.edges))
    controller = MLXGraphController(
        graph=graph,
        decision_backend=PythonDecisionBackend(),
        policy=policy,
        hidden_state_source=source,
        capture_hidden=True,
    )
    runtime = GraphRuntime(
        graph=graph,
        store=SQLiteRunStore(tmp_path / "policy-usage.sqlite3"),
        provider=MockProvider(),
        controller=controller,
    )
    state = await runtime.run(
        "quick fix: rename one variable",
        run_id="policy-usage",
    )

    # Fast path: one route context plus one transition context for every
    # non-terminal node. Stop and edge share each transition context.
    assert state.status == "completed"
    assert state.metrics.policy_calls == 7
    assert state.metrics.policy_prompt_tokens == 7 * 16
    assert len(source.calls) == 7
    assert state.metrics.total_tokens == (
        state.metrics.prompt_tokens
        + state.metrics.completion_tokens
        + state.metrics.policy_prompt_tokens
    )
