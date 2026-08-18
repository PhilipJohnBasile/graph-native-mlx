from pathlib import Path

import pytest

from graph_model.graph import load_default_graph
from graph_model.models import Budget
from graph_model.provider import MockProvider, OpenAICompatibleProvider
from graph_model.runtime import BudgetExceeded, GraphRuntime, GraphRuntimeError
from graph_model.store import SQLiteRunStore


class FailFirstReviewProvider(MockProvider):
    def __init__(self) -> None:
        self.review_calls = 0

    async def complete_json(self, *, system: str, user: str, temperature=None):
        if "semantic verifier" in system:
            self.review_calls += 1
            verdict = "fail" if self.review_calls == 1 else "pass"
            return {
                "verdict": verdict,
                "reasons": ["intentional test verdict"],
                "confidence": 0.9,
            }, max(1, len(user) // 4), 12
        return await super().complete_json(system=system, user=user, temperature=temperature)


@pytest.mark.asyncio
async def test_fast_path_completes_without_plan(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    runtime = GraphRuntime(graph=load_default_graph(), store=store, provider=MockProvider())
    state = await runtime.run("quick fix: rename one variable")
    assert state.status == "completed"
    assert state.data["route"] == "fast"
    assert "plan" not in state.completed_nodes
    assert state.completed_nodes == ["intake", "context", "implement", "tests", "review", "finish"]


@pytest.mark.asyncio
async def test_bounded_repair_reuses_upstream_state(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    runtime = GraphRuntime(graph=load_default_graph(), store=store, provider=MockProvider())
    state = await runtime.run("fix failing CI [force-fail-once]")
    assert state.status == "completed"
    assert state.data["repair_count"] == 1
    assert state.completed_nodes.count("context") == 1
    assert state.completed_nodes.count("plan") == 1
    assert state.completed_nodes.count("tests") == 2
    assert state.completed_nodes.count("repair") == 1


@pytest.mark.asyncio
async def test_plan_revision_back_edge_is_bounded_and_reaches_execution(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    runtime = GraphRuntime(graph=load_default_graph(), store=store, provider=MockProvider())
    state = await runtime.run("implement a production feature [bad-plan]")
    assert state.status == "completed"
    assert state.completed_nodes.count("plan") == 2
    assert state.completed_nodes.count("plan_check") == 2
    assert state.data["plan_revision_count"] == 1


@pytest.mark.asyncio
async def test_semantic_review_can_repair_and_review_again(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    provider = FailFirstReviewProvider()
    runtime = GraphRuntime(graph=load_default_graph(), store=store, provider=provider)
    state = await runtime.run("quick fix: rename one variable")
    assert state.status == "completed"
    assert provider.review_calls == 2
    assert state.completed_nodes.count("review") == 2
    assert state.completed_nodes.count("tests") == 2
    assert state.completed_nodes.count("repair") == 1


@pytest.mark.asyncio
async def test_always_failing_verifier_reaches_explicit_abort(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    runtime = GraphRuntime(graph=load_default_graph(), store=store, provider=MockProvider())
    state = await runtime.run("fix broken regression [force-fail-always]")
    assert state.status == "failed"
    assert state.completed_nodes[-1] == "abort"
    assert state.completed_nodes.count("tests") == 3
    assert state.data["repair_count"] == 2


@pytest.mark.asyncio
async def test_checkpoint_resume_starts_at_next_node(tmp_path: Path) -> None:
    db = tmp_path / "runs.sqlite3"
    store = SQLiteRunStore(db)
    runtime = GraphRuntime(graph=load_default_graph(), store=store, provider=MockProvider())
    partial = await runtime.run(
        "implement a production feature",
        run_id="resume-test",
        stop_after_steps=3,
    )
    assert partial.status == "running"
    assert partial.current_node == "plan_check"

    resumed = await runtime.run(run_id="resume-test")
    assert resumed.status == "completed"
    assert resumed.completed_nodes.count("intake") == 1
    assert resumed.completed_nodes.count("context") == 1
    assert resumed.completed_nodes.count("plan") == 1


@pytest.mark.asyncio
async def test_resume_rejects_a_different_model_provider(tmp_path: Path) -> None:
    db = tmp_path / "runs.sqlite3"
    store = SQLiteRunStore(db)
    first = GraphRuntime(graph=load_default_graph(), store=store, provider=MockProvider())
    await first.run("implement a production feature", run_id="provider-test", stop_after_steps=2)

    different = OpenAICompatibleProvider(base_url="http://127.0.0.1:9999/v1", model="other")
    resumed = GraphRuntime(graph=load_default_graph(), store=store, provider=different)
    with pytest.raises(GraphRuntimeError, match="resume provider"):
        await resumed.run(run_id="provider-test")


@pytest.mark.asyncio
async def test_call_budget_is_checked_for_the_next_node_kind(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    runtime = GraphRuntime(graph=load_default_graph(), store=store, provider=MockProvider())
    with pytest.raises(BudgetExceeded, match="llm_calls 0/0"):
        await runtime.run("quick fix: rename one variable", budget=Budget(max_llm_calls=0))
    state = store.list_runs()[0]
    assert state.completed_nodes == ["intake", "context"]
    assert state.status == "failed"
