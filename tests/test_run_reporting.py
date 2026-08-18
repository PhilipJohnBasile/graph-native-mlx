from __future__ import annotations

from pathlib import Path

from graph_model.cli import build_parser
from graph_model.graph import load_default_graph
from graph_model.models import NodeResult, RunState
from graph_model.run_reporting import build_run_report, list_run_summaries
from graph_model.store import SQLiteRunStore


def _completed_run(store: SQLiteRunStore, tmp_path: Path, *, run_id: str) -> RunState:
    graph = load_default_graph()
    patch_path = tmp_path / f"{run_id}.patch"
    patch_path.write_text("diff --git a/a.py b/a.py\n", encoding="utf-8")
    state = RunState.new(graph=graph, task="Fix the failing add function", run_id=run_id)
    store.create_run(state)
    state.current_node = "finish"
    state.status = "completed"
    state.completed_nodes = ["intake", "context", "plan", "implement", "tests", "review", "finish"]
    state.step_count = 1
    state.data.update(
        {
            "route": "deep",
            "repair_count": 0,
            "test_report": {
                "verdict": "pass",
                "workspace_mutated": False,
                "changed_files": ["a.py"],
                "commands": [
                    {
                        "command": "python -m pytest -q",
                        "exit_code": 0,
                        "passed": True,
                        "timed_out": False,
                        "duration_seconds": 0.2,
                    }
                ],
            },
            "review": {
                "verdict": "pass",
                "confidence": 0.97,
                "reasons": ["Tests passed and the patch is minimal."],
            },
            "workspace": {
                "mode": "worktree",
                "source_root": str(tmp_path / "source"),
                "active_root": str(tmp_path / "worktree"),
                "base_commit": "abc123",
            },
        }
    )
    state.artifacts["verified-patch.json"] = {
        "path": str(patch_path),
        "sha256": "f" * 64,
        "bytes": patch_path.stat().st_size,
        "changed_files": ["a.py"],
    }
    state.metrics.llm_calls = 2
    state.metrics.tool_calls = 3
    state.metrics.policy_calls = 4
    state.metrics.prompt_tokens = 100
    state.metrics.completion_tokens = 20
    state.metrics.policy_prompt_tokens = 30
    hidden = {
        "sha256": "a" * 64,
        "path": str(tmp_path / "hidden.json"),
        "feature_size": 256,
        "raw_hidden_size": 5120,
        "prompt_tokens": 451,
        "layer_labels": ["final"],
        "pooling": "last-token",
        "model_fingerprint": "b" * 64,
    }
    result = NodeResult(
        delta={
            "router": {
                "source": "mlx-hardcoded-route",
                "policy_metrics": {
                    "hidden_state": hidden,
                    "hidden_state_cache_hit": False,
                },
            }
        },
        progress_key="done",
    )
    store.commit_step(
        state=state,
        node_id="intake",
        input_hash="input",
        result=result,
        cached=False,
        event_payload={
            "edge_decision": {
                "source": "mlx-hard-masked-edge",
                "policy_metrics": {
                    "hidden_state": hidden,
                    "hidden_state_cache_hit": True,
                },
            }
        },
    )
    store.save_terminal_event(state, "run_completed")
    return state


def test_run_reporting_lists_latest_and_builds_concise_evidence(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    first = _completed_run(store, tmp_path, run_id="first")
    second = RunState.new(graph=load_default_graph(), task="Second task", run_id="second")
    store.create_run(second)

    assert store.latest_run().run_id == "second"
    assert store.latest_run(status="completed").run_id == first.run_id
    assert [state.run_id for state in store.list_runs(limit=1)] == ["second"]

    rows = list_run_summaries(store, limit=10, status="completed")
    assert len(rows) == 1
    assert rows[0]["run_id"] == "first"
    assert rows[0]["verified_patch"] is True
    assert rows[0]["tokens"] == 150

    report = build_run_report(store, "first")
    assert report["run"]["status"] == "completed"
    assert report["hidden_policy"]["unique_artifacts"] == 1
    assert report["hidden_policy"]["feature_sizes"] == [256]
    assert report["hidden_policy"]["raw_hidden_sizes"] == [5120]
    assert report["hidden_policy"]["cache_hits"] == 1
    assert report["verification"]["tests"]["verdict"] == "pass"
    assert report["verification"]["review"]["confidence"] == 0.97
    assert report["patch"]["exists"] is True


def test_cli_accepts_latest_summary_and_report() -> None:
    parser = build_parser()
    trace = parser.parse_args(["trace", "--latest", "--summary"])
    assert trace.command == "trace"
    assert trace.latest is True
    assert trace.summary is True

    report = parser.parse_args(["report", "--latest", "--latest-status", "completed"])
    assert report.command == "report"
    assert report.latest is True
    assert report.latest_status == "completed"
