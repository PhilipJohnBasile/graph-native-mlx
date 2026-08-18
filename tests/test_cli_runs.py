from __future__ import annotations

import argparse
import json
from pathlib import Path

from graph_model.cli import _runs_command, _trace_command, build_parser
from graph_model.graph import load_default_graph
from graph_model.models import RunState
from graph_model.store import SQLiteRunStore


def _seed_runs(path: Path) -> None:
    graph = load_default_graph()
    store = SQLiteRunStore(path)

    older = RunState.new(graph=graph, task="older task", run_id="older-run")
    store.create_run(older)
    older.status = "failed"
    older.current_node = "abort"
    older.step_count = 3
    older.error = "fixture failure"
    store.save_terminal_event(older, "run_failed")

    latest = RunState.new(graph=graph, task="latest task", run_id="latest-run")
    store.create_run(latest)
    latest.status = "completed"
    latest.current_node = "finish"
    latest.step_count = 9
    latest.data["route"] = "deep"
    store.save_terminal_event(latest, "run_completed")


def test_runs_command_lists_latest_first(tmp_path: Path, capsys) -> None:
    database = tmp_path / "runs.sqlite3"
    _seed_runs(database)

    code = _runs_command(argparse.Namespace(db=str(database), limit=10, status=None))
    rows = json.loads(capsys.readouterr().out)

    assert code == 0
    assert [row["run_id"] for row in rows] == ["latest-run", "older-run"]
    assert rows[0]["status"] == "completed"
    assert rows[0]["route"] == "deep"


def test_trace_latest_resolves_latest_run_id(tmp_path: Path, capsys) -> None:
    database = tmp_path / "runs.sqlite3"
    _seed_runs(database)

    code = _trace_command(
        argparse.Namespace(db=str(database), run_id=None, latest=True)
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["run_id"] == "latest-run"
    assert payload["events"][-1]["event_type"] == "run_completed"


def test_trace_parser_requires_run_id_or_latest() -> None:
    parser = build_parser()
    args = parser.parse_args(["trace", "--latest"])

    assert args.latest is True
    assert args.run_id is None
