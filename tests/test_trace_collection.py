from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from graph_model.graph import load_default_graph
from graph_model.mlx_native.controller import MLXGraphController
from graph_model.mlx_native.decision import PythonDecisionBackend
from graph_model.provider import ModelProvider
from graph_model.store import SQLiteRunStore
from graph_model.trace_collection import (
    RepositoryTraceTask,
    collect_repository_traces,
    read_repository_trace_manifest,
    write_trace_collection_summary,
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test User")
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (root / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "initial")
    return root


CORRECT_PATCH = """diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""


class _TraceProvider(ModelProvider):
    @property
    def identity(self) -> dict[str, str]:
        return {"kind": "trace-test", "version": "1"}

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        del temperature
        if "planning node" in system:
            payload: dict[str, Any] = {
                "steps": ["inspect calc.py", "replace subtraction with addition", "run tests"],
                "risks": [],
                "acceptance_tests": ["pytest passes"],
            }
        elif "repository patch proposal node" in system:
            payload = {
                "summary": "Use addition",
                "patch": CORRECT_PATCH,
                "assumptions": [],
                "no_changes_needed": False,
            }
        elif "semantic verifier" in system:
            parsed = json.loads(user)
            assert parsed["test_report"]["verdict"] == "pass"
            payload = {
                "verdict": "pass",
                "reasons": ["tests pass and the requested behavior is present"],
                "confidence": 0.99,
            }
        else:
            raise AssertionError(f"unexpected system prompt: {system}")
        return payload, max(1, len(user) // 4), 32


def test_trace_manifest_parsing_is_deterministic_and_rejects_duplicates(tmp_path: Path) -> None:
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(
        "# repository traces\n"
        + json.dumps(
            {
                "repo": "/tmp/project-a",
                "task": "Fix the failing unit test",
                "test_commands": ["python3 -m pytest -q"],
                "tags": ["unit", "repair"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    first = read_repository_trace_manifest(manifest, run_prefix="sample")
    second = read_repository_trace_manifest(manifest, run_prefix="sample")
    assert first == second
    assert first[0].run_id.startswith("sample-0001-")
    assert first[0].test_commands == ("python3 -m pytest -q",)
    assert first[0].tags == ("unit", "repair")

    duplicate = tmp_path / "duplicates.jsonl"
    duplicate.write_text(
        json.dumps({"run_id": "same", "repo": "/tmp/a", "task": "one"})
        + "\n"
        + json.dumps({"run_id": "same", "repo": "/tmp/b", "task": "two"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate trace run_id"):
        read_repository_trace_manifest(duplicate)


@pytest.mark.asyncio
async def test_trace_collection_runs_real_repository_and_writes_summary(tmp_path: Path) -> None:
    source = _repo(tmp_path)
    task = RepositoryTraceTask(
        run_id="trace-repository",
        task="quick fix: correct the failing add function",
        repo=str(source),
        test_commands=(f"{sys.executable} -m pytest -q",),
        tags=("fixture", "fast-path"),
    )
    graph = load_default_graph()
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    controller = MLXGraphController(
        graph=graph,
        decision_backend=PythonDecisionBackend(),
    )
    summary = await collect_repository_traces(
        tasks=[task],
        graph=graph,
        store=store,
        provider=_TraceProvider(),
        controller=controller,
        workspace_home=tmp_path / "worktrees",
        artifact_root=tmp_path / "artifacts",
    )

    assert summary["requested_tasks"] == 1
    assert summary["tasks"] == 1
    assert summary["status_counts"] == {"completed": 1}
    result = summary["results"][0]
    assert result["run_id"] == "trace-repository"
    assert result["status"] == "completed"
    assert result["tags"] == ["fixture", "fast-path"]
    assert result["hidden_artifacts"] == 0
    assert "finish" in result["completed_nodes"]
    assert (source / "calc.py").read_text(encoding="utf-8").endswith("return a - b\n")

    output = write_trace_collection_summary(summary, tmp_path / "summary.json")
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["status_counts"] == {"completed": 1}

    resumed = await collect_repository_traces(
        tasks=[task],
        graph=graph,
        store=store,
        provider=_TraceProvider(),
        controller=controller,
        workspace_home=tmp_path / "worktrees",
        artifact_root=tmp_path / "artifacts",
    )
    assert resumed["results"][0]["existing"] is True
