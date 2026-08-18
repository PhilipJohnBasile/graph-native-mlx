from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from graph_model.graph import load_default_graph
from graph_model.provider import ModelProvider
from graph_model.runtime import GraphRuntime
from graph_model.store import SQLiteRunStore
from graph_model.workspace import RepositoryWorkspace, workspace_initial_data


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


INITIAL_WRONG_PATCH = """diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a * b
"""

REPAIR_PATCH = """diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a * b
+    return a + b
"""


class RepairingRepositoryProvider(ModelProvider):
    @property
    def identity(self) -> dict[str, str]:
        return {"kind": "scripted-repository", "version": "1"}

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
                "steps": ["inspect calc.py", "correct add", "run tests"],
                "risks": [],
                "acceptance_tests": ["pytest passes"],
            }
        elif "repository patch proposal node" in system:
            payload = {
                "summary": "Attempt to correct add",
                "patch": INITIAL_WRONG_PATCH,
                "assumptions": [],
                "no_changes_needed": False,
            }
        elif "failure diagnosis node" in system:
            parsed = json.loads(user)
            assert parsed["test_report"]["verdict"] == "fail"
            payload = {
                "root_causes": ["multiplication was used instead of addition"],
                "repair_steps": ["replace a * b with a + b"],
                "files_to_change": ["calc.py"],
                "evidence": ["test_add expected 5 but received 6"],
            }
        elif "repair patch node" in system:
            payload = {
                "summary": "Use addition",
                "patch": REPAIR_PATCH,
                "assumptions": [],
                "no_changes_needed": False,
            }
        elif "semantic verifier" in system:
            parsed = json.loads(user)
            assert parsed["test_report"]["verdict"] == "pass"
            payload = {
                "verdict": "pass",
                "reasons": ["requested behavior is implemented and tests pass"],
                "confidence": 0.99,
            }
        else:
            raise AssertionError(f"unexpected system prompt: {system}")
        return payload, max(1, len(user) // 4), 32


@pytest.mark.asyncio
async def test_real_repository_graph_repairs_locally_and_exports_verified_patch(
    tmp_path: Path,
) -> None:
    source = _repo(tmp_path)
    initial_data = workspace_initial_data(
        source_root=source,
        workspace_home=tmp_path / "worktrees",
        artifact_root=tmp_path / "artifacts",
        test_commands=[f"{sys.executable} -m pytest -q"],
    )
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    runtime = GraphRuntime(
        graph=load_default_graph(),
        store=store,
        provider=RepairingRepositoryProvider(),
    )

    state = await runtime.run(
        "Fix the failing add function regression and verify it",
        run_id="real-repair",
        initial_data=initial_data,
    )

    assert state.status == "completed"
    assert state.data["repair_count"] == 1
    assert state.completed_nodes.count("context") == 1
    assert state.completed_nodes.count("plan") == 1
    assert state.completed_nodes.count("apply") == 2
    assert state.completed_nodes.count("tests") == 2
    assert state.completed_nodes.count("diagnose") == 1
    assert state.completed_nodes.count("repair") == 1
    assert state.data["test_report"]["verdict"] == "pass"

    workspace = RepositoryWorkspace.from_state_data(state.data, run_id=state.run_id)
    assert workspace is not None
    assert (workspace.active_root / "calc.py").read_text(encoding="utf-8").endswith(
        "return a + b\n"
    )
    assert (source / "calc.py").read_text(encoding="utf-8").endswith("return a - b\n")

    manifest = state.artifacts["verified-patch.json"]
    patch = Path(manifest["path"]).read_text(encoding="utf-8")
    assert "return a + b" in patch
    assert "return a * b" not in patch
    assert manifest["sha256"]
    assert state.output["workspace"]["promotion_required"] is True
