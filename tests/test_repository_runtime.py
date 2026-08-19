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


def _multifile_repo(tmp_path: Path) -> Path:
    root = tmp_path / "multifile-repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test User")
    (root / "email_local.py").write_text(
        "def normalize_local(value: str) -> str:\n    return value\n",
        encoding="utf-8",
    )
    (root / "email_domain.py").write_text(
        "def normalize_domain(value: str) -> str:\n    return value\n",
        encoding="utf-8",
    )
    (root / "email_service.py").write_text(
        "from email_domain import normalize_domain\n"
        "from email_local import normalize_local\n\n"
        "def canonical_email(local: str, domain: str) -> str:\n"
        "    return f\"{normalize_local(local)}@{normalize_domain(domain)}\"\n",
        encoding="utf-8",
    )
    (root / "test_email_service.py").write_text(
        "from email_service import canonical_email\n\n"
        "def test_canonical_email_trims_and_lowercases_both_parts():\n"
        "    assert canonical_email(' Alice ', ' Example.COM ') == 'alice@example.com'\n",
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "initial")
    return root


MULTIFILE_PATCH_ENVELOPE = """GRAPH_PATCH_V1
GRAPH_PATCH_META_BEGIN
{"summary":"Normalize both email components","assumptions":["Tests are immutable"],"no_changes_needed":false}
GRAPH_PATCH_META_END
GRAPH_PATCH_DIFF_BEGIN
diff --git a/email_local.py b/email_local.py
--- a/email_local.py
+++ b/email_local.py
@@ -1,2 +1,2 @@
 def normalize_local(value: str) -> str:
-    return value
+    return value.strip().lower()
diff --git a/email_domain.py b/email_domain.py
--- a/email_domain.py
+++ b/email_domain.py
@@ -1,2 +1,2 @@
 def normalize_domain(value: str) -> str:
-    return value
+    return value.strip().lower()
GRAPH_PATCH_DIFF_END
"""


class MultiFilePatchEnvelopeProvider(ModelProvider):
    @property
    def identity(self) -> dict[str, str]:
        return {"kind": "multifile-patch-envelope", "version": "1"}

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
                "steps": [
                    "inspect both normalization modules",
                    "make one coherent two-file patch",
                    "run tests",
                ],
                "risks": ["keep tests unchanged"],
                "acceptance_tests": ["pytest passes"],
            }
        elif "semantic verifier" in system:
            parsed = json.loads(user)
            assert parsed["test_report"]["verdict"] == "pass"
            payload = {
                "verdict": "pass",
                "reasons": ["both source modules changed and tests pass"],
                "confidence": 0.99,
            }
        else:
            raise AssertionError(f"unexpected JSON system prompt: {system}")
        return payload, max(1, len(user) // 4), 32

    async def complete_patch(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        del user, temperature
        assert "repository patch proposal node" in system
        from graph_model.provider import _parse_patch_proposal

        return _parse_patch_proposal(MULTIFILE_PATCH_ENVELOPE), 128, 192


@pytest.mark.asyncio
async def test_real_repository_graph_applies_patch_native_multifile_change(
    tmp_path: Path,
) -> None:
    source = _multifile_repo(tmp_path)
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
        provider=MultiFilePatchEnvelopeProvider(),
    )

    state = await runtime.run(
        "Refactor email canonicalization across both source modules, preserve tests, and verify.",
        run_id="real-multifile-envelope",
        initial_data=initial_data,
    )

    assert state.status == "completed"
    assert state.data["test_report"]["verdict"] == "pass"
    assert state.data["review"]["verdict"] == "pass"
    assert sorted(state.data["test_report"]["changed_files"]) == ["email_domain.py", "email_local.py"]

    workspace = RepositoryWorkspace.from_state_data(state.data, run_id=state.run_id)
    assert workspace is not None
    assert "strip().lower()" in (workspace.active_root / "email_local.py").read_text(
        encoding="utf-8"
    )
    assert "strip().lower()" in (workspace.active_root / "email_domain.py").read_text(
        encoding="utf-8"
    )
    assert "return value\n" in (source / "email_local.py").read_text(encoding="utf-8")
    assert "return value\n" in (source / "email_domain.py").read_text(encoding="utf-8")

    manifest = state.artifacts["verified-patch.json"]
    patch = Path(manifest["path"]).read_text(encoding="utf-8")
    assert patch.count("diff --git") == 2
    assert "email_local.py" in patch
    assert "email_domain.py" in patch


def _pagination_repo(tmp_path: Path) -> Path:
    root = tmp_path / "pagination-repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test User")
    (root / "paging").mkdir()
    (root / "paging/__init__.py").write_text("", encoding="utf-8")
    (root / "paging/cursor.py").write_text(
        "def decode_cursor(value: str | None) -> int:\n"
        "    if value is None:\n"
        "        return 1\n"
        "    return int(value)\n",
        encoding="utf-8",
    )
    (root / "paging/query.py").write_text(
        "from .cursor import decode_cursor\n\n"
        "def query_offset(params: dict[str, str]) -> int:\n"
        "    return decode_cursor(params.get('cursor'))\n",
        encoding="utf-8",
    )
    (root / "paging/service.py").write_text(
        "def page_start(cursor: str | None) -> int:\n"
        "    return int(cursor or 1)\n",
        encoding="utf-8",
    )
    (root / "test_paging.py").write_text(
        "import pytest\n\n"
        "from paging.cursor import decode_cursor\n"
        "from paging.query import query_offset\n"
        "from paging.service import page_start\n\n"
        "def test_contract():\n"
        "    assert decode_cursor(None) == 0\n"
        "    assert decode_cursor('') == 0\n"
        "    assert query_offset({}) == 0\n"
        "    assert page_start(None) == 0\n"
        "    assert decode_cursor('7') == 7\n"
        "    with pytest.raises(ValueError):\n"
        "        decode_cursor('-1')\n",
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "initial")
    return root


PAGINATION_PATCH = """diff --git a/paging/cursor.py b/paging/cursor.py
--- a/paging/cursor.py
+++ b/paging/cursor.py
@@ -1,4 +1,7 @@
 def decode_cursor(value: str | None) -> int:
-    if value is None:
-        return 1
-    return int(value)
+    if value is None or value == "":
+        return 0
+    offset = int(value)
+    if offset < 0:
+        raise ValueError("cursor offset must be non-negative")
+    return offset
diff --git a/paging/service.py b/paging/service.py
--- a/paging/service.py
+++ b/paging/service.py
@@ -1,2 +1,4 @@
+from .cursor import decode_cursor
+
 def page_start(cursor: str | None) -> int:
-    return int(cursor or 1)
+    return decode_cursor(cursor)
"""


class AppealedPaginationProvider(ModelProvider):
    def __init__(self) -> None:
        self.initial_reviews = 0
        self.appeals = 0

    @property
    def identity(self) -> dict[str, str]:
        return {"kind": "appealed-pagination", "version": "1"}

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
                "steps": ["update cursor decoder", "share it from service", "verify"],
                "risks": [],
                "acceptance_tests": ["pytest passes"],
            }
        elif "repository patch proposal node" in system:
            payload = {
                "summary": "Use one cursor decoder",
                "patch": PAGINATION_PATCH,
                "assumptions": [],
                "no_changes_needed": False,
            }
        elif "independent appeal verifier" in system:
            self.appeals += 1
            parsed = json.loads(user)
            assert parsed["contract_oracle"]["verdict"] == "pass"
            assert parsed["initial_review"]["verdict"] == "fail"
            payload = {
                "verdict": "pass",
                "reasons": [
                    "The oracle proves query_offset already used decode_cursor and page_start now does too."
                ],
                "confidence": 1.0,
            }
        elif "semantic verifier" in system:
            self.initial_reviews += 1
            payload = {
                "verdict": "fail",
                "reasons": [
                    "query_offset was not changed to use decode_cursor"
                ],
                "confidence": 1.0,
            }
        else:
            raise AssertionError(f"unexpected system prompt: {system}")
        return payload, max(1, len(user) // 4), 32


@pytest.mark.asyncio
async def test_authoritative_contract_oracle_prevents_false_semantic_repair(
    tmp_path: Path,
) -> None:
    source = _pagination_repo(tmp_path)
    initial_data = workspace_initial_data(
        source_root=source,
        workspace_home=tmp_path / "worktrees",
        artifact_root=tmp_path / "artifacts",
        test_commands=[f"{sys.executable} -m pytest -q"],
    )
    initial_data["contract_oracle"] = {
        "name": "pagination-shared-decoder",
        "authoritative": True,
        "checks": [
            {
                "kind": "allowed_changed_files",
                "paths": ["paging/cursor.py", "paging/service.py"],
                "required": ["paging/cursor.py", "paging/service.py"],
            },
            {
                "kind": "python_function_calls",
                "path": "paging/query.py",
                "function": "query_offset",
                "callee": "decode_cursor",
            },
            {
                "kind": "python_function_calls",
                "path": "paging/service.py",
                "function": "page_start",
                "callee": "decode_cursor",
            },
            {"kind": "tests_pass"},
            {"kind": "tests_unchanged"},
            {"kind": "files_end_newline"},
        ],
    }
    provider = AppealedPaginationProvider()
    runtime = GraphRuntime(
        graph=load_default_graph(),
        store=SQLiteRunStore(tmp_path / "runs.sqlite3"),
        provider=provider,
    )
    state = await runtime.run(
        "Implement a multi-file pagination cursor migration and preserve tests.",
        run_id="pagination-appeal",
        initial_data=initial_data,
    )

    assert state.status == "completed"
    assert state.data["repair_count"] == 0
    assert state.data["review"]["initial"]["verdict"] == "fail"
    assert state.data["review"]["contract_oracle"]["verdict"] == "pass"
    assert state.data["review"]["appeal"] is None
    assert state.data["review"]["adjudicated"] is True
    assert state.data["review"]["adjudication_mode"] == "authoritative-contract-oracle"
    assert provider.initial_reviews == 1
    assert provider.appeals == 0


@pytest.mark.asyncio
async def test_non_authoritative_contract_oracle_uses_independent_appeal(
    tmp_path: Path,
) -> None:
    source = _pagination_repo(tmp_path)
    initial_data = workspace_initial_data(
        source_root=source,
        workspace_home=tmp_path / "worktrees",
        artifact_root=tmp_path / "artifacts",
        test_commands=[f"{sys.executable} -m pytest -q"],
    )
    initial_data["contract_oracle"] = {
        "name": "pagination-shared-decoder",
        "checks": [
            {
                "kind": "python_function_calls",
                "path": "paging/query.py",
                "function": "query_offset",
                "callee": "decode_cursor",
            },
            {
                "kind": "python_function_calls",
                "path": "paging/service.py",
                "function": "page_start",
                "callee": "decode_cursor",
            },
            {"kind": "tests_pass"},
            {"kind": "tests_unchanged"},
            {"kind": "files_end_newline"},
        ],
    }
    provider = AppealedPaginationProvider()
    runtime = GraphRuntime(
        graph=load_default_graph(),
        store=SQLiteRunStore(tmp_path / "runs.sqlite3"),
        provider=provider,
    )
    state = await runtime.run(
        "Implement a multi-file pagination cursor migration and preserve tests.",
        run_id="pagination-independent-appeal",
        initial_data=initial_data,
    )

    assert state.status == "completed"
    assert state.data["repair_count"] == 0
    assert state.data["review"]["contract_oracle"]["verdict"] == "pass"
    assert state.data["review"]["appeal"]["verdict"] == "pass"
    assert state.data["review"]["adjudication_mode"] == "independent-appeal"
    assert provider.initial_reviews == 1
    assert provider.appeals == 1
