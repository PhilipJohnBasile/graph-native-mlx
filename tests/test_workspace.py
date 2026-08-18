from __future__ import annotations

import hashlib
import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

from graph_model.workspace import (
    CommandPolicyError,
    PatchError,
    RepositoryWorkspace,
    WorkspaceConfig,
    parse_bounded_command,
    validate_patch,
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
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "initial")
    return root


def _workspace(tmp_path: Path, root: Path, *, run_id: str = "run-1") -> RepositoryWorkspace:
    config = WorkspaceConfig(
        source_root=str(root),
        workspace_home=str(tmp_path / "worktrees"),
        artifact_root=str(tmp_path / "artifacts"),
        test_commands=[f"{sys.executable} -m pytest -q"],
    )
    workspace = RepositoryWorkspace(config, run_id=run_id)
    workspace.ensure_prepared()
    return workspace


PATCH = """diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""


def test_worktree_patch_is_idempotent_and_source_stays_clean(tmp_path: Path) -> None:
    source = _repo(tmp_path)
    workspace = _workspace(tmp_path, source)

    first = workspace.apply_patch(PATCH, idempotency_key="run-1:apply:one")
    replay = workspace.apply_patch(PATCH, idempotency_key="run-1:apply:one")

    assert first.applied is True
    assert first.replayed is False
    assert replay.replayed is True
    assert (workspace.active_root / "calc.py").read_text(encoding="utf-8").endswith(
        "return a + b\n"
    )
    assert (source / "calc.py").read_text(encoding="utf-8").endswith("return a - b\n")
    assert _git(source, "status", "--porcelain") == ""


def test_apply_recovers_when_patch_landed_before_commit_ledger(tmp_path: Path) -> None:
    source = _repo(tmp_path)
    workspace = _workspace(tmp_path, source)
    key = "run-1:apply:interrupted"
    workspace.apply_patch(PATCH, idempotency_key=key)

    operation_sha = hashlib.sha256(key.encode("utf-8")).hexdigest()
    committed = workspace.operations_dir / f"{operation_sha}.committed.json"
    committed.unlink()

    recovered = workspace.apply_patch(PATCH, idempotency_key=key)
    assert recovered.applied is False
    assert recovered.recovered_after_interruption is True
    assert committed.exists()


def test_verifier_ignores_common_python_cache_files(tmp_path: Path) -> None:
    source = _repo(tmp_path)
    workspace = _workspace(tmp_path, source)
    workspace.apply_patch(PATCH, idempotency_key="run-1:apply:test")

    report = workspace.run_tests()

    assert report["verdict"] == "pass"
    assert report["changed_files"] == ["calc.py"]
    assert report["commands"][0]["passed"] is True


def test_patch_policy_rejects_escape_sensitive_binary_and_renames(tmp_path: Path) -> None:
    source = _repo(tmp_path)
    with pytest.raises(PatchError, match="invalid relative path"):
        validate_patch(
            "diff --git a/../outside b/../outside\n",
            root=source,
            max_files=10,
        )
    with pytest.raises(PatchError, match="sensitive path"):
        validate_patch(
            "diff --git a/.env b/.env\n--- a/.env\n+++ b/.env\n",
            root=source,
            max_files=10,
        )
    with pytest.raises(PatchError, match="binary patches"):
        validate_patch(
            "diff --git a/data.bin b/data.bin\nGIT binary patch\n",
            root=source,
            max_files=10,
        )
    with pytest.raises(PatchError, match="rename and copy"):
        validate_patch(
            "diff --git a/a.py b/b.py\nrename from a.py\nrename to b.py\n",
            root=source,
            max_files=10,
        )


def test_command_policy_disables_shell_control_and_mutating_git(tmp_path: Path) -> None:
    source = _repo(tmp_path)
    with pytest.raises(CommandPolicyError, match="shell control"):
        parse_bounded_command(
            "python3 -m pytest && rm -rf .",
            cwd=source,
            allowed_commands=["python3"],
        )
    with pytest.raises(CommandPolicyError, match="git subcommand"):
        parse_bounded_command(
            "git clean -fdx",
            cwd=source,
            allowed_commands=["git"],
        )


def test_verified_patch_promotion_is_hash_checked_and_idempotent(tmp_path: Path) -> None:
    source = _repo(tmp_path)
    workspace = _workspace(tmp_path, source)
    workspace.apply_patch(PATCH, idempotency_key="run-1:apply:promotion")
    manifest = workspace.export_patch()

    with pytest.raises(Exception, match="hash mismatch"):
        workspace.promote_verified_patch(manifest["path"], "0" * 64)

    first = workspace.promote_verified_patch(manifest["path"], manifest["sha256"])
    second = workspace.promote_verified_patch(manifest["path"], manifest["sha256"])

    assert first["status"] == "applied"
    assert first["replayed"] is False
    assert second["replayed"] is True
    assert (source / "calc.py").read_text(encoding="utf-8").endswith("return a + b\n")


def test_unified_diff_paths_must_match_diff_header(tmp_path: Path) -> None:
    source = _repo(tmp_path)
    mismatched = """diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/tests/test_calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""
    with pytest.raises(PatchError, match="does not match"):
        validate_patch(mismatched, root=source, max_files=10)


def test_run_id_cannot_escape_workspace_or_artifact_roots(tmp_path: Path) -> None:
    source = _repo(tmp_path)
    workspace = _workspace(tmp_path, source, run_id="../../outside/../../danger")

    assert workspace.active_root.is_relative_to((tmp_path / "worktrees").resolve())
    assert workspace.artifact_root.is_relative_to((tmp_path / "artifacts").resolve())
    assert ".." not in workspace.active_root.name
    assert ".." not in workspace.artifact_root.name


def test_workspace_and_artifact_roots_must_be_outside_source(tmp_path: Path) -> None:
    source = _repo(tmp_path)
    workspace_inside = RepositoryWorkspace(
        WorkspaceConfig(
            source_root=str(source),
            workspace_home=str(source / ".worktrees"),
            artifact_root=str(tmp_path / "artifacts"),
        ),
        run_id="inside-workspace",
    )
    with pytest.raises(Exception, match="workspace home must be outside"):
        workspace_inside.ensure_prepared()

    artifact_inside = RepositoryWorkspace(
        WorkspaceConfig(
            source_root=str(source),
            workspace_home=str(tmp_path / "worktrees"),
            artifact_root=str(source / ".artifacts"),
        ),
        run_id="inside-artifacts",
    )
    with pytest.raises(Exception, match="artifact root must be outside"):
        artifact_inside.ensure_prepared()


def test_verifier_reports_tracked_workspace_mutation(tmp_path: Path) -> None:
    source = _repo(tmp_path)
    command = (
        f"{sys.executable} -c \"open('calc.py', 'a', encoding='utf-8').write('# mutated\\\\n')\""
    )
    config = WorkspaceConfig(
        source_root=str(source),
        workspace_home=str(tmp_path / "worktrees"),
        artifact_root=str(tmp_path / "artifacts"),
        test_commands=[command],
    )
    workspace = RepositoryWorkspace(config, run_id="mutating-test")
    workspace.ensure_prepared()
    workspace.apply_patch(PATCH, idempotency_key="mutating-test:apply")

    report = workspace.run_tests()

    assert report["verdict"] == "fail"
    assert report["workspace_mutated"] is True
    assert report["workspace_fingerprint_before"] != report["workspace_fingerprint"]


def test_command_resolution_ignores_repository_path_spoofing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _repo(tmp_path)
    fake = source / "python3"
    fake.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{source}:{os.environ.get('PATH', '')}")

    argv = parse_bounded_command(
        "python3 -c 'print(1)'",
        cwd=source,
        allowed_commands=["python3"],
    )

    assert Path(argv[0]).resolve() != fake.resolve()
    assert not Path(argv[0]).resolve().is_relative_to(source.resolve())


def test_untrusted_absolute_python_path_is_rejected(tmp_path: Path) -> None:
    source = _repo(tmp_path)
    fake_dir = tmp_path / "fake-bin"
    fake_dir.mkdir()
    fake = fake_dir / "python3"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)

    with pytest.raises(CommandPolicyError, match="trusted allowlisted path"):
        parse_bounded_command(
            f"{fake} -c 'print(1)'",
            cwd=source,
            allowed_commands=["python3"],
        )


def test_test_detection_requires_an_actual_package_script(tmp_path: Path) -> None:
    from graph_model.workspace import detect_test_commands, verification_commands

    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"build": "vite build"}}), encoding="utf-8"
    )
    assert detect_test_commands(tmp_path) == []
    assert verification_commands(tmp_path) == ["git diff --check"]

    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest run"}}), encoding="utf-8"
    )
    assert detect_test_commands(tmp_path) == ["npm test"]
    assert verification_commands(tmp_path, ["npm test"]) == [
        "git diff --check",
        "npm test",
    ]


def test_cleanup_removes_dirty_worktree_but_retains_patch_for_promotion(
    tmp_path: Path,
) -> None:
    source = _repo(tmp_path)
    workspace = _workspace(tmp_path, source, run_id="cleanup-run")
    workspace.apply_patch(PATCH, idempotency_key="cleanup-run:apply")
    manifest = workspace.export_patch()
    active = workspace.active_root

    with pytest.raises(Exception, match="uncommitted changes"):
        workspace.cleanup_worktree(force=False)

    report = workspace.cleanup_worktree(force=True)
    assert report["status"] == "removed"
    assert not active.exists()
    assert Path(manifest["path"]).is_file()

    promoted = workspace.promote_verified_patch(manifest["path"], manifest["sha256"])
    assert promoted["status"] == "applied"
    assert (source / "calc.py").read_text(encoding="utf-8").endswith("return a + b\n")
