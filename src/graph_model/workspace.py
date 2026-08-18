from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:  # pragma: no cover - Windows fallback is exercised only off POSIX
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

_FILE_LOCKS: dict[str, threading.RLock] = {}
_FILE_LOCKS_GUARD = threading.RLock()



class WorkspaceError(RuntimeError):
    """Base error for repository workspace operations."""


class PatchError(WorkspaceError):
    """Raised when a model-proposed patch is invalid or cannot be applied safely."""


class CommandPolicyError(WorkspaceError):
    """Raised when a configured verifier command violates the command policy."""


class WorkspaceMode(str, Enum):
    WORKTREE = "worktree"
    IN_PLACE = "in-place"


DEFAULT_ALLOWED_COMMANDS: tuple[str, ...] = (
    "git",
    "python",
    "python3",
    "pytest",
    "uv",
    "rye",
    "poetry",
    "tox",
    "nox",
    "npm",
    "pnpm",
    "yarn",
    "bun",
    "node",
    "cargo",
    "rustc",
    "go",
    "swift",
    "xcodebuild",
    "make",
    "cmake",
    "ctest",
    "ninja",
    "dotnet",
    "gradle",
    "gradlew",
    "mvn",
    "mvnw",
)

_BLOCKED_CONTROL_TOKENS = {
    "|",
    "||",
    "&&",
    ";",
    ">",
    ">>",
    "<",
    "<<",
    "2>",
    "2>>",
    "&",
    "`",
}

_GIT_READ_ONLY_SUBCOMMANDS = {
    "diff",
    "status",
    "ls-files",
    "rev-parse",
    "grep",
    "show",
    "log",
    "branch",
}

_EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".graph-model",
    ".idea",
    ".mypy_cache",
    ".next",
    ".nuxt",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "vendor",
}

_BINARY_SUFFIXES = {
    ".7z",
    ".a",
    ".avi",
    ".bin",
    ".bmp",
    ".class",
    ".dylib",
    ".eot",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".lockb",
    ".mov",
    ".mp3",
    ".mp4",
    ".o",
    ".otf",
    ".pdf",
    ".png",
    ".pyc",
    ".so",
    ".tar",
    ".tgz",
    ".ttf",
    ".wav",
    ".webm",
    ".webp",
    ".woff",
    ".woff2",
    ".xz",
    ".zip",
}

_PRIORITY_FILENAMES = {
    "cargo.toml",
    "go.mod",
    "makefile",
    "package.json",
    "package.swift",
    "pyproject.toml",
    "readme.md",
    "requirements.txt",
    "tox.ini",
}

_SENSITIVE_EXACT_NAMES = {
    ".env",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
}

_SENSITIVE_SUFFIXES = {
    ".jks",
    ".key",
    ".p12",
    ".pem",
    ".pfx",
}

_LANGUAGE_BY_SUFFIX = {
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".cs": "C#",
    ".css": "CSS",
    ".dart": "Dart",
    ".gd": "GDScript",
    ".go": "Go",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".html": "HTML",
    ".java": "Java",
    ".js": "JavaScript",
    ".json": "JSON",
    ".jsx": "JavaScript/JSX",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".lua": "Lua",
    ".m": "Objective-C",
    ".md": "Markdown",
    ".metal": "Metal",
    ".mm": "Objective-C++",
    ".php": "PHP",
    ".py": "Python",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".sh": "Shell",
    ".sql": "SQL",
    ".swift": "Swift",
    ".toml": "TOML",
    ".ts": "TypeScript",
    ".tsx": "TypeScript/TSX",
    ".vue": "Vue",
    ".xml": "XML",
    ".yaml": "YAML",
    ".yml": "YAML",
}


class WorkspaceConfig(BaseModel):
    """Serializable configuration stored in explicit graph state."""

    model_config = ConfigDict(extra="forbid")

    source_root: str
    mode: WorkspaceMode = WorkspaceMode.WORKTREE
    base_ref: str = "HEAD"
    active_root: str | None = None
    base_commit: str | None = None
    workspace_home: str | None = None
    artifact_root: str | None = None
    test_commands: list[str] = Field(default_factory=list)
    allowed_commands: list[str] = Field(default_factory=lambda: list(DEFAULT_ALLOWED_COMMANDS))
    command_timeout_seconds: float = Field(default=300.0, gt=0, le=7_200)
    max_command_output_bytes: int = Field(default=200_000, ge=1_024, le=10_000_000)
    max_context_files: int = Field(default=18, ge=1, le=100)
    max_context_file_bytes: int = Field(default=40_000, ge=1_024, le=1_000_000)
    max_context_bytes: int = Field(default=180_000, ge=4_096, le=4_000_000)
    max_patch_bytes: int = Field(default=500_000, ge=1_024, le=5_000_000)
    max_patch_files: int = Field(default=32, ge=1, le=500)
    allow_sensitive_paths: bool = False

    @field_validator("source_root")
    @classmethod
    def source_root_must_be_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_root cannot be empty")
        return value

    @field_validator("base_ref")
    @classmethod
    def base_ref_must_be_safe(cls, value: str) -> str:
        if not value.strip() or value.startswith("-") or any(ch in value for ch in "\r\n\x00"):
            raise ValueError("base_ref must be a non-option Git revision")
        return value

    @field_validator("test_commands")
    @classmethod
    def commands_must_be_nonempty(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value.strip():
                raise ValueError("test commands cannot be empty")
        return values

    @model_validator(mode="after")
    def validate_roots(self) -> "WorkspaceConfig":
        if self.active_root and self.mode == WorkspaceMode.IN_PLACE:
            source = Path(self.source_root).expanduser().resolve(strict=False)
            active = Path(self.active_root).expanduser().resolve(strict=False)
            if source != active:
                raise ValueError("in-place workspace active_root must equal source_root")
        return self


@dataclass(frozen=True)
class CommandResult:
    command: str
    argv: tuple[str, ...]
    exit_code: int | None
    duration_seconds: float
    timed_out: bool
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool

    @property
    def passed(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "argv": list(self.argv),
            "exit_code": self.exit_code,
            "duration_seconds": round(self.duration_seconds, 6),
            "timed_out": self.timed_out,
            "passed": self.passed,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
        }


@dataclass(frozen=True)
class PatchApplication:
    patch_sha256: str
    patch_artifact: str
    changed_files: tuple[str, ...]
    applied: bool
    replayed: bool
    recovered_after_interruption: bool
    before_fingerprint: str
    after_fingerprint: str
    diff_stat: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "patch_sha256": self.patch_sha256,
            "patch_artifact": self.patch_artifact,
            "changed_files": list(self.changed_files),
            "applied": self.applied,
            "replayed": self.replayed,
            "recovered_after_interruption": self.recovered_after_interruption,
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "diff_stat": self.diff_stat,
        }


def workspace_initial_data(
    *,
    source_root: str | Path,
    mode: str = WorkspaceMode.WORKTREE.value,
    base_ref: str = "HEAD",
    workspace_home: str | Path | None = None,
    artifact_root: str | Path | None = None,
    test_commands: Sequence[str] = (),
    allowed_commands: Sequence[str] = DEFAULT_ALLOWED_COMMANDS,
    command_timeout_seconds: float = 300.0,
    max_command_output_bytes: int = 200_000,
    max_context_files: int = 18,
    max_context_file_bytes: int = 40_000,
    max_context_bytes: int = 180_000,
    max_patch_bytes: int = 500_000,
    max_patch_files: int = 32,
    allow_sensitive_paths: bool = False,
) -> dict[str, Any]:
    config = WorkspaceConfig(
        source_root=str(Path(source_root).expanduser().resolve(strict=False)),
        mode=WorkspaceMode(mode),
        base_ref=base_ref,
        workspace_home=(
            str(Path(workspace_home).expanduser().resolve(strict=False))
            if workspace_home is not None
            else None
        ),
        artifact_root=(
            str(Path(artifact_root).expanduser().resolve(strict=False))
            if artifact_root is not None
            else None
        ),
        test_commands=list(test_commands),
        allowed_commands=list(allowed_commands),
        command_timeout_seconds=command_timeout_seconds,
        max_command_output_bytes=max_command_output_bytes,
        max_context_files=max_context_files,
        max_context_file_bytes=max_context_file_bytes,
        max_context_bytes=max_context_bytes,
        max_patch_bytes=max_patch_bytes,
        max_patch_files=max_patch_files,
        allow_sensitive_paths=allow_sensitive_paths,
    )
    return {"workspace": config.model_dump(mode="json")}


class RepositoryWorkspace:
    """A Git-backed, bounded coding workspace.

    The default worktree mode leaves the source checkout untouched. This class constrains paths,
    patch size, command shape, output size, and time, but it is not a hostile-code sandbox: tests
    execute local repository code with the current user's OS permissions.
    """

    def __init__(self, config: WorkspaceConfig, *, run_id: str) -> None:
        self.config = config
        self.run_id = run_id

    @classmethod
    def from_state_data(cls, data: dict[str, Any], *, run_id: str) -> "RepositoryWorkspace | None":
        payload = data.get("workspace")
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise WorkspaceError("state.data.workspace must be an object")
        try:
            config = WorkspaceConfig.model_validate(payload)
        except ValueError as exc:
            raise WorkspaceError(f"invalid workspace configuration: {exc}") from exc
        return cls(config, run_id=run_id)

    @property
    def source_root(self) -> Path:
        return Path(self.config.source_root).expanduser().resolve(strict=False)

    @property
    def active_root(self) -> Path:
        if not self.config.active_root:
            raise WorkspaceError("workspace has not been prepared")
        return Path(self.config.active_root).expanduser().resolve(strict=False)

    @property
    def artifact_root(self) -> Path:
        root = self.config.artifact_root or os.getenv(
            "GRAPH_MODEL_ARTIFACT_ROOT", str(Path.home() / ".graph-model" / "artifacts")
        )
        return (
            Path(root).expanduser().resolve(strict=False)
            / _safe_run_component(self.run_id)
        )

    @property
    def operations_dir(self) -> Path:
        return self.artifact_root / "operations"

    @property
    def patches_dir(self) -> Path:
        return self.artifact_root / "patches"

    @property
    def workspace_lock_path(self) -> Path:
        return self.artifact_root / "workspace.lock"

    def ensure_prepared(self) -> dict[str, Any]:
        source = _require_git_top_level(self.source_root)
        if source != self.source_root.resolve(strict=True):
            raise WorkspaceError(
                f"--repo must point to the Git top level; received {self.source_root}, top level is {source}"
            )

        artifact_base = Path(
            self.config.artifact_root
            or os.getenv(
                "GRAPH_MODEL_ARTIFACT_ROOT",
                str(Path.home() / ".graph-model" / "artifacts"),
            )
        ).expanduser().resolve(strict=False)
        _reject_path_inside_repository(artifact_base, source, "artifact root")

        if self.config.active_root:
            active = _require_git_top_level(self.active_root)
            if active != self.active_root.resolve(strict=True):
                raise WorkspaceError(f"active workspace root is not a Git top level: {self.active_root}")
            if _git_common_dir(active) != _git_common_dir(source):
                raise WorkspaceError("active workspace belongs to a different Git repository")
            current_head = _git_text(active, "rev-parse", "HEAD").strip()
            if self.config.base_commit and current_head != self.config.base_commit:
                raise WorkspaceError(
                    "workspace HEAD changed after the run started: "
                    f"expected {self.config.base_commit}, found {current_head}"
                )
            self._ensure_artifact_directories()
            return self.config.model_dump(mode="json")

        source_status = _git_text(
            source, "status", "--porcelain=v1", "--untracked-files=all"
        )
        if source_status.strip():
            raise WorkspaceError(
                "repository must be clean before a graph run; commit, stash, or use a clean clone. "
                "Dirty state would make verification and patch promotion ambiguous."
            )

        requested_commit = _resolve_commit(source, self.config.base_ref)
        if self.config.mode == WorkspaceMode.IN_PLACE:
            active = source
            base_commit = requested_commit
            current_head = _git_text(source, "rev-parse", "HEAD").strip()
            if base_commit != current_head:
                raise WorkspaceError(
                    "in-place mode requires base_ref to resolve to the checked-out HEAD"
                )
        else:
            home = self.config.workspace_home or os.getenv(
                "GRAPH_MODEL_WORKSPACE_HOME", str(Path.home() / ".graph-model" / "worktrees")
            )
            workspace_base = Path(home).expanduser().resolve(strict=False)
            _reject_path_inside_repository(workspace_base, source, "workspace home")
            repo_key = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
            active = workspace_base / repo_key / _safe_run_component(self.run_id)
            active.parent.mkdir(parents=True, exist_ok=True)
            if active.exists():
                existing = _require_git_top_level(active)
                if existing != active.resolve(strict=True):
                    raise WorkspaceError(f"existing worktree path is invalid: {active}")
                if _git_common_dir(active) != _git_common_dir(source):
                    raise WorkspaceError(
                        "existing deterministic worktree belongs to a different repository"
                    )
                active_head = _git_text(active, "rev-parse", "HEAD").strip()
                if active_head != requested_commit:
                    raise WorkspaceError(
                        "existing deterministic worktree is pinned to a different commit"
                    )
            else:
                _run_trusted(
                    [
                        "git",
                        "-C",
                        str(source),
                        "worktree",
                        "add",
                        "--detach",
                        str(active),
                        requested_commit,
                    ],
                    cwd=source,
                    timeout_seconds=120,
                )
            base_commit = _git_text(active, "rev-parse", "HEAD").strip()

        self.config = self.config.model_copy(
            update={
                "source_root": str(source),
                "active_root": str(active),
                "base_commit": base_commit,
            }
        )
        self._ensure_artifact_directories()
        return self.config.model_dump(mode="json")

    def _ensure_artifact_directories(self) -> None:
        self.operations_dir.mkdir(parents=True, exist_ok=True)
        self.patches_dir.mkdir(parents=True, exist_ok=True)

    def collect_context(self, task: str) -> dict[str, Any]:
        root = self.active_root
        files = self._repository_files()
        selected = self._select_context_files(files, task)
        status = _git_text(root, "status", "--porcelain=v1", "--untracked-files=all")
        branch = _git_text(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
        head = _git_text(root, "rev-parse", "HEAD").strip()
        commands = verification_commands(root, self.config.test_commands)
        profile = Counter(
            _LANGUAGE_BY_SUFFIX.get(Path(path).suffix.lower(), "Other") for path in files
        )
        context = {
            "mode": self.config.mode.value,
            "source_root": str(self.source_root),
            "active_root": str(root),
            "base_commit": self.config.base_commit,
            "head": head,
            "branch": branch,
            "clean_at_collection": not bool(status.strip()),
            "status": _relevant_status_lines(status),
            "file_count": len(files),
            "file_tree": files[:1_000],
            "file_tree_truncated": len(files) > 1_000,
            "language_profile": dict(profile.most_common()),
            "selected_files": selected,
            "test_commands": commands,
            "workspace_fingerprint": self.fingerprint(),
            "constraints": {
                "text_patches_only": True,
                "max_patch_bytes": self.config.max_patch_bytes,
                "max_patch_files": self.config.max_patch_files,
                "shell_disabled": True,
                "command_timeout_seconds": self.config.command_timeout_seconds,
            },
        }
        return context

    def current_evidence(self, *, max_diff_bytes: int | None = None) -> dict[str, Any]:
        root = self.active_root
        max_bytes = max_diff_bytes or min(self.config.max_context_bytes, 200_000)
        diff = _git_bytes(root, "diff", "--no-ext-diff", "--binary", "HEAD")
        diff_text, diff_truncated = _decode_truncated(diff, max_bytes)
        status = _git_text(root, "status", "--porcelain=v1", "--untracked-files=all")
        changed = self.changed_files()
        return {
            "workspace_fingerprint": self.fingerprint(),
            "status": _relevant_status_lines(status),
            "changed_files": changed,
            "diff": diff_text,
            "diff_truncated": diff_truncated,
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
            "diff_stat": _git_text(root, "diff", "--stat", "HEAD").strip(),
        }

    def changed_files(self) -> list[str]:
        output = _git_bytes(
            self.active_root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        paths: set[str] = set()
        items = output.split(b"\0")
        index = 0
        while index < len(items):
            item = items[index]
            index += 1
            if not item:
                continue
            text = item.decode("utf-8", errors="surrogateescape")
            if len(text) < 4:
                continue
            status = text[:2]
            path = text[3:]
            if "R" in status or "C" in status:
                if index < len(items) and items[index]:
                    path = items[index].decode("utf-8", errors="surrogateescape")
                    index += 1
            normalized = _normalize_relative_path(path)
            if not _is_excluded_path(normalized):
                paths.add(normalized)
        return sorted(paths)

    def fingerprint(self) -> str:
        root = self.active_root
        digest = hashlib.sha256()
        digest.update(_git_bytes(root, "rev-parse", "HEAD"))
        digest.update(_git_bytes(root, "diff", "--no-ext-diff", "--binary", "HEAD"))
        # Untracked files created by a proposed patch are marked intent-to-add and therefore appear
        # in the diff. Relevant untracked files that were not marked are hashed explicitly. Known
        # caches and build products are ignored so deterministic tests do not invalidate their own
        # evidence merely by creating __pycache__ or similar output.
        for path in self._untracked_files():
            digest.update(path.encode("utf-8", errors="surrogateescape"))
            file_path = self.safe_path(path)
            if file_path.is_file() and not file_path.is_symlink():
                digest.update(_hash_file(file_path).encode("ascii"))
        return digest.hexdigest()

    def apply_patch(
        self,
        patch: str,
        *,
        idempotency_key: str,
        no_changes_needed: bool = False,
    ) -> PatchApplication:
        self._ensure_artifact_directories()
        normalized = normalize_patch(patch)
        patch_bytes = normalized.encode("utf-8")
        if len(patch_bytes) > self.config.max_patch_bytes:
            raise PatchError(
                f"patch is {len(patch_bytes)} bytes; limit is {self.config.max_patch_bytes}"
            )
        if normalized:
            paths = validate_patch(
                normalized,
                root=self.active_root,
                max_files=self.config.max_patch_files,
                allow_sensitive_paths=self.config.allow_sensitive_paths,
            )
        else:
            if not no_changes_needed:
                raise PatchError("model returned an empty patch without no_changes_needed=true")
            paths = []

        patch_sha = hashlib.sha256(patch_bytes).hexdigest()
        operation_sha = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        intent_path = self.operations_dir / f"{operation_sha}.intent.json"
        commit_path = self.operations_dir / f"{operation_sha}.committed.json"
        operation_lock_path = self.operations_dir / f"{operation_sha}.lock"
        patch_path = self._write_patch_artifact(patch_sha[:16], normalized)

        with _exclusive_file_lock(self.workspace_lock_path):
            with _exclusive_file_lock(operation_lock_path):
                if commit_path.exists():
                    payload = _load_json_object(commit_path)
                    if payload.get("patch_sha256") != patch_sha:
                        raise PatchError(
                            "idempotency key was previously used for a different patch"
                        )
                    return _application_from_ledger(
                        payload, patch_path=patch_path, replayed=True
                    )

                prior_intent: dict[str, Any] | None = None
                if intent_path.exists():
                    prior_intent = _load_json_object(intent_path)
                    if prior_intent.get("patch_sha256") != patch_sha:
                        raise PatchError(
                            "idempotency intent was previously written for a different patch"
                        )
                    if prior_intent.get("active_root") != str(self.active_root):
                        raise PatchError(
                            "idempotency intent belongs to a different active workspace"
                        )

                if prior_intent is None:
                    before_fingerprint = self.fingerprint()
                    before_changed_files = self.changed_files()
                    intent = {
                        "format_version": 1,
                        "idempotency_key_sha256": operation_sha,
                        "patch_sha256": patch_sha,
                        "patch_artifact": str(patch_path),
                        "changed_files": list(paths),
                        "active_root": str(self.active_root),
                        "before_fingerprint": before_fingerprint,
                        "before_changed_files": before_changed_files,
                        "created_at": time.time(),
                    }
                    _atomic_write_json(intent_path, intent)
                else:
                    intent = prior_intent
                    before_fingerprint = str(intent["before_fingerprint"])
                    before_changed_files = [
                        str(value) for value in intent.get("before_changed_files", [])
                    ]

                if not normalized:
                    current_fingerprint = self.fingerprint()
                    if prior_intent is not None and current_fingerprint != before_fingerprint:
                        raise PatchError(
                            "no-op idempotency recovery found a changed workspace"
                        )
                    payload = {
                        **intent,
                        "applied": False,
                        "replayed": False,
                        "recovered_after_interruption": prior_intent is not None,
                        "after_fingerprint": current_fingerprint,
                        "diff_stat": _git_text(
                            self.active_root, "diff", "--stat", "HEAD"
                        ).strip(),
                        "committed_at": time.time(),
                    }
                    _atomic_write_json(commit_path, payload)
                    return _application_from_ledger(
                        payload, patch_path=patch_path, replayed=False
                    )

                patch_check = _run_trusted(
                    [
                        "git",
                        "apply",
                        "--check",
                        "--recount",
                        "--whitespace=nowarn",
                        str(patch_path),
                    ],
                    cwd=self.active_root,
                    timeout_seconds=60,
                    check=False,
                )
                recovered = False
                applied = False
                patch_present = False
                if patch_check.returncode == 0:
                    _run_trusted(
                        [
                            "git",
                            "apply",
                            "--recount",
                            "--whitespace=nowarn",
                            str(patch_path),
                        ],
                        cwd=self.active_root,
                        timeout_seconds=60,
                    )
                    applied = True
                    patch_present = True
                else:
                    reverse_check = _run_trusted(
                        [
                            "git",
                            "apply",
                            "--reverse",
                            "--check",
                            "--recount",
                            "--whitespace=nowarn",
                            str(patch_path),
                        ],
                        cwd=self.active_root,
                        timeout_seconds=60,
                        check=False,
                    )
                    if reverse_check.returncode == 0 and prior_intent is not None:
                        recovered = True
                        patch_present = True
                    elif reverse_check.returncode == 0:
                        raise PatchError(
                            "patch already appears in the workspace without a matching "
                            "idempotency intent"
                        )
                    else:
                        error = (patch_check.stderr or patch_check.stdout).strip()
                        raise PatchError(f"git apply --check rejected the patch: {error}")

                try:
                    self._mark_new_files_intent_to_add(paths)
                    after_changed_files = set(self.changed_files())
                    before_paths = set(before_changed_files)
                    unexpected = after_changed_files.difference(before_paths).difference(paths)
                    if unexpected:
                        raise PatchError(
                            f"patch changed undeclared paths: {sorted(unexpected)}"
                        )

                    after_fingerprint = self.fingerprint()
                    payload = {
                        **intent,
                        "applied": applied,
                        "replayed": False,
                        "recovered_after_interruption": recovered,
                        "after_fingerprint": after_fingerprint,
                        "diff_stat": _git_text(
                            self.active_root, "diff", "--stat", "HEAD"
                        ).strip(),
                        "committed_at": time.time(),
                    }
                    _atomic_write_json(commit_path, payload)
                except BaseException as exc:
                    if patch_present:
                        self._rollback_patch(
                            patch_path,
                            declared_paths=paths,
                            before_changed_files=before_changed_files,
                            expected_fingerprint=before_fingerprint,
                        )
                    raise exc

                return _application_from_ledger(
                    payload, patch_path=patch_path, replayed=False
                )

    def _rollback_patch(
        self,
        patch_path: Path,
        *,
        declared_paths: Sequence[str],
        before_changed_files: Sequence[str],
        expected_fingerprint: str,
    ) -> None:
        reverse_check = _run_trusted(
            [
                "git",
                "apply",
                "--reverse",
                "--check",
                "--recount",
                "--whitespace=nowarn",
                str(patch_path),
            ],
            cwd=self.active_root,
            timeout_seconds=60,
            check=False,
        )
        if reverse_check.returncode != 0:
            detail = (reverse_check.stderr or reverse_check.stdout).strip()
            raise WorkspaceError(
                "patch validation failed after mutation and rollback could not be checked: "
                f"{detail}"
            )
        _run_trusted(
            [
                "git",
                "apply",
                "--reverse",
                "--recount",
                "--whitespace=nowarn",
                str(patch_path),
            ],
            cwd=self.active_root,
            timeout_seconds=60,
        )
        newly_declared = sorted(set(declared_paths).difference(before_changed_files))
        if newly_declared:
            _run_trusted(
                ["git", "reset", "-q", "HEAD", "--", *newly_declared],
                cwd=self.active_root,
                timeout_seconds=60,
                check=False,
            )
        actual_fingerprint = self.fingerprint()
        if actual_fingerprint != expected_fingerprint:
            raise WorkspaceError(
                "patch rollback did not restore the pre-application workspace fingerprint"
            )

    def run_tests(self) -> dict[str, Any]:
        with _exclusive_file_lock(self.workspace_lock_path):
            commands = verification_commands(self.active_root, self.config.test_commands)
            before_fingerprint = self.fingerprint()
            results: list[CommandResult] = []
            for command in commands:
                result = run_bounded_command(
                    command,
                    cwd=self.active_root,
                    allowed_commands=self.config.allowed_commands,
                    timeout_seconds=self.config.command_timeout_seconds,
                    max_output_bytes=self.config.max_command_output_bytes,
                    extra_env={
                        "GRAPH_MODEL_RUN_ID": self.run_id,
                        "GRAPH_MODEL_WORKSPACE": str(self.active_root),
                    },
                )
                results.append(result)
                if not result.passed:
                    break
            commands_passed = bool(results) and all(result.passed for result in results)
            after_fingerprint = self.fingerprint()
            workspace_mutated = after_fingerprint != before_fingerprint
            passed = commands_passed and not workspace_mutated
            return {
                "verdict": "pass" if passed else "fail",
                "commands": [result.as_dict() for result in results],
                "configured_commands": commands,
                "workspace_fingerprint_before": before_fingerprint,
                "workspace_fingerprint": after_fingerprint,
                "workspace_mutated": workspace_mutated,
                "changed_files": self.changed_files(),
                "diff_stat": _git_text(
                    self.active_root, "diff", "--stat", "HEAD"
                ).strip(),
            }

    def export_patch(self, name: str = "verified.patch") -> dict[str, Any]:
        self._ensure_artifact_directories()
        patch = _git_bytes(self.active_root, "diff", "--no-ext-diff", "--binary", "HEAD")
        path = self.artifact_root / name
        _atomic_write_bytes(path, patch)
        return {
            "path": str(path),
            "sha256": hashlib.sha256(patch).hexdigest(),
            "bytes": len(patch),
            "changed_files": self.changed_files(),
            "base_commit": self.config.base_commit,
        }

    def promote_verified_patch(self, patch_path: str | Path, patch_sha256: str) -> dict[str, Any]:
        if self.config.mode == WorkspaceMode.IN_PLACE:
            return {
                "status": "already-in-place",
                "source_root": str(self.source_root),
                "active_root": str(self.active_root),
            }
        source = _require_git_top_level(self.source_root)
        current = _git_text(source, "rev-parse", "HEAD").strip()
        if current != self.config.base_commit:
            raise WorkspaceError(
                "source HEAD changed since the run began; refusing to promote a stale patch"
            )
        patch_file = Path(patch_path).expanduser().resolve(strict=True)
        actual_sha = _hash_file(patch_file)
        if actual_sha != patch_sha256:
            raise WorkspaceError(
                f"verified patch hash mismatch: expected {patch_sha256}, found {actual_sha}"
            )
        patch_text = patch_file.read_text(encoding="utf-8")
        promotion_config = self.config.model_copy(
            update={"mode": WorkspaceMode.IN_PLACE, "active_root": str(source)}
        )
        target = RepositoryWorkspace(promotion_config, run_id=self.run_id)
        promotion_key = f"promote:{self.run_id}:{patch_sha256}"
        operation_sha = hashlib.sha256(promotion_key.encode("utf-8")).hexdigest()
        committed = target.operations_dir / f"{operation_sha}.committed.json"
        status = _git_text(source, "status", "--porcelain=v1", "--untracked-files=all")
        if status.strip() and not committed.exists():
            raise WorkspaceError("source repository must be clean before patch promotion")
        application = target.apply_patch(
            patch_text,
            idempotency_key=promotion_key,
        )
        return {"status": "applied", **application.as_dict(), "source_root": str(source)}

    def cleanup_worktree(self, *, force: bool = False) -> dict[str, Any]:
        """Remove the detached run worktree while retaining audit and patch artifacts."""

        if self.config.mode == WorkspaceMode.IN_PLACE:
            return {
                "status": "not-applicable",
                "mode": self.config.mode.value,
                "source_root": str(self.source_root),
                "active_root": str(self.active_root),
            }
        if not self.config.active_root:
            return {"status": "not-prepared", "mode": self.config.mode.value}

        source = _require_git_top_level(self.source_root)
        active = Path(self.config.active_root).expanduser().resolve(strict=False)
        with _exclusive_file_lock(self.workspace_lock_path):
            if not active.exists():
                _run_trusted(
                    ["git", "-C", str(source), "worktree", "prune"],
                    cwd=source,
                    timeout_seconds=60,
                    check=False,
                )
                return {
                    "status": "already-removed",
                    "mode": self.config.mode.value,
                    "source_root": str(source),
                    "active_root": str(active),
                    "artifacts_retained": str(self.artifact_root),
                }
            active_top = _require_git_top_level(active)
            if active_top != active or _git_common_dir(active) != _git_common_dir(source):
                raise WorkspaceError("refusing to remove an unrelated or malformed worktree")
            status = _git_text(
                active, "status", "--porcelain=v1", "--untracked-files=all"
            )
            if status.strip() and not force:
                raise WorkspaceError(
                    "run worktree has uncommitted changes; pass force only after its patch "
                    "artifact has been retained"
                )
            command = ["git", "-C", str(source), "worktree", "remove"]
            if force:
                command.append("--force")
            command.append(str(active))
            _run_trusted(command, cwd=source, timeout_seconds=120)
            _run_trusted(
                ["git", "-C", str(source), "worktree", "prune"],
                cwd=source,
                timeout_seconds=60,
                check=False,
            )
            return {
                "status": "removed",
                "mode": self.config.mode.value,
                "source_root": str(source),
                "active_root": str(active),
                "dirty_before_removal": bool(status.strip()),
                "artifacts_retained": str(self.artifact_root),
            }

    def read_changed_file_context(self, *, max_bytes: int = 100_000) -> list[dict[str, Any]]:
        remaining = max_bytes
        output: list[dict[str, Any]] = []
        for path in self.changed_files():
            if remaining <= 0:
                break
            file_path = self.safe_path(path)
            if not file_path.exists() or not file_path.is_file() or file_path.is_symlink():
                continue
            data = file_path.read_bytes()
            if not _looks_text(data):
                continue
            text = data.decode("utf-8", errors="replace")
            excerpt = text[: min(len(text), remaining, self.config.max_context_file_bytes)]
            remaining -= len(excerpt.encode("utf-8"))
            output.append(
                {
                    "path": path,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "content": excerpt,
                    "truncated": len(excerpt) < len(text),
                }
            )
        return output

    def safe_path(self, relative_path: str) -> Path:
        normalized = _normalize_relative_path(relative_path)
        candidate = (self.active_root / normalized).resolve(strict=False)
        try:
            candidate.relative_to(self.active_root)
        except ValueError as exc:
            raise WorkspaceError(f"path escapes workspace: {relative_path!r}") from exc
        return candidate

    def _repository_files(self) -> list[str]:
        output = _git_bytes(
            self.active_root,
            "ls-files",
            "-co",
            "--exclude-standard",
            "-z",
        )
        files: list[str] = []
        for raw in output.split(b"\0"):
            if not raw:
                continue
            path = _normalize_relative_path(raw.decode("utf-8", errors="surrogateescape"))
            if _is_excluded_path(path):
                continue
            file_path = self.safe_path(path)
            if not file_path.is_file() or file_path.is_symlink():
                continue
            files.append(path)
        return sorted(set(files))

    def _select_context_files(self, files: Sequence[str], task: str) -> list[dict[str, Any]]:
        task_lower = task.lower()
        task_terms = {
            term
            for term in re.findall(r"[a-zA-Z_][a-zA-Z0-9_.-]{2,}", task_lower)
            if term not in {"the", "and", "for", "with", "that", "this", "from", "into"}
        }
        scored: list[tuple[float, str, bytes]] = []
        scan_budget = max(self.config.max_context_bytes * 20, 2_000_000)
        scanned = 0
        for path in files:
            file_path = self.safe_path(path)
            try:
                size = file_path.stat().st_size
            except OSError:
                continue
            if size > max(self.config.max_context_file_bytes * 20, 1_000_000):
                continue
            suffix = file_path.suffix.lower()
            if suffix in _BINARY_SUFFIXES:
                continue
            if scanned + size > scan_budget and scored:
                continue
            try:
                data = file_path.read_bytes()
            except OSError:
                continue
            scanned += len(data)
            if not _looks_text(data):
                continue
            path_lower = path.lower()
            path_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_-]{1,}", path_lower))
            score = float(len(task_terms.intersection(path_tokens)) * 12)
            if path_lower in task_lower or any(
                f"/{path_lower}" in task_lower for _ in (0,)
            ):
                score += 100
            if file_path.name.lower() in _PRIORITY_FILENAMES:
                score += 6
            if "test" in task_lower and ("test" in path_lower or "spec" in path_lower):
                score += 8
            text_lower = data[:250_000].decode("utf-8", errors="ignore").lower()
            content_matches = sum(min(3, text_lower.count(term)) for term in task_terms)
            score += min(30, content_matches)
            if score > 0 or len(scored) < self.config.max_context_files:
                scored.append((score, path, data))

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected: list[dict[str, Any]] = []
        remaining = self.config.max_context_bytes
        for score, path, data in scored:
            if len(selected) >= self.config.max_context_files or remaining <= 0:
                break
            text = data.decode("utf-8", errors="replace")
            budget = min(self.config.max_context_file_bytes, remaining)
            excerpt = _relevant_excerpt(text, task_terms, budget)
            encoded_length = len(excerpt.encode("utf-8"))
            if encoded_length == 0:
                continue
            remaining -= encoded_length
            selected.append(
                {
                    "path": path,
                    "score": round(score, 3),
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "content": excerpt,
                    "truncated": encoded_length < len(data),
                }
            )
        return selected

    def _untracked_files(self) -> list[str]:
        output = _git_bytes(
            self.active_root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        )
        values = [
            _normalize_relative_path(value.decode("utf-8", errors="surrogateescape"))
            for value in output.split(b"\0")
            if value
        ]
        return sorted(value for value in values if not _is_excluded_path(value))

    def _mark_new_files_intent_to_add(self, paths: Sequence[str]) -> None:
        new_paths = [path for path in paths if path in set(self._untracked_files())]
        if not new_paths:
            return
        _run_trusted(
            ["git", "add", "-N", "--", *new_paths],
            cwd=self.active_root,
            timeout_seconds=60,
        )

    def _write_patch_artifact(self, stem: str, patch: str) -> Path:
        self.patches_dir.mkdir(parents=True, exist_ok=True)
        path = self.patches_dir / f"{stem}.patch"
        if path.exists():
            current = path.read_text(encoding="utf-8")
            if current != patch:
                raise WorkspaceError(f"patch artifact collision at {path}")
            return path
        _atomic_write_bytes(path, patch.encode("utf-8"))
        return path

@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    key = str(path.resolve(strict=False))
    with _FILE_LOCKS_GUARD:
        thread_lock = _FILE_LOCKS.setdefault(key, threading.RLock())
    with thread_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _application_from_ledger(
    payload: dict[str, Any], *, patch_path: Path, replayed: bool
) -> PatchApplication:
    return PatchApplication(
        patch_sha256=str(payload["patch_sha256"]),
        patch_artifact=str(patch_path),
        changed_files=tuple(str(value) for value in payload.get("changed_files", [])),
        applied=bool(payload.get("applied", False)),
        replayed=replayed,
        recovered_after_interruption=bool(
            payload.get("recovered_after_interruption", False)
        ),
        before_fingerprint=str(payload["before_fingerprint"]),
        after_fingerprint=str(payload["after_fingerprint"]),
        diff_stat=str(payload.get("diff_stat", "")),
    )



def _is_excluded_path(path: str) -> bool:
    pure = PurePosixPath(path)
    if any(part in _EXCLUDED_DIRECTORY_NAMES for part in pure.parts[:-1]):
        return True
    return pure.suffix.lower() in {".pyc", ".pyo"}


def _relevant_status_lines(status: str) -> list[str]:
    lines: list[str] = []
    for line in status.splitlines():
        path = line[3:] if len(line) >= 4 else ""
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        try:
            normalized = _normalize_relative_path(path)
        except WorkspaceError:
            lines.append(line)
            continue
        if not _is_excluded_path(normalized):
            lines.append(line)
    return lines



def normalize_patch(patch: str) -> str:
    text = patch.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```diff", "```patch"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text:
        return ""
    return text + "\n"


def validate_patch(
    patch: str,
    *,
    root: str | Path,
    max_files: int,
    allow_sensitive_paths: bool = False,
) -> list[str]:
    if "GIT binary patch" in patch or re.search(
        r"^Binary files ", patch, flags=re.MULTILINE
    ):
        raise PatchError("binary patches are not supported")
    if re.search(r"^(?:old|new) mode 120000$", patch, flags=re.MULTILINE):
        raise PatchError("symlink patches are not supported")
    if re.search(r"^new file mode 160000$", patch, flags=re.MULTILINE):
        raise PatchError("submodule patches are not supported")
    if re.search(r"^(?:rename|copy) (?:from|to) ", patch, flags=re.MULTILINE):
        raise PatchError("rename and copy patches are not supported in this release")

    paths: set[str] = set()
    section: dict[str, Any] | None = None

    def finish_section() -> None:
        if section is None:
            return
        old_header = section.get("old_header")
        new_header = section.get("new_header")
        if (old_header is None) != (new_header is None):
            raise PatchError(
                f"diff section for {section['old']!r} must contain both --- and +++ headers"
            )
        if old_header is None:
            return
        if old_header == "/dev/null" and new_header == "/dev/null":
            raise PatchError("a diff section cannot use /dev/null for both paths")
        if old_header not in {section["old"], "/dev/null"}:
            raise PatchError(
                "--- path does not match its diff --git header: "
                f"{old_header!r} != {section['old']!r}"
            )
        if new_header not in {section["new"], "/dev/null"}:
            raise PatchError(
                "+++ path does not match its diff --git header: "
                f"{new_header!r} != {section['new']!r}"
            )
        if old_header == "/dev/null" and new_header != section["new"]:
            raise PatchError("new-file patch has an invalid +++ path")
        if new_header == "/dev/null" and old_header != section["old"]:
            raise PatchError("deleted-file patch has an invalid --- path")

    for line in patch.splitlines():
        if line.startswith("diff --git "):
            finish_section()
            try:
                parts = shlex.split(line)
            except ValueError as exc:
                raise PatchError(f"invalid diff header: {line!r}") from exc
            if len(parts) != 4:
                raise PatchError(f"invalid diff header: {line!r}")
            old_path = _validate_patch_path(
                _strip_git_prefix(parts[2]), root, allow_sensitive_paths
            )
            new_path = _validate_patch_path(
                _strip_git_prefix(parts[3]), root, allow_sensitive_paths
            )
            if old_path != new_path:
                raise PatchError(
                    "diff --git paths differ; rename and copy patches are not supported: "
                    f"{old_path!r} -> {new_path!r}"
                )
            paths.update({old_path, new_path})
            section = {
                "old": old_path,
                "new": new_path,
                "old_header": None,
                "new_header": None,
                "in_hunks": False,
            }
            continue

        if section is None:
            continue
        if line.startswith("@@"):
            section["in_hunks"] = True
            continue
        if section["in_hunks"]:
            continue
        if line.startswith("--- "):
            if section["old_header"] is not None:
                raise PatchError("diff section contains duplicate --- headers")
            raw = _parse_unified_header_path(line[4:])
            if raw == "/dev/null":
                section["old_header"] = raw
            else:
                section["old_header"] = _validate_patch_path(
                    _strip_git_prefix(raw), root, allow_sensitive_paths
                )
        elif line.startswith("+++ "):
            if section["new_header"] is not None:
                raise PatchError("diff section contains duplicate +++ headers")
            raw = _parse_unified_header_path(line[4:])
            if raw == "/dev/null":
                section["new_header"] = raw
            else:
                section["new_header"] = _validate_patch_path(
                    _strip_git_prefix(raw), root, allow_sensitive_paths
                )

    finish_section()
    if not paths:
        raise PatchError("patch must contain at least one 'diff --git' header")
    if len(paths) > max_files:
        raise PatchError(f"patch changes {len(paths)} files; limit is {max_files}")
    return sorted(paths)


def _parse_unified_header_path(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise PatchError("unified diff path header cannot be empty")
    if raw.startswith('"'):
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            raise PatchError(f"invalid quoted unified diff path: {raw!r}") from exc
        if len(parts) != 1:
            raise PatchError(f"invalid quoted unified diff path: {raw!r}")
        return parts[0]
    return raw.split("\t", 1)[0]


def verification_commands(
    root: str | Path, configured_commands: Sequence[str] = ()
) -> list[str]:
    selected = list(configured_commands) if configured_commands else detect_test_commands(root)
    return list(dict.fromkeys(["git diff --check", *selected]))


def detect_test_commands(root: str | Path) -> list[str]:
    root_path = Path(root)
    commands: list[str] = []
    if (root_path / "uv.lock").exists() and (
        (root_path / "tests").exists() or (root_path / "pyproject.toml").exists()
    ):
        commands.append("uv run pytest -q")
    elif (root_path / "poetry.lock").exists():
        commands.append("poetry run pytest -q")
    elif (root_path / "tox.ini").exists():
        commands.append("tox -q")
    elif (root_path / "pyproject.toml").exists() and (root_path / "tests").exists():
        commands.append(f"{shlex.quote(sys.executable)} -m pytest -q")
    elif (root_path / "pytest.ini").exists() or (root_path / "conftest.py").exists():
        commands.append(f"{shlex.quote(sys.executable)} -m pytest -q")

    package_json = root_path / "package.json"
    if package_json.exists() and _package_has_test_script(package_json):
        if (root_path / "pnpm-lock.yaml").exists():
            commands.append("pnpm test")
        elif (root_path / "yarn.lock").exists():
            commands.append("yarn test")
        elif (root_path / "bun.lock").exists() or (root_path / "bun.lockb").exists():
            commands.append("bun run test")
        else:
            commands.append("npm test")
    if (root_path / "Cargo.toml").exists():
        commands.append("cargo test")
    if (root_path / "go.mod").exists():
        commands.append("go test ./...")
    if (root_path / "Package.swift").exists():
        commands.append("swift test")
    if (root_path / "CMakeLists.txt").exists() and (root_path / "build").is_dir():
        commands.append("ctest --test-dir build --output-on-failure")
    return list(dict.fromkeys(commands))


def _package_has_test_script(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    scripts = payload.get("scripts")
    if not isinstance(scripts, dict):
        return False
    command = scripts.get("test")
    if not isinstance(command, str) or not command.strip():
        return False
    normalized = command.lower().replace(" ", "")
    return "error:notestspecified" not in normalized


def parse_bounded_command(
    command: str,
    *,
    cwd: str | Path,
    allowed_commands: Sequence[str],
) -> tuple[str, ...]:
    if "\n" in command or "\r" in command or "\x00" in command:
        raise CommandPolicyError("commands cannot contain newlines or NUL bytes")
    try:
        argv = shlex.split(command, posix=True)
    except ValueError as exc:
        raise CommandPolicyError(f"invalid command quoting: {exc}") from exc
    if not argv:
        raise CommandPolicyError("command cannot be empty")
    if any(token in _BLOCKED_CONTROL_TOKENS for token in argv):
        raise CommandPolicyError("shell control operators are not allowed")
    if any("$(" in token or "`" in token for token in argv):
        raise CommandPolicyError("shell substitutions are not allowed")

    root = Path(cwd).resolve(strict=True)
    executable = argv[0]
    executable_path = Path(executable).expanduser()
    basename = executable_path.name
    allowed_names = {
        Path(value).name for value in allowed_commands if value and "/" not in value
    }
    allowed_paths = {
        Path(value).expanduser().resolve(strict=False)
        for value in allowed_commands
        if "/" in value
    }
    python_allowed = basename.startswith("python") and any(
        value in allowed_names for value in {"python", "python3", basename}
    )
    name_allowed = basename in allowed_names or python_allowed

    search_path = _sanitized_search_path(root)
    system_match_raw = shutil.which(basename, path=search_path)
    system_match = (
        Path(system_match_raw).resolve(strict=True) if system_match_raw is not None else None
    )
    current_python = Path(sys.executable).resolve(strict=True)

    if executable.startswith("./") or "/" in executable:
        if executable_path.is_absolute():
            resolved = executable_path.resolve(strict=True)
            trusted = resolved in allowed_paths or (
                name_allowed
                and (
                    resolved == system_match
                    or (python_allowed and resolved == current_python)
                )
            )
            if not trusted:
                raise CommandPolicyError(
                    f"absolute executable is not a trusted allowlisted path: {resolved}"
                )
        else:
            unresolved = root / executable_path
            resolved = unresolved.resolve(strict=True)
            _ensure_under_root(resolved, root)
            if not name_allowed and resolved not in allowed_paths:
                raise CommandPolicyError(
                    f"repository-local executable {basename!r} is not in the allowlist"
                )
    else:
        if not name_allowed:
            raise CommandPolicyError(
                f"executable {basename!r} is not in the workspace allowlist"
            )
        if system_match is None:
            raise CommandPolicyError(
                f"allowlisted executable {basename!r} was not found on the sanitized PATH"
            )
        resolved = system_match

    if not resolved.is_file():
        raise CommandPolicyError(f"executable is not a file: {resolved}")
    argv[0] = str(resolved)

    if basename == "git":
        subcommand = argv[1] if len(argv) > 1 else None
        if subcommand not in _GIT_READ_ONLY_SUBCOMMANDS:
            raise CommandPolicyError(
                f"git subcommand {subcommand!r} is not allowed in verifier commands"
            )
    return tuple(argv)


def run_bounded_command(
    command: str,
    *,
    cwd: str | Path,
    allowed_commands: Sequence[str] = DEFAULT_ALLOWED_COMMANDS,
    timeout_seconds: float = 300.0,
    max_output_bytes: int = 200_000,
    extra_env: dict[str, str] | None = None,
) -> CommandResult:
    root = Path(cwd).resolve(strict=True)
    argv = parse_bounded_command(command, cwd=root, allowed_commands=allowed_commands)
    environment = _minimal_environment(extra_env, cwd=root)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="graph-model-command-") as temp_dir:
        stdout_path = Path(temp_dir) / "stdout.log"
        stderr_path = Path(temp_dir) / "stderr.log"
        timed_out = False
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                list(argv),
                cwd=root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
            try:
                exit_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                exit_code = None
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        stdout, stdout_truncated = _read_log(stdout_path, max_output_bytes)
        stderr, stderr_truncated = _read_log(stderr_path, max_output_bytes)
    return CommandResult(
        command=command,
        argv=argv,
        exit_code=exit_code,
        duration_seconds=time.monotonic() - started,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )


def _safe_run_component(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", run_id)
    slug = re.sub(r"\.{2,}", "-", slug).strip(".-_")[:48]
    return f"{slug or 'run'}-{digest}"


def _reject_path_inside_repository(path: Path, repository: Path, label: str) -> None:
    try:
        path.relative_to(repository)
    except ValueError:
        return
    raise WorkspaceError(f"{label} must be outside the source repository: {path}")


def _resolve_commit(root: Path, revision: str) -> str:
    result = _run_trusted(
        ["git", "-C", str(root), "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=root,
        timeout_seconds=30,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise WorkspaceError(f"base_ref does not resolve to a commit: {revision!r}: {detail}")
    return result.stdout.strip()


def _git_common_dir(root: Path) -> Path:
    raw = _git_text(root, "rev-parse", "--git-common-dir").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=True)


def _require_git_top_level(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise WorkspaceError(f"repository path does not exist or is not a directory: {path}")
    result = _run_trusted(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        cwd=path,
        timeout_seconds=30,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise WorkspaceError(f"path is not inside a Git worktree: {path}: {detail}")
    return Path(result.stdout.strip()).resolve(strict=True)


def _git_text(root: Path, *arguments: str) -> str:
    return _run_trusted(
        ["git", "-C", str(root), *arguments], cwd=root, timeout_seconds=60
    ).stdout


def _git_bytes(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise WorkspaceError(
            f"git {' '.join(arguments)} failed: {result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def _run_trusted(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    timeout_seconds: float,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(argv),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise WorkspaceError(f"command failed ({result.returncode}): {shlex.join(argv)}: {detail}")
    return result


def _normalize_relative_path(path: str) -> str:
    value = path.replace("\\", "/")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise WorkspaceError(f"invalid relative path: {path!r}")
    return pure.as_posix()


def _strip_git_prefix(path: str) -> str:
    if path == "/dev/null":
        return path
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    raise PatchError(f"diff paths must use a/ and b/ prefixes: {path!r}")


def _validate_patch_path(path: str, root: str | Path, allow_sensitive_paths: bool) -> str:
    if path == "/dev/null":
        raise PatchError("diff --git headers cannot use /dev/null")
    try:
        normalized = _normalize_relative_path(path)
    except WorkspaceError as exc:
        raise PatchError(str(exc)) from exc
    parts = PurePosixPath(normalized).parts
    if ".git" in parts or ".graph-model" in parts:
        raise PatchError(f"patch cannot modify internal control paths: {normalized}")
    name = parts[-1].lower()
    if not allow_sensitive_paths:
        sensitive_env = name.startswith(".env.") and name not in {".env.example", ".env.sample"}
        if (
            name in _SENSITIVE_EXACT_NAMES
            or sensitive_env
            or PurePosixPath(normalized).suffix.lower() in _SENSITIVE_SUFFIXES
            or ".ssh" in parts
        ):
            raise PatchError(f"patch targets a sensitive path: {normalized}")
    root_path = Path(root).resolve(strict=True)
    candidate = (root_path / normalized).resolve(strict=False)
    _ensure_under_root(candidate, root_path)
    existing = root_path
    for part in parts:
        existing = existing / part
        if existing.is_symlink():
            raise PatchError(f"patch cannot traverse or modify symlinks: {normalized}")
    return normalized


def _ensure_under_root(candidate: Path, root: Path) -> None:
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise WorkspaceError(f"path escapes workspace root: {candidate}") from exc


def _looks_text(data: bytes) -> bool:
    if b"\x00" in data[:8_192]:
        return False
    if not data:
        return True
    sample = data[:8_192]
    control = sum(byte < 9 or 13 < byte < 32 for byte in sample)
    return control / len(sample) < 0.02


def _relevant_excerpt(text: str, terms: Iterable[str], max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    lines = text.splitlines()
    lower_lines = [line.lower() for line in lines]
    matches = [
        index
        for index, line in enumerate(lower_lines)
        if any(term in line for term in terms)
    ]
    ranges: list[tuple[int, int]] = []
    for index in matches[:20]:
        start = max(0, index - 12)
        end = min(len(lines), index + 13)
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    if not ranges:
        ranges = [(0, min(len(lines), 120)), (max(0, len(lines) - 40), len(lines))]
    pieces: list[str] = []
    used = 0
    for start, end in ranges:
        header = f"\n--- lines {start + 1}-{end} ---\n"
        body = "\n".join(f"{line_no + 1}: {lines[line_no]}" for line_no in range(start, end))
        piece = header + body
        piece_bytes = piece.encode("utf-8")
        if used + len(piece_bytes) > max_bytes:
            remaining = max_bytes - used
            if remaining > 0:
                pieces.append(piece_bytes[:remaining].decode("utf-8", errors="ignore"))
            break
        pieces.append(piece)
        used += len(piece_bytes)
    return "".join(pieces).lstrip()


def _sanitized_search_path(root: Path) -> str:
    entries: list[str] = []
    for raw in os.environ.get("PATH", os.defpath).split(os.pathsep):
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError:
            entries.append(str(candidate))
    if not entries:
        entries = os.defpath.split(os.pathsep)
    return os.pathsep.join(dict.fromkeys(entries))


def _minimal_environment(
    extra_env: dict[str, str] | None, *, cwd: Path
) -> dict[str, str]:
    allowed_names = {
        "CARGO_HOME",
        "CONDA_PREFIX",
        "DEVELOPER_DIR",
        "GOPATH",
        "GOROOT",
        "HOME",
        "JAVA_HOME",
        "LANG",
        "LC_ALL",
        "NPM_CONFIG_CACHE",
        "PATH",
        "RUSTUP_HOME",
        "SDKROOT",
        "TMP",
        "TEMP",
        "TMPDIR",
        "USER",
        "VIRTUAL_ENV",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed_names}
    environment["PATH"] = _sanitized_search_path(cwd)
    environment.setdefault("LANG", "C.UTF-8")
    environment["CI"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        for key, value in extra_env.items():
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
                raise CommandPolicyError(f"invalid environment variable name: {key!r}")
            environment[key] = value
    return environment


def _read_log(path: Path, max_bytes: int) -> tuple[str, bool]:
    data = path.read_bytes()
    truncated = len(data) > max_bytes
    if truncated:
        head_size = max_bytes // 2
        tail_size = max_bytes - head_size
        data = data[:head_size] + b"\n... output truncated ...\n" + data[-tail_size:]
    return data.decode("utf-8", errors="replace"), truncated


def _decode_truncated(data: bytes, max_bytes: int) -> tuple[str, bool]:
    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace"), truncated


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8"),
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkspaceError(f"invalid operation ledger {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WorkspaceError(f"operation ledger must contain an object: {path}")
    return payload
