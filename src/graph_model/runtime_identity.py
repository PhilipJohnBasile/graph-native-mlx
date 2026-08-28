"""Finite, content-addressed identities for provenance-sensitive runtimes.

The candidate controller uses these helpers before importing or executing code.
They deliberately define a finite trust boundary: immutable/root-owned Apple OS
paths are trusted as part of the host TCB; mutable non-system runtime content is
enumerated and hashed without invoking a shell or an ambient resolver.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import subprocess
import sys
import sysconfig
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


GIT_RUNTIME_IDENTITY_FORMAT = "graph-native-git-runtime-identity-v1"
GO_RUNTIME_IDENTITY_FORMAT = "graph-native-go-runtime-identity-v1"
PYTHON_RUNTIME_IDENTITY_FORMAT = "graph-native-python-runtime-identity-v1"
MACHO_RUNTIME_CLOSURE_FORMAT = "graph-native-macho-runtime-closure-v1"
TREE_MANIFEST_FORMAT = "graph-native-runtime-tree-manifest-v1"

CLT_GIT_EXECUTABLE = Path("/Library/Developer/CommandLineTools/usr/bin/git")
CLT_GIT_EXEC_PATH = Path("/Library/Developer/CommandLineTools/usr/libexec/git-core")
TRUSTED_OS_DEPENDENCY_PREFIXES = (Path("/usr/lib"), Path("/System/Library"))
# Runtime identities may need to authenticate MLX native payloads.  Keep that
# boundary finite while admitting the observed 162,449,848-byte metallib.
_MAX_RUNTIME_FILE_BYTES = 256 * 1024 * 1024
# A native graph has finite inputs, but keep context-sensitive @rpath traversal
# explicitly bounded as well.  The cap is far above ordinary wheel graphs.
_MAX_MACHO_RESOLUTION_CONTEXTS = 131_072

_GIT_REQUIRED_BUILTINS = frozenset(
    {
        "add",
        "apply",
        "archive",
        "cat-file",
        "clean",
        "config",
        "diff",
        "for-each-ref",
        "ls-files",
        "ls-tree",
        "reset",
        "rev-parse",
        "show",
        "status",
        "worktree",
    }
)
_GO_ENV_KEYS = (
    "CC",
    "CGO_ENABLED",
    "CXX",
    "GOARCH",
    "GOOS",
    "GOROOT",
    "GOTOOLDIR",
)


class RuntimeIdentityError(RuntimeError):
    """A runtime falls outside the finite authenticated trust boundary."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _stable_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _hash_file_descriptor(descriptor: int) -> str:
    """Hash one exact regular file through a finite descriptor-bound read."""

    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeIdentityError("runtime-file-not-regular")
    if before.st_size > _MAX_RUNTIME_FILE_BYTES:
        raise RuntimeIdentityError("runtime-file-exceeds-reader-bound")
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    bytes_read = 0
    while True:
        chunk = os.read(
            descriptor,
            min(1024 * 1024, before.st_size - bytes_read + 1),
        )
        if not chunk:
            break
        bytes_read += len(chunk)
        # The descriptor can grow after the fstat above.  Stop on the first
        # byte outside that opened snapshot rather than continuing a hostile
        # stream until the global reader bound.
        if bytes_read > before.st_size:
            raise RuntimeIdentityError("runtime-file-replaced")
        if bytes_read > _MAX_RUNTIME_FILE_BYTES:
            raise RuntimeIdentityError("runtime-file-exceeds-reader-bound")
        digest.update(chunk)
    if bytes_read != before.st_size:
        raise RuntimeIdentityError("runtime-file-replaced")
    return digest.hexdigest()


def _read_stable_regular_file(
    path: Path, *, retained_bytes: int | None = None
) -> tuple[bytes, dict[str, Any]]:
    """Hash one stable runtime file, retaining at most the requested prefix.

    The default retains the full file for existing byte-consuming callers.
    Identity-only paths pass ``0``; Mach-O inspection first retains four bytes,
    so large non-Mach-O payloads are streamed and never materialized in RAM.
    """

    if retained_bytes is not None and (
        isinstance(retained_bytes, bool)
        or not isinstance(retained_bytes, int)
        or retained_bytes < 0
    ):
        raise RuntimeIdentityError("runtime-file-retention-limit-invalid")
    target = path.absolute()
    try:
        before = target.lstat()
        if stat.S_ISLNK(before.st_mode):
            raise RuntimeIdentityError(f"runtime-file-unavailable:{target}")
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeIdentityError(f"runtime-file-not-regular:{target}")
        if before.st_size > _MAX_RUNTIME_FILE_BYTES:
            raise RuntimeIdentityError("runtime-file-exceeds-reader-bound")
        descriptor = os.open(
            target,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise RuntimeIdentityError(f"runtime-file-unavailable:{target}") from exc
    if not stat.S_ISREG(before.st_mode):
        os.close(descriptor)
        raise RuntimeIdentityError(f"runtime-file-not-regular:{target}")
    try:
        opened = os.fstat(descriptor)
        if opened.st_size > _MAX_RUNTIME_FILE_BYTES:
            raise RuntimeIdentityError("runtime-file-exceeds-reader-bound")
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        retained_count = 0
        digest = hashlib.sha256()
        bytes_read = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, opened.st_size - bytes_read + 1),
            )
            if not chunk:
                break
            bytes_read += len(chunk)
            # As above, the opened descriptor is an exact snapshot contract:
            # any post-open growth fails immediately.
            if bytes_read > opened.st_size:
                raise RuntimeIdentityError(f"runtime-file-replaced:{target}")
            if bytes_read > _MAX_RUNTIME_FILE_BYTES:
                raise RuntimeIdentityError("runtime-file-exceeds-reader-bound")
            if retained_bytes is None:
                chunks.append(chunk)
            elif retained_count < retained_bytes:
                retained = chunk[: retained_bytes - retained_count]
                chunks.append(retained)
                retained_count += len(retained)
            digest.update(chunk)
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = target.lstat()
    except OSError as exc:
        raise RuntimeIdentityError(f"runtime-file-replaced:{target}") from exc
    if not (
        bytes_read == opened.st_size
        and _stable_metadata(before)
        == _stable_metadata(opened)
        == _stable_metadata(after_read)
        == _stable_metadata(after_path)
    ):
        raise RuntimeIdentityError(f"runtime-file-replaced:{target}")
    return b"".join(chunks), {
        "path": str(target),
        "bytes": opened.st_size,
        "device": opened.st_dev,
        "inode": opened.st_ino,
        "links": opened.st_nlink,
        "mode": stat.S_IMODE(opened.st_mode),
        "uid": opened.st_uid,
        "gid": opened.st_gid,
        "mtime_ns": opened.st_mtime_ns,
        "ctime_ns": opened.st_ctime_ns,
        "sha256": digest.hexdigest(),
    }


def _path_is_beneath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_safe_root_owned_path(path: Path, *, directory: bool) -> os.stat_result:
    """Require every existing component to be root-owned and non-link."""

    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise RuntimeIdentityError(f"runtime-path-unavailable:{current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeIdentityError(f"runtime-path-symlink:{current}")
        if metadata.st_uid != 0 or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeIdentityError(f"runtime-path-not-root-controlled:{current}")
    if directory and not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeIdentityError(f"runtime-path-not-directory:{path}")
    if not directory and not stat.S_ISREG(metadata.st_mode):
        raise RuntimeIdentityError(f"runtime-path-not-regular-file:{path}")
    return metadata


def regular_file_identity(path: str | Path) -> dict[str, Any]:
    _, identity = _read_stable_regular_file(Path(path), retained_bytes=0)
    return identity


def tree_manifest(
    root: str | Path,
    *,
    reject_symlinks: bool,
    require_root_owned: bool,
    excluded_top_level_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Hash a directory tree without ever following a descendant symlink."""

    canonical_root = Path(root).absolute()
    if require_root_owned:
        root_metadata = _require_safe_root_owned_path(canonical_root, directory=True)
    else:
        try:
            root_metadata = canonical_root.lstat()
        except OSError as exc:
            raise RuntimeIdentityError(
                f"runtime-tree-unavailable:{canonical_root}"
            ) from exc
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
            root_metadata.st_mode
        ):
            raise RuntimeIdentityError(f"runtime-tree-invalid:{canonical_root}")

    excluded = frozenset(excluded_top_level_names)
    entries: list[dict[str, Any]] = []
    file_count = 0
    total_bytes = 0

    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)

    def walk(directory_fd: int, relative_parent: str) -> None:
        nonlocal file_count, total_bytes
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise RuntimeIdentityError("runtime-tree-scan-failed") from exc
        for name in names:
            try:
                name.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise RuntimeIdentityError("runtime-tree-path-not-utf8") from exc
            relative = f"{relative_parent}/{name}" if relative_parent else name
            if "/" not in relative and relative in excluded:
                continue
            try:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise RuntimeIdentityError("runtime-tree-stat-failed") from exc
            if require_root_owned and (
                metadata.st_uid != 0 or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            ):
                raise RuntimeIdentityError(
                    f"runtime-tree-entry-not-root-controlled:{relative}"
                )
            common = {
                "path": relative,
                "mode": stat.S_IMODE(metadata.st_mode),
                "uid": metadata.st_uid,
                "gid": metadata.st_gid,
            }
            if stat.S_ISREG(metadata.st_mode):
                try:
                    descriptor = os.open(
                        name, os.O_RDONLY | no_follow, dir_fd=directory_fd
                    )
                except OSError as exc:
                    raise RuntimeIdentityError("runtime-tree-open-failed") from exc
                try:
                    opened = os.fstat(descriptor)
                    digest = _hash_file_descriptor(descriptor)
                    after_read = os.fstat(descriptor)
                finally:
                    os.close(descriptor)
                after_path = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if not (
                    _stable_metadata(metadata)
                    == _stable_metadata(opened)
                    == _stable_metadata(after_read)
                    == _stable_metadata(after_path)
                ):
                    raise RuntimeIdentityError(
                        f"runtime-tree-entry-replaced:{relative}"
                    )
                entries.append(
                    {
                        **common,
                        "kind": "file",
                        "bytes": metadata.st_size,
                        "sha256": digest,
                    }
                )
                file_count += 1
                total_bytes += metadata.st_size
            elif stat.S_ISDIR(metadata.st_mode):
                entries.append({**common, "kind": "directory"})
                try:
                    child_descriptor = os.open(
                        name,
                        os.O_RDONLY | directory_flag | no_follow,
                        dir_fd=directory_fd,
                    )
                except OSError as exc:
                    raise RuntimeIdentityError("runtime-tree-open-failed") from exc
                try:
                    opened = os.fstat(child_descriptor)
                    if _stable_metadata(metadata) != _stable_metadata(opened):
                        raise RuntimeIdentityError(
                            f"runtime-tree-entry-replaced:{relative}"
                        )
                    walk(child_descriptor, relative)
                    after_scan = os.fstat(child_descriptor)
                finally:
                    os.close(child_descriptor)
                after_path = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if not (
                    _stable_metadata(opened)
                    == _stable_metadata(after_scan)
                    == _stable_metadata(after_path)
                ):
                    raise RuntimeIdentityError(
                        f"runtime-tree-entry-replaced:{relative}"
                    )
            elif stat.S_ISLNK(metadata.st_mode):
                if reject_symlinks:
                    raise RuntimeIdentityError(f"runtime-tree-symlink:{relative}")
                try:
                    link_target = os.readlink(name, dir_fd=directory_fd)
                except OSError as exc:
                    raise RuntimeIdentityError("runtime-tree-readlink-failed") from exc
                after_link = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if _stable_metadata(metadata) != _stable_metadata(after_link):
                    raise RuntimeIdentityError(
                        f"runtime-tree-entry-replaced:{relative}"
                    )
                entries.append({**common, "kind": "symlink", "target": link_target})
            else:
                raise RuntimeIdentityError(f"runtime-tree-special-entry:{relative}")

    try:
        root_descriptor = os.open(
            canonical_root, os.O_RDONLY | directory_flag | no_follow
        )
    except OSError as exc:
        raise RuntimeIdentityError("runtime-tree-open-failed") from exc
    try:
        root_before = os.fstat(root_descriptor)
        walk(root_descriptor, "")
        root_after = os.fstat(root_descriptor)
    finally:
        os.close(root_descriptor)
    try:
        root_path_after = canonical_root.lstat()
    except OSError as exc:
        raise RuntimeIdentityError("runtime-tree-root-replaced") from exc
    if not (
        _stable_metadata(root_metadata)
        == _stable_metadata(root_before)
        == _stable_metadata(root_after)
        == _stable_metadata(root_path_after)
    ):
        raise RuntimeIdentityError("runtime-tree-root-mutated")
    payload = {
        "format": TREE_MANIFEST_FORMAT,
        "root": str(canonical_root),
        "entries": entries,
        "file_count": file_count,
        "total_bytes": total_bytes,
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def trusted_git_subprocess_environment() -> dict[str, str]:
    """Return the exhaustive environment for direct CLT Git execution."""

    return {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_COUNT": "4",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_KEY_0": "core.excludesFile",
        "GIT_CONFIG_KEY_1": "core.attributesFile",
        "GIT_CONFIG_KEY_2": "core.fsmonitor",
        "GIT_CONFIG_KEY_3": "core.hooksPath",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_VALUE_0": os.devnull,
        "GIT_CONFIG_VALUE_1": os.devnull,
        "GIT_CONFIG_VALUE_2": "false",
        "GIT_CONFIG_VALUE_3": os.devnull,
        "GIT_EXEC_PATH": str(CLT_GIT_EXEC_PATH),
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "DEVELOPER_DIR": "/Library/Developer/CommandLineTools",
        "HOME": "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": "/tmp",
        "XDG_CONFIG_HOME": "/var/empty",
    }


def trusted_clt_git_executable() -> Path:
    _require_safe_root_owned_path(CLT_GIT_EXECUTABLE, directory=False)
    _require_safe_root_owned_path(CLT_GIT_EXEC_PATH, directory=True)
    return CLT_GIT_EXECUTABLE


def trusted_git_argv(arguments: Sequence[str]) -> tuple[str, ...]:
    return (str(trusted_clt_git_executable()), *arguments)


def _run_identity_command(
    argv: Sequence[str], environment: Mapping[str, str], *, timeout: float = 30.0
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(argv),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise RuntimeIdentityError("runtime-identity-command-failed") from exc
    if result.returncode != 0:
        raise RuntimeIdentityError("runtime-identity-command-failed")
    return result


def observe_git_runtime_identity() -> dict[str, Any]:
    executable = trusted_clt_git_executable()
    environment = trusted_git_subprocess_environment()
    version = _run_identity_command((str(executable), "--version"), environment)
    exec_path = _run_identity_command((str(executable), "--exec-path"), environment)
    builtins_result = _run_identity_command(
        (str(executable), "--list-cmds=builtins"), environment
    )
    reported_exec_path = Path(exec_path.stdout.strip()).absolute()
    if reported_exec_path != CLT_GIT_EXEC_PATH:
        raise RuntimeIdentityError("git-exec-path-mismatch")
    builtins = sorted(filter(None, builtins_result.stdout.splitlines()))
    if not _GIT_REQUIRED_BUILTINS.issubset(builtins):
        raise RuntimeIdentityError("git-required-command-not-builtin")
    exec_path_manifest = tree_manifest(
        CLT_GIT_EXEC_PATH,
        reject_symlinks=False,
        require_root_owned=True,
    )
    trusted_clt_usr = Path("/Library/Developer/CommandLineTools/usr")
    symlink_targets: dict[Path, dict[str, Any]] = {}
    for entry in exec_path_manifest["entries"]:
        if entry["kind"] != "symlink":
            continue
        link_path = CLT_GIT_EXEC_PATH / entry["path"]
        try:
            target = link_path.resolve(strict=True)
        except OSError as exc:
            raise RuntimeIdentityError("git-exec-path-link-unresolved") from exc
        if not _path_is_beneath(target, trusted_clt_usr):
            raise RuntimeIdentityError("git-exec-path-link-outside-clt")
        _require_safe_root_owned_path(target, directory=False)
        symlink_targets[target] = regular_file_identity(target)
    payload = {
        "format": GIT_RUNTIME_IDENTITY_FORMAT,
        "executable": regular_file_identity(executable),
        "exec_path": exec_path_manifest,
        "exec_path_symlink_targets": [
            symlink_targets[path] for path in sorted(symlink_targets, key=str)
        ],
        "environment": environment,
        "environment_sha256": canonical_sha256(environment),
        "version": version.stdout.strip(),
        "version_stdout_sha256": hashlib.sha256(version.stdout.encode()).hexdigest(),
        "version_stderr_sha256": hashlib.sha256(version.stderr.encode()).hexdigest(),
        "version_stdout_bytes": len(version.stdout.encode()),
        "version_stderr_bytes": len(version.stderr.encode()),
        "required_builtins": sorted(_GIT_REQUIRED_BUILTINS),
        "reported_builtins_sha256": canonical_sha256(builtins),
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def authenticate_git_runtime_identity(expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = observe_git_runtime_identity()
    if observed != dict(expected):
        raise RuntimeIdentityError("git-runtime-identity-mismatch")
    return observed


def observe_go_runtime_identity(
    go_executable: str | Path, environment: Mapping[str, str]
) -> dict[str, Any]:
    """Observe a closed Go runtime, including every byte beneath GOROOT."""

    executable = trusted_root_owned_go_executable(go_executable)
    strict_environment = dict(sorted(environment.items()))
    if strict_environment.get("CGO_ENABLED") != "0":
        raise RuntimeIdentityError("go-runtime-cgo-not-disabled")
    argv = (str(executable), "env", "-json", *_GO_ENV_KEYS)
    result = _run_identity_command(argv, strict_environment)
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeIdentityError("go-runtime-env-invalid") from exc
    if not isinstance(raw, dict) or set(raw) != set(_GO_ENV_KEYS):
        raise RuntimeIdentityError("go-runtime-env-invalid")
    go_env = {key: raw[key] for key in _GO_ENV_KEYS}
    if any(not isinstance(value, str) or not value for value in go_env.values()):
        raise RuntimeIdentityError("go-runtime-env-invalid")
    if go_env["CGO_ENABLED"] != "0":
        raise RuntimeIdentityError("go-runtime-cgo-not-disabled")
    goroot = Path(go_env["GOROOT"]).absolute()
    gotooldir = Path(go_env["GOTOOLDIR"]).absolute()
    _require_safe_root_owned_path(goroot, directory=True)
    _require_safe_root_owned_path(gotooldir, directory=True)
    if not _path_is_beneath(gotooldir, goroot):
        raise RuntimeIdentityError("go-runtime-tool-dir-outside-root")
    manifest = tree_manifest(
        goroot,
        reject_symlinks=True,
        require_root_owned=True,
    )
    gotooldir_relative = gotooldir.relative_to(goroot).as_posix()
    tool_prefix = f"{gotooldir_relative}/"
    tool_entries = []
    for entry in manifest["entries"]:
        relative = entry["path"]
        if relative.startswith(tool_prefix):
            tool_entries.append({**entry, "path": relative.removeprefix(tool_prefix)})
    gotooldir_payload = {
        "format": TREE_MANIFEST_FORMAT,
        "root": str(gotooldir),
        "entries": tool_entries,
        "file_count": sum(entry["kind"] == "file" for entry in tool_entries),
        "total_bytes": sum(
            entry.get("bytes", 0) for entry in tool_entries if entry["kind"] == "file"
        ),
    }
    gotooldir_manifest = {
        **gotooldir_payload,
        "manifest_sha256": canonical_sha256(gotooldir_payload),
    }
    payload = {
        "format": GO_RUNTIME_IDENTITY_FORMAT,
        "executable": regular_file_identity(executable),
        "go_env_argv": list(argv),
        "go_env_argv_sha256": canonical_sha256(list(argv)),
        "go_env": go_env,
        "go_env_sha256": canonical_sha256(go_env),
        "goroot_manifest": manifest,
        "goroot_manifest_sha256": manifest["manifest_sha256"],
        "gotooldir_relative_path": gotooldir_relative,
        "gotooldir_manifest": gotooldir_manifest,
        "gotooldir_manifest_sha256": gotooldir_manifest["manifest_sha256"],
        "environment": strict_environment,
        "environment_sha256": canonical_sha256(strict_environment),
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def trusted_root_owned_go_executable(go_executable: str | Path) -> Path:
    """Validate a Go launcher against the root-owned, no-symlink host TCB."""

    executable = Path(go_executable).absolute()
    if executable.name != "go":
        raise RuntimeIdentityError("go-runtime-executable-name-invalid")
    _require_safe_root_owned_path(executable, directory=False)
    return executable


def authenticate_go_runtime_identity(
    expected: Mapping[str, Any],
    go_executable: str | Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    observed = observe_go_runtime_identity(go_executable, environment)
    if observed != dict(expected):
        raise RuntimeIdentityError("go-runtime-identity-mismatch")
    return observed


# Mach-O load commands used to discover the transitive native dependency graph.
_MACHO_32_MAGICS = {b"\xfe\xed\xfa\xce": ">", b"\xce\xfa\xed\xfe": "<"}
_MACHO_64_MAGICS = {b"\xfe\xed\xfa\xcf": ">", b"\xcf\xfa\xed\xfe": "<"}
_FAT_MAGICS = {
    b"\xca\xfe\xba\xbe": (">", False),
    b"\xbe\xba\xfe\xca": ("<", False),
    b"\xca\xfe\xba\xbf": (">", True),
    b"\xbf\xba\xfe\xca": ("<", True),
}
_DYLIB_COMMANDS = {0xC, 0x18, 0x1F, 0x23, 0x20}
_LC_ID_DYLIB = 0xD
_LC_RPATH = 0x1C
_LC_REQ_DYLD = 0x80000000
_MH_EXECUTE = 0x2
_MH_BUNDLE = 0x8


def _macho_slices(data: bytes) -> list[bytes]:
    if len(data) < 4:
        return []
    fat = _FAT_MAGICS.get(data[:4])
    if fat is None:
        return [data] if data[:4] in {*_MACHO_32_MAGICS, *_MACHO_64_MAGICS} else []
    endian, is_64 = fat
    if len(data) < 8:
        raise RuntimeIdentityError("macho-fat-header-truncated")
    count = struct.unpack_from(f"{endian}I", data, 4)[0]
    entry_size = 32 if is_64 else 20
    slices: list[bytes] = []
    for index in range(count):
        offset = 8 + index * entry_size
        if offset + entry_size > len(data):
            raise RuntimeIdentityError("macho-fat-arch-truncated")
        if is_64:
            file_offset, size = struct.unpack_from(f"{endian}QQ", data, offset + 8)
        else:
            file_offset, size = struct.unpack_from(f"{endian}II", data, offset + 8)
        if file_offset + size > len(data):
            raise RuntimeIdentityError("macho-fat-slice-truncated")
        slices.append(data[file_offset : file_offset + size])
    return slices


def _read_macho_paths(
    path: Path,
) -> tuple[list[str], list[str], dict[str, Any], bool, bool, list[str]]:
    # Most frozen distribution payloads are not Mach-O binaries (notably MLX
    # metallib files). Stream and hash those under the runtime bound while
    # retaining only their magic, rather than materializing them in memory.
    magic, probe_identity = _read_stable_regular_file(path, retained_bytes=4)
    if magic not in {
        *_MACHO_32_MAGICS,
        *_MACHO_64_MAGICS,
        *_FAT_MAGICS,
    }:
        return [], [], probe_identity, False, False, []
    data, identity = _read_stable_regular_file(path)
    if identity != probe_identity:
        raise RuntimeIdentityError(f"runtime-file-replaced:{path}")
    slices = _macho_slices(data)
    dependencies: list[str] = []
    rpaths: list[str] = []
    install_names: list[str] = []
    is_entry_image = False

    def append_once(values: list[str], item: str) -> None:
        if item not in values:
            values.append(item)

    for image in slices:
        magic = image[:4]
        if magic in _MACHO_64_MAGICS:
            endian = _MACHO_64_MAGICS[magic]
            header_size = 32
        elif magic in _MACHO_32_MAGICS:
            endian = _MACHO_32_MAGICS[magic]
            header_size = 28
        else:
            continue
        if len(image) < header_size:
            raise RuntimeIdentityError("macho-header-truncated")
        filetype = struct.unpack_from(f"{endian}I", image, 12)[0]
        if filetype in {_MH_EXECUTE, _MH_BUNDLE}:
            is_entry_image = True
        command_count, command_bytes = struct.unpack_from(f"{endian}II", image, 16)
        end = header_size + command_bytes
        if end > len(image):
            raise RuntimeIdentityError("macho-load-commands-truncated")
        cursor = header_size
        for _ in range(command_count):
            if cursor + 8 > end:
                raise RuntimeIdentityError("macho-load-command-truncated")
            command, command_size = struct.unpack_from(f"{endian}II", image, cursor)
            if command_size < 8 or cursor + command_size > end:
                raise RuntimeIdentityError("macho-load-command-invalid")
            base_command = command & ~_LC_REQ_DYLD
            if (
                base_command in _DYLIB_COMMANDS
                or base_command == _LC_ID_DYLIB
                or base_command == _LC_RPATH
            ):
                if command_size < 12:
                    raise RuntimeIdentityError("macho-string-command-invalid")
                string_offset = struct.unpack_from(f"{endian}I", image, cursor + 8)[0]
                if string_offset >= command_size:
                    raise RuntimeIdentityError("macho-string-offset-invalid")
                raw = image[cursor + string_offset : cursor + command_size]
                raw = raw.split(b"\0", 1)[0]
                try:
                    value = raw.decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    raise RuntimeIdentityError("macho-path-not-utf8") from exc
                if base_command == _LC_RPATH:
                    append_once(rpaths, value)
                elif base_command == _LC_ID_DYLIB:
                    append_once(install_names, value)
                elif value:
                    append_once(dependencies, value)
            cursor += command_size
    return dependencies, rpaths, identity, bool(slices), is_entry_image, install_names


def _is_trusted_os_dependency(path: Path) -> bool:
    absolute = path.absolute()
    return any(
        _path_is_beneath(absolute, prefix) for prefix in TRUSTED_OS_DEPENDENCY_PREFIXES
    )


def _lexical_absolute_path(path: Path) -> Path:
    """Normalize dot segments without resolving any potentially mutable link."""

    return Path(os.path.abspath(path))


def _expand_loader_token(value: str, *, loader: Path, executable: Path) -> Path | None:
    if value == "@loader_path":
        return loader.parent
    if value.startswith("@loader_path/"):
        return loader.parent / value.removeprefix("@loader_path/")
    if value == "@executable_path":
        return executable.parent
    if value.startswith("@executable_path/"):
        return executable.parent / value.removeprefix("@executable_path/")
    if value.startswith("/"):
        return Path(value)
    return None


def _resolve_macho_dependency(
    value: str,
    *,
    loader: Path,
    executable: Path,
    rpaths: Sequence[Path],
    owned_roots: Sequence[Path],
) -> Path:
    candidates: list[Path] = []
    if value.startswith("@rpath/"):
        suffix = value.removeprefix("@rpath/")
        for rpath in rpaths:
            candidates.append(rpath / suffix)
    else:
        expanded = _expand_loader_token(value, loader=loader, executable=executable)
        if expanded is not None:
            candidates.append(expanded)
    if not candidates and not value.startswith("@"):
        candidates.append(loader.parent / value)
        candidates.extend(root / "lib" / value for root in owned_roots)
    for candidate in candidates:
        candidate = _lexical_absolute_path(candidate)
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeIdentityError("macho-dependency-unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeIdentityError("macho-dependency-symlink")
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeIdentityError("macho-dependency-not-regular")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise RuntimeIdentityError("macho-dependency-unavailable") from exc
        # lstat above only protects the final name.  Reject an intermediate
        # symlink too: a retargetable rpath component is not an owned payload.
        if resolved != candidate:
            raise RuntimeIdentityError("macho-dependency-symlink")
        return resolved
    raise RuntimeIdentityError(f"macho-dependency-unresolved:{value}")


def _resolved_macho_rpaths(
    rpaths: Sequence[str], *, loader: Path, executable: Path
) -> tuple[Path, ...]:
    """Expand a loader's declared LC_RPATH entries into inherited candidates."""

    resolved: list[Path] = []
    for value in rpaths:
        path = _expand_loader_token(value, loader=loader, executable=executable)
        if path is not None:
            absolute = _lexical_absolute_path(path)
            if absolute not in resolved:
                resolved.append(absolute)
    return tuple(resolved)


def _dedupe_macho_rpaths(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Preserve dyld first-match order while removing repeated path frames."""

    ordered: list[Path] = []
    for path in paths:
        absolute = _lexical_absolute_path(path)
        if absolute not in ordered:
            ordered.append(absolute)
    return tuple(ordered)


def _stdlib_root() -> Path:
    value = sysconfig.get_path("stdlib")
    if not value:
        raise RuntimeIdentityError("python-stdlib-unavailable")
    return Path(value)


def observe_macho_runtime_closure(
    executable: str | Path,
    *,
    required_binary_paths: Iterable[str | Path] = (),
    preloaded_binary_paths: Iterable[str | Path] = (),
    owned_native_roots: Iterable[str | Path],
) -> dict[str, Any]:
    """Close and content-bind a required Mach-O dependency graph.

    Every non-system dependency must resolve beneath one supplied owned root.
    Paths in the returned identity are root-relative so a private snapshot can
    produce a stable receipt independent of its temporary directory name.
    """

    roots = tuple(
        sorted(
            {Path(root).absolute().resolve(strict=True) for root in owned_native_roots},
            key=str,
        )
    )
    if not roots:
        raise RuntimeIdentityError("macho-owned-root-required")

    def required_file(value: str | Path) -> Path:
        supplied = Path(value).absolute()
        try:
            metadata = supplied.lstat()
        except OSError as exc:
            raise RuntimeIdentityError("macho-input-unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeIdentityError("macho-input-symlink")
        try:
            resolved = supplied.resolve(strict=True)
        except OSError as exc:
            raise RuntimeIdentityError("macho-input-unavailable") from exc
        if not any(_path_is_beneath(resolved, root) for root in roots):
            raise RuntimeIdentityError("macho-input-outside-trust-root")
        return resolved

    executable_path = required_file(executable)
    preload_paths = {required_file(path) for path in preloaded_binary_paths}
    required_paths = {
        executable_path,
        *(required_file(path) for path in required_binary_paths),
        *preload_paths,
    }
    initial_observations = {
        path: _read_macho_paths(path) for path in sorted(required_paths, key=str)
    }
    observed_files: dict[Path, dict[str, Any]] = {
        path: observation[2] for path, observation in initial_observations.items()
    }
    preloaded_by_install_name: dict[str, Path] = {}
    for path in sorted(preload_paths, key=str):
        install_names = initial_observations[path][5]
        if len(install_names) != 1:
            raise RuntimeIdentityError("macho-preload-install-name-invalid")
        install_name = install_names[0]
        prior = preloaded_by_install_name.setdefault(install_name, path)
        if prior != path:
            raise RuntimeIdentityError("macho-preload-install-name-collision")
    pending: deque[tuple[Path, tuple[Path, ...]]] = deque(
        (path, ()) for path in sorted(required_paths, key=str)
    )
    visited: set[tuple[Path, tuple[Path, ...]]] = set()
    edges: set[tuple[Path, str, Path]] = set()
    trusted_system: set[str] = set()
    while pending:
        loader, inherited_rpaths = pending.popleft()
        context = (loader, inherited_rpaths)
        if context in visited:
            continue
        if len(visited) >= _MAX_MACHO_RESOLUTION_CONTEXTS:
            raise RuntimeIdentityError("macho-resolution-context-limit")
        visited.add(context)
        dependencies, rpaths, loader_identity, _, _, _ = _read_macho_paths(loader)
        prior_identity = observed_files.get(loader)
        if prior_identity is not None and prior_identity != loader_identity:
            raise RuntimeIdentityError("macho-runtime-mutated-during-observation")
        observed_files[loader] = loader_identity
        effective_rpaths = _dedupe_macho_rpaths(
            (
                *_resolved_macho_rpaths(
                    rpaths,
                    loader=loader,
                    executable=executable_path,
                ),
                *inherited_rpaths,
            )
        )
        for dependency in dependencies:
            if dependency.startswith("/"):
                declared_absolute = Path(os.path.normpath(dependency))
                if _is_trusted_os_dependency(declared_absolute):
                    edges.add((loader, dependency, declared_absolute))
                    trusted_system.add(str(declared_absolute))
                    continue
            resolved = preloaded_by_install_name.get(dependency)
            if resolved is None:
                resolved = _resolve_macho_dependency(
                    dependency,
                    loader=loader,
                    executable=executable_path,
                    rpaths=effective_rpaths,
                    owned_roots=roots,
                )
            edges.add((loader, dependency, resolved))
            if _is_trusted_os_dependency(resolved):
                trusted_system.add(str(resolved))
                continue
            if not any(_path_is_beneath(resolved, root) for root in roots):
                raise RuntimeIdentityError("macho-dependency-outside-trust-root")
            pending.append((resolved, effective_rpaths))

    if any(
        regular_file_identity(path) != identity
        for path, identity in observed_files.items()
    ):
        raise RuntimeIdentityError("macho-runtime-mutated-during-observation")

    def normalized_path(path: Path) -> str:
        if _is_trusted_os_dependency(path):
            return f"system:{path}"
        for index, root in enumerate(roots):
            if _path_is_beneath(path, root):
                relative = path.relative_to(root).as_posix()
                return f"root:{index}/{relative}"
        raise RuntimeIdentityError("macho-dependency-outside-trust-root")

    files = [
        {
            "path": normalized_path(path),
            "bytes": identity["bytes"],
            "sha256": identity["sha256"],
        }
        for path, identity in sorted(
            observed_files.items(), key=lambda item: str(item[0])
        )
    ]
    payload = {
        "format": MACHO_RUNTIME_CLOSURE_FORMAT,
        "executable": normalized_path(executable_path),
        "required_inputs": [
            normalized_path(path) for path in sorted(required_paths, key=str)
        ],
        "preloaded_images": [
            {"install_name": install_name, "path": normalized_path(path)}
            for install_name, path in sorted(preloaded_by_install_name.items())
        ],
        "files": sorted(files, key=lambda item: item["path"]),
        "edges": [
            {
                "loader": normalized_path(loader),
                "declared": declared,
                "resolved": normalized_path(resolved),
            }
            for loader, declared, resolved in sorted(
                edges,
                key=lambda item: (str(item[0]), item[1], str(item[2])),
            )
        ],
        "trusted_system_dependencies": sorted(trusted_system),
        "trusted_system_prefixes": [
            str(path) for path in TRUSTED_OS_DEPENDENCY_PREFIXES
        ],
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def observe_python_runtime_identity(
    *,
    executable: str | Path = sys.executable,
    prefix: str | Path = sys.base_prefix,
    stdlib: str | Path | None = None,
    additional_binary_paths: Iterable[str | Path] = (),
    owned_native_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Hash Python code plus recursive non-Apple native dependencies.

    ``additional_binary_paths`` must include native files from any frozen owned
    distributions that the caller will import. Their Python source remains the
    caller's distribution-manifest responsibility; this function closes their
    Mach-O loader graph.
    """

    executable_path = Path(executable).absolute().resolve(strict=True)
    owned_prefix = Path(prefix).absolute().resolve(strict=True)
    owned_roots = tuple(
        sorted(
            {
                owned_prefix,
                *(
                    Path(path).absolute().resolve(strict=True)
                    for path in owned_native_roots
                ),
            },
            key=str,
        )
    )
    stdlib_path = Path(stdlib) if stdlib is not None else _stdlib_root()
    stdlib_path = stdlib_path.absolute().resolve(strict=True)
    if not _path_is_beneath(executable_path, owned_prefix):
        raise RuntimeIdentityError("python-executable-outside-prefix")
    if not _path_is_beneath(stdlib_path, owned_prefix):
        raise RuntimeIdentityError("python-stdlib-outside-prefix")
    stdlib_manifest = tree_manifest(
        stdlib_path,
        reject_symlinks=True,
        require_root_owned=False,
        excluded_top_level_names=("lib-dynload", "site-packages"),
    )
    lib_dynload = stdlib_path / "lib-dynload"
    lib_dynload_manifest = (
        tree_manifest(
            lib_dynload,
            reject_symlinks=True,
            require_root_owned=False,
        )
        if lib_dynload.is_dir()
        else None
    )

    initial_paths = {executable_path}
    if lib_dynload.is_dir():
        for candidate in lib_dynload.iterdir():
            if candidate.is_file() and not candidate.is_symlink():
                initial_paths.add(candidate.resolve(strict=True))
    for candidate in additional_binary_paths:
        supplied_path = Path(candidate).absolute()
        try:
            supplied_metadata = supplied_path.lstat()
        except OSError as exc:
            raise RuntimeIdentityError("python-owned-binary-unavailable") from exc
        if stat.S_ISLNK(supplied_metadata.st_mode):
            raise RuntimeIdentityError("python-owned-binary-symlink")
        try:
            path = supplied_path.resolve(strict=True)
        except OSError as exc:
            raise RuntimeIdentityError("python-owned-binary-unavailable") from exc
        if path != supplied_path:
            raise RuntimeIdentityError("python-owned-binary-symlink")
        if not any(_path_is_beneath(path, root) for root in owned_roots):
            raise RuntimeIdentityError("python-owned-binary-outside-prefix")
        initial_paths.add(path)

    # Authenticate every supplied root before resolving its dependencies.  The
    # entry-image classification establishes whether a root's unresolved
    # @rpath dependencies are mandatory; standalone dylib roots remain
    # supplied and hashed, but are not loader entry points by themselves.
    initial_observations = {
        path: _read_macho_paths(path) for path in sorted(initial_paths, key=str)
    }
    observed_files: dict[Path, dict[str, Any]] = {
        path: observation[2] for path, observation in initial_observations.items()
    }
    entry_image_roots = {
        path for path, observation in initial_observations.items() if observation[4]
    }
    pending: deque[tuple[Path, tuple[Path, ...], bool]] = deque(
        (path, (), path in entry_image_roots) for path in sorted(initial_paths, key=str)
    )
    # Resolution depends only on the loader, its ordered inherited LC_RPATH
    # frame, and whether that loader is required.  This finite semantic graph
    # deduplicates cycles and diamonds without suppressing a distinct rpath
    # context just because it appears in an earlier ancestry.
    visited: set[tuple[Path, tuple[Path, ...], bool]] = set()
    native_files: list[dict[str, Any]] = []
    native_file_paths: set[Path] = set()
    native_edges: set[tuple[str, str, str]] = set()
    trusted_system: set[str] = set()
    while pending:
        loader, inherited_rpaths, context_required = pending.popleft()
        context = (loader, inherited_rpaths, context_required)
        if context in visited:
            continue
        if len(visited) >= _MAX_MACHO_RESOLUTION_CONTEXTS:
            raise RuntimeIdentityError("macho-resolution-context-limit")
        visited.add(context)
        (
            dependencies,
            rpaths,
            loader_identity,
            is_macho,
            _,
            _,
        ) = _read_macho_paths(loader)
        prior_identity = observed_files.get(loader)
        if prior_identity is not None and prior_identity != loader_identity:
            raise RuntimeIdentityError("python-runtime-mutated-during-observation")
        observed_files[loader] = loader_identity
        if is_macho and loader not in native_file_paths:
            native_files.append(loader_identity)
            native_file_paths.add(loader)
        # LC_RPATH entries declared by the current image take precedence.  The
        # ordered inherited frame is then available for dependencies such as
        # MLX's libmlx -> libjaccl chain, but never invented as a fallback.
        effective_rpaths = _dedupe_macho_rpaths(
            (
                *_resolved_macho_rpaths(
                    rpaths, loader=loader, executable=executable_path
                ),
                *inherited_rpaths,
            )
        )
        # A non-Mach-O initial path is harmless (for example a script shim), but
        # still content-bound as an input file below.
        for dependency in dependencies:
            if dependency.startswith("/"):
                declared_absolute = Path(os.path.normpath(dependency))
                if _is_trusted_os_dependency(declared_absolute):
                    native_edges.add((str(loader), dependency, str(declared_absolute)))
                    trusted_system.add(str(declared_absolute))
                    continue
            try:
                resolved = _resolve_macho_dependency(
                    dependency,
                    loader=loader,
                    executable=executable_path,
                    rpaths=effective_rpaths,
                    owned_roots=owned_roots,
                )
            except RuntimeIdentityError as exc:
                if (
                    dependency.startswith("@rpath/")
                    and not context_required
                    and str(exc) == f"macho-dependency-unresolved:{dependency}"
                ):
                    continue
                raise
            native_edges.add((str(loader), dependency, str(resolved)))
            if _is_trusted_os_dependency(resolved):
                trusted_system.add(str(resolved))
                continue
            if not any(_path_is_beneath(resolved, root) for root in owned_roots):
                raise RuntimeIdentityError(
                    "python-native-dependency-outside-trust-root"
                )
            # Every edge carries the current ordered frame and requiredness to
            # its child.  Enqueue even an already-seen context; the semantic
            # visited set above is the cycle/diamond termination boundary.
            child_context = (
                resolved,
                effective_rpaths,
                context_required or resolved in entry_image_roots,
            )
            pending.append(child_context)

    ending_stdlib_manifest = tree_manifest(
        stdlib_path,
        reject_symlinks=True,
        require_root_owned=False,
        excluded_top_level_names=("lib-dynload", "site-packages"),
    )
    ending_lib_dynload_manifest = (
        tree_manifest(
            lib_dynload,
            reject_symlinks=True,
            require_root_owned=False,
        )
        if lib_dynload.is_dir()
        else None
    )
    if (
        ending_stdlib_manifest != stdlib_manifest
        or ending_lib_dynload_manifest != lib_dynload_manifest
        or any(
            regular_file_identity(path) != identity
            for path, identity in observed_files.items()
        )
    ):
        raise RuntimeIdentityError("python-runtime-mutated-during-observation")

    input_files = [observed_files[path] for path in sorted(initial_paths, key=str)]
    payload = {
        "format": PYTHON_RUNTIME_IDENTITY_FORMAT,
        "python": {
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "version": list(sys.version_info[:5]),
            "abiflags": getattr(sys, "abiflags", ""),
            "prefix": str(Path(sys.prefix).absolute()),
            "base_prefix": str(Path(sys.base_prefix).absolute()),
        },
        "executable": observed_files[executable_path],
        "prefix": str(owned_prefix),
        "owned_native_roots": [str(root) for root in owned_roots],
        "stdlib": stdlib_manifest,
        "stdlib_manifest_sha256": stdlib_manifest["manifest_sha256"],
        "lib_dynload": lib_dynload_manifest,
        "lib_dynload_manifest_sha256": (
            lib_dynload_manifest["manifest_sha256"]
            if lib_dynload_manifest is not None
            else None
        ),
        "native_input_files": input_files,
        "native_files": sorted(native_files, key=lambda item: item["path"]),
        "native_edges": [
            {"loader": loader, "declared": declared, "resolved": resolved}
            for loader, declared, resolved in sorted(native_edges)
        ],
        "trusted_system_dependencies": sorted(trusted_system),
        "trusted_system_prefixes": [
            str(path) for path in TRUSTED_OS_DEPENDENCY_PREFIXES
        ],
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def authenticate_python_runtime_identity(
    expected: Mapping[str, Any],
    *,
    executable: str | Path = sys.executable,
    prefix: str | Path = sys.base_prefix,
    stdlib: str | Path | None = None,
    additional_binary_paths: Iterable[str | Path] = (),
    owned_native_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    observed = observe_python_runtime_identity(
        executable=executable,
        prefix=prefix,
        stdlib=stdlib,
        additional_binary_paths=additional_binary_paths,
        owned_native_roots=owned_native_roots,
    )
    if observed != dict(expected):
        raise RuntimeIdentityError("python-runtime-identity-mismatch")
    return observed
