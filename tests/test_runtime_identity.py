from __future__ import annotations

import ast
import json
import os
import stat
import struct
from pathlib import Path

import pytest

import graph_model.runtime_identity as runtime_identity
import graph_model.workspace as workspace_module
from graph_model.runtime_identity import (
    RuntimeIdentityError,
    authenticate_go_runtime_identity,
    authenticate_git_runtime_identity,
    authenticate_python_runtime_identity,
    observe_go_runtime_identity,
    observe_git_runtime_identity,
    observe_macho_runtime_closure,
    observe_python_runtime_identity,
    regular_file_identity,
    tree_manifest,
    trusted_clt_git_executable,
    trusted_git_subprocess_environment,
)


def _macho_command(command: int, value: str, *, dylib: bool) -> bytes:
    encoded = value.encode() + b"\0"
    fixed = 24 if dylib else 12
    size = (fixed + len(encoded) + 7) & ~7
    if dylib:
        header = struct.pack("<IIIIII", command, size, fixed, 0, 0, 0)
    else:
        header = struct.pack("<III", command, size, fixed)
    return header + encoded + bytes(size - len(header) - len(encoded))


def _macho(
    path: Path,
    dependencies: list[str],
    *,
    rpaths: tuple[str, ...] = (),
    filetype: int = 2,
    install_name: str | None = None,
) -> None:
    commands = [
        *(_macho_command(0x8000001C, value, dylib=False) for value in rpaths),
        *(
            [_macho_command(0xD, install_name, dylib=True)]
            if install_name is not None
            else []
        ),
        *(_macho_command(0xC, value, dylib=True) for value in dependencies),
    ]
    header = struct.pack(
        "<IiiIIIII",
        0xFEEDFACF,
        0x100000C,
        0,
        filetype,
        len(commands),
        sum(map(len, commands)),
        0,
        0,
    )
    path.write_bytes(header + b"".join(commands))


def test_macho_runtime_closure_is_root_relative_and_transitive(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    executable = root / "bin" / "runtime"
    library = root / "lib" / "libprimary.dylib"
    leaf = root / "lib" / "libleaf.dylib"
    executable.parent.mkdir(parents=True)
    library.parent.mkdir()
    _macho(
        executable,
        ["@rpath/libprimary.dylib", "/usr/lib/libSystem.B.dylib"],
        rpaths=("@executable_path/../lib",),
    )
    _macho(library, ["@loader_path/libleaf.dylib"], filetype=6)
    _macho(leaf, [], filetype=6)

    closure = observe_macho_runtime_closure(
        executable,
        required_binary_paths=(library,),
        owned_native_roots=(root,),
    )

    assert {item["path"] for item in closure["files"]} == {
        "root:0/bin/runtime",
        "root:0/lib/libprimary.dylib",
        "root:0/lib/libleaf.dylib",
    }
    assert closure["identity_sha256"]
    assert "/usr/lib/libSystem.B.dylib" in closure["trusted_system_dependencies"]


def test_macho_runtime_closure_resolves_explicit_preloads_by_install_name(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    executable = root / "runtime"
    library = root / "lib" / "libprimary.dylib"
    leaf = library.with_name("libleaf.dylib")
    library.parent.mkdir(parents=True)
    _macho(executable, [])
    _macho(library, ["@rpath/libleaf.dylib"], filetype=6)
    _macho(
        leaf,
        [],
        filetype=6,
        install_name="@rpath/libleaf.dylib",
    )

    closure = observe_macho_runtime_closure(
        executable,
        required_binary_paths=(library,),
        preloaded_binary_paths=(leaf,),
        owned_native_roots=(root,),
    )

    assert {item["path"] for item in closure["files"]} == {
        "root:0/lib/libleaf.dylib",
        "root:0/lib/libprimary.dylib",
        "root:0/runtime",
    }
    assert closure["preloaded_images"] == [
        {
            "install_name": "@rpath/libleaf.dylib",
            "path": "root:0/lib/libleaf.dylib",
        }
    ]


def test_macho_runtime_closure_does_not_invent_an_rpath_fallback(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    executable = root / "runtime"
    library = root / "lib" / "libprimary.dylib"
    leaf = library.with_name("libleaf.dylib")
    library.parent.mkdir(parents=True)
    _macho(executable, [])
    _macho(library, ["@rpath/libleaf.dylib"], filetype=6)
    _macho(leaf, [], filetype=6, install_name="@rpath/libleaf.dylib")

    with pytest.raises(
        RuntimeIdentityError,
        match="macho-dependency-unresolved:@rpath/libleaf.dylib",
    ):
        observe_macho_runtime_closure(
            executable,
            required_binary_paths=(library,),
            owned_native_roots=(root,),
        )


def test_macho_runtime_closure_rejects_external_native_code(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    executable = root / "runtime"
    external = tmp_path / "external.dylib"
    root.mkdir()
    _macho(executable, [str(external)])
    _macho(external, [], filetype=6)

    with pytest.raises(
        RuntimeIdentityError,
        match="macho-dependency-outside-trust-root",
    ):
        observe_macho_runtime_closure(
            executable,
            owned_native_roots=(root,),
        )


def test_tree_manifest_detects_byte_drift_and_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    target = root / "module.py"
    target.write_text("before\n", encoding="utf-8")
    before = tree_manifest(root, reject_symlinks=True, require_root_owned=False)

    target.write_text("after\n", encoding="utf-8")
    after = tree_manifest(root, reject_symlinks=True, require_root_owned=False)

    assert before["manifest_sha256"] != after["manifest_sha256"]
    (root / "replacement").symlink_to(target)
    with pytest.raises(RuntimeIdentityError, match="runtime-tree-symlink"):
        tree_manifest(root, reject_symlinks=True, require_root_owned=False)


def test_runtime_file_reader_is_bounded_stable_and_never_follows_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact = tmp_path / "exact.bin"
    oversized = tmp_path / "oversized.bin"
    exact.write_bytes(b"12345678")
    oversized.write_bytes(b"123456789")
    monkeypatch.setattr(runtime_identity, "_MAX_RUNTIME_FILE_BYTES", 8)
    requested: list[int] = []
    real_read = runtime_identity.os.read

    def bounded_read(fd: int, count: int) -> bytes:
        requested.append(count)
        return real_read(fd, count)

    monkeypatch.setattr(runtime_identity.os, "read", bounded_read)
    assert regular_file_identity(exact)["bytes"] == 8
    assert requested and max(requested) <= 9
    requested.clear()
    with pytest.raises(RuntimeIdentityError, match="runtime-file-exceeds-reader-bound"):
        regular_file_identity(oversized)
    assert requested == []

    alias = tmp_path / "alias.bin"
    alias.symlink_to(exact)
    with pytest.raises(RuntimeIdentityError, match="runtime-file-unavailable"):
        regular_file_identity(alias)

    mutated = tmp_path / "mutated.bin"
    mutated.write_bytes(b"abcd")
    did_mutate = False

    def read_then_replace(fd: int, count: int) -> bytes:
        nonlocal did_mutate
        block = real_read(fd, count)
        if not did_mutate:
            did_mutate = True
            replacement = tmp_path / "replacement.bin"
            replacement.write_bytes(b"wxyz")
            replacement.replace(mutated)
        return block

    monkeypatch.setattr(runtime_identity.os, "read", read_then_replace)
    with pytest.raises(RuntimeIdentityError, match="runtime-file-replaced"):
        regular_file_identity(mutated)


def test_runtime_file_readers_fail_on_first_byte_after_opened_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_read = runtime_identity.os.read

    def assert_post_open_growth_fails(path: Path, observe: object) -> None:
        did_grow = False
        requested: list[int] = []

        def read_then_grow(fd: int, count: int) -> bytes:
            nonlocal did_grow
            requested.append(count)
            block = real_read(fd, count)
            if block and not did_grow:
                did_grow = True
                path.write_bytes(path.read_bytes() + b"!")
            return block

        monkeypatch.setattr(runtime_identity.os, "read", read_then_grow)
        with pytest.raises(RuntimeIdentityError, match="runtime-file-replaced"):
            observe()
        # The first probe past the opened size must terminate observation.
        assert requested[-1] == 1
        monkeypatch.setattr(runtime_identity.os, "read", real_read)

    descriptor_path = tmp_path / "descriptor-growth.bin"
    descriptor_path.write_bytes(b"abcd")

    def hash_descriptor() -> str:
        descriptor = os.open(descriptor_path, os.O_RDONLY)
        try:
            return runtime_identity._hash_file_descriptor(descriptor)
        finally:
            os.close(descriptor)

    assert_post_open_growth_fails(descriptor_path, hash_descriptor)

    stable_path = tmp_path / "stable-growth.bin"
    stable_path.write_bytes(b"abcd")
    assert_post_open_growth_fails(
        stable_path,
        lambda: regular_file_identity(stable_path),
    )


def test_tree_manifest_rejects_one_byte_over_runtime_file_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    (root / "exact.bin").write_bytes(b"12345678")
    monkeypatch.setattr(runtime_identity, "_MAX_RUNTIME_FILE_BYTES", 8)
    assert (
        tree_manifest(root, reject_symlinks=True, require_root_owned=False)[
            "file_count"
        ]
        == 1
    )
    (root / "over.bin").write_bytes(b"123456789")
    with pytest.raises(RuntimeIdentityError, match="runtime-file-exceeds-reader-bound"):
        tree_manifest(root, reject_symlinks=True, require_root_owned=False)


def test_python_runtime_identity_closes_owned_macho_graph_and_detects_drift(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "python-prefix"
    executable = prefix / "bin" / "python"
    stdlib = prefix / "lib" / "python"
    dynload = stdlib / "lib-dynload"
    dynload.mkdir(parents=True)
    executable.parent.mkdir(parents=True)
    owned = prefix / "lib" / "libowned.dylib"
    executable.write_text("python launcher\n", encoding="utf-8")
    (stdlib / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _macho(
        dynload / "extension.so",
        ["@rpath/libowned.dylib", "/usr/lib/libSystem.B.dylib"],
        rpaths=("@loader_path/../..",),
    )
    _macho(owned, [])

    identity = observe_python_runtime_identity(
        executable=executable,
        prefix=prefix,
        stdlib=stdlib,
    )

    assert any(item["path"] == str(owned) for item in identity["native_files"])
    assert "/usr/lib/libSystem.B.dylib" in identity["trusted_system_dependencies"]
    authenticate_python_runtime_identity(
        identity,
        executable=executable,
        prefix=prefix,
        stdlib=stdlib,
    )
    replacement = prefix / "lib" / "replacement.dylib"
    replacement.write_bytes(owned.read_bytes() + b"drift")
    replacement.replace(owned)
    with pytest.raises(RuntimeIdentityError, match="python-runtime-identity-mismatch"):
        authenticate_python_runtime_identity(
            identity,
            executable=executable,
            prefix=prefix,
            stdlib=stdlib,
        )


def _python_runtime_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    prefix = tmp_path / "python-prefix"
    executable = prefix / "bin" / "python"
    stdlib = prefix / "lib" / "python"
    (stdlib / "lib-dynload").mkdir(parents=True)
    executable.parent.mkdir(parents=True)
    executable.write_text("python launcher\n", encoding="utf-8")
    (stdlib / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    return prefix, executable, stdlib


def test_python_runtime_identity_inherits_declared_rpath_in_order(
    tmp_path: Path,
) -> None:
    prefix, executable, stdlib = _python_runtime_layout(tmp_path)
    extension = prefix / "site-packages" / "mlx" / "core.so"
    primary = extension.parent / "first" / "libmlx.dylib"
    secondary = extension.parent / "second" / "libmlx.dylib"
    jaccl = extension.parent / "first" / "libjaccl.dylib"
    primary.parent.mkdir(parents=True)
    secondary.parent.mkdir(parents=True)
    _macho(
        extension,
        ["@rpath/libmlx.dylib"],
        rpaths=("@loader_path/first", "@loader_path/second"),
    )
    _macho(primary, ["@rpath/libjaccl.dylib"])
    _macho(secondary, [])
    _macho(jaccl, [])

    identity = observe_python_runtime_identity(
        executable=executable,
        prefix=prefix,
        stdlib=stdlib,
        additional_binary_paths=[extension],
        owned_native_roots=[prefix],
    )

    edges = {
        (item["loader"], item["declared"]): item["resolved"]
        for item in identity["native_edges"]
    }
    assert edges[(str(extension), "@rpath/libmlx.dylib")] == str(primary)
    assert edges[(str(primary), "@rpath/libjaccl.dylib")] == str(jaccl)
    assert {item["path"] for item in identity["native_files"]} == {
        str(extension),
        str(primary),
        str(jaccl),
    }


def test_python_runtime_identity_prefers_child_rpath_before_inherited_frame(
    tmp_path: Path,
) -> None:
    prefix, executable, stdlib = _python_runtime_layout(tmp_path)
    native = prefix / "native"
    parents = native / "parents"
    local = parents / "local"
    parents.mkdir(parents=True)
    local.mkdir()
    extension = native / "extension.so"
    parent = parents / "parent.dylib"
    local_leaf = local / "libleaf.dylib"
    inherited_leaf = parents / "libleaf.dylib"
    _macho(
        extension,
        ["@rpath/parent.dylib"],
        rpaths=("@loader_path/parents",),
        filetype=8,
    )
    _macho(
        parent,
        ["@rpath/libleaf.dylib"],
        rpaths=("@loader_path/local",),
        filetype=6,
    )
    _macho(local_leaf, [], filetype=6)
    _macho(inherited_leaf, [], filetype=6)

    identity = observe_python_runtime_identity(
        executable=executable,
        prefix=prefix,
        stdlib=stdlib,
        additional_binary_paths=[extension],
    )

    edges = {
        (item["loader"], item["declared"]): item["resolved"]
        for item in identity["native_edges"]
    }
    assert edges[(str(parent), "@rpath/libleaf.dylib")] == str(local_leaf)


def test_python_runtime_identity_requires_every_required_rpath_context(
    tmp_path: Path,
) -> None:
    prefix, executable, stdlib = _python_runtime_layout(tmp_path)
    native = prefix / "native"
    libraries = native / "libraries"
    native.mkdir()
    libraries.mkdir()
    resolving_root = native / "a-resolving.bundle"
    failing_root = native / "z-failing.bundle"
    loader = native / "loader.dylib"
    leaf = libraries / "libleaf.dylib"
    _macho(
        resolving_root,
        ["@loader_path/loader.dylib"],
        rpaths=("@loader_path/libraries",),
        filetype=8,
    )
    _macho(
        failing_root,
        ["@loader_path/loader.dylib"],
        filetype=8,
    )
    _macho(loader, ["@rpath/libleaf.dylib"], filetype=6)
    _macho(leaf, [], filetype=6)

    with pytest.raises(
        RuntimeIdentityError,
        match="macho-dependency-unresolved:@rpath/libleaf.dylib",
    ):
        observe_python_runtime_identity(
            executable=executable,
            prefix=prefix,
            stdlib=stdlib,
            additional_binary_paths=[resolving_root, failing_root],
        )


def test_python_runtime_identity_hashes_nonrequired_dylib_root_without_dyld_claim(
    tmp_path: Path,
) -> None:
    prefix, executable, stdlib = _python_runtime_layout(tmp_path)
    native = prefix / "native"
    native.mkdir()
    standalone = native / "standalone.dylib"
    _macho(standalone, ["@rpath/not-required.dylib"], filetype=6)

    identity = observe_python_runtime_identity(
        executable=executable,
        prefix=prefix,
        stdlib=stdlib,
        additional_binary_paths=[standalone],
    )

    assert identity["native_input_files"][-1]["path"] == str(standalone)
    assert any(item["path"] == str(standalone) for item in identity["native_files"])


def test_python_runtime_identity_rejects_symlinked_macho_candidates(
    tmp_path: Path,
) -> None:
    prefix, executable, stdlib = _python_runtime_layout(tmp_path)
    native = prefix / "native"
    native.mkdir()
    extension = native / "extension.bundle"
    target = native / "target.dylib"
    _macho(
        extension,
        ["@rpath/libtarget.dylib"],
        rpaths=("@loader_path",),
        filetype=8,
    )
    _macho(target, [], filetype=6)
    (native / "libtarget.dylib").symlink_to(target)

    with pytest.raises(RuntimeIdentityError, match="macho-dependency-symlink"):
        observe_python_runtime_identity(
            executable=executable,
            prefix=prefix,
            stdlib=stdlib,
            additional_binary_paths=[extension],
        )

    (native / "libtarget.dylib").unlink()
    redirected = native / "redirected"
    redirected.mkdir()
    retargeted = redirected / "libtarget.dylib"
    _macho(retargeted, [], filetype=6)
    (native / "redirect").symlink_to(redirected, target_is_directory=True)
    _macho(
        extension,
        ["@rpath/libtarget.dylib"],
        rpaths=("@loader_path/redirect",),
        filetype=8,
    )

    with pytest.raises(RuntimeIdentityError, match="macho-dependency-symlink"):
        observe_python_runtime_identity(
            executable=executable,
            prefix=prefix,
            stdlib=stdlib,
            additional_binary_paths=[extension],
        )


def test_python_runtime_identity_binds_exact_native_input_file_metadata(
    tmp_path: Path,
) -> None:
    prefix, executable, stdlib = _python_runtime_layout(tmp_path)
    native = prefix / "native"
    native.mkdir()
    frozen = native / "frozen.dylib"
    _macho(frozen, [], filetype=6)

    identity = observe_python_runtime_identity(
        executable=executable,
        prefix=prefix,
        stdlib=stdlib,
        additional_binary_paths=[frozen],
    )
    frozen_input = next(
        item for item in identity["native_input_files"] if item["path"] == str(frozen)
    )
    assert frozen_input["links"] == 1
    assert {
        "device",
        "inode",
        "links",
        "mtime_ns",
        "ctime_ns",
    } <= frozen_input.keys()

    conda_source = tmp_path / "conda-source.dylib"
    conda_source.write_bytes(frozen.read_bytes())
    frozen.unlink()
    os.link(conda_source, frozen)

    with pytest.raises(RuntimeIdentityError, match="python-runtime-identity-mismatch"):
        authenticate_python_runtime_identity(
            identity,
            executable=executable,
            prefix=prefix,
            stdlib=stdlib,
            additional_binary_paths=[frozen],
        )


def test_python_runtime_identity_bounds_rpath_cycles_and_dedupes_diamonds(
    tmp_path: Path,
) -> None:
    prefix, executable, stdlib = _python_runtime_layout(tmp_path)
    native = prefix / "native"
    native.mkdir()
    extension = native / "extension.so"
    left = native / "left.dylib"
    right = native / "right.dylib"
    shared = native / "shared.dylib"
    _macho(
        extension,
        ["@rpath/left.dylib", "@rpath/right.dylib"],
        rpaths=("@loader_path",),
    )
    _macho(left, ["@rpath/shared.dylib"])
    _macho(right, ["@rpath/shared.dylib"])
    _macho(shared, ["@rpath/left.dylib"])

    identity = observe_python_runtime_identity(
        executable=executable,
        prefix=prefix,
        stdlib=stdlib,
        additional_binary_paths=[extension],
    )

    assert {item["path"] for item in identity["native_files"]} == {
        str(extension),
        str(left),
        str(right),
        str(shared),
    }
    assert len(identity["native_edges"]) == 5
    assert len({tuple(sorted(item.items())) for item in identity["native_edges"]}) == 5


def test_python_runtime_identity_uses_finite_semantic_contexts_not_ancestry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix, executable, stdlib = _python_runtime_layout(tmp_path)
    native = prefix / "native"
    redirect = native / "redirect"
    native.mkdir()
    redirect.mkdir()
    root = native / "root.bundle"
    a = native / "a.dylib"
    b = native / "b.dylib"
    left = native / "left.dylib"
    right = native / "right.dylib"
    first = native / "first.dylib"
    second = native / "second.dylib"
    direct_child = native / "child.dylib"
    redirected_child = redirect / "child.dylib"
    terminal = native / "terminal.dylib"
    _macho(
        root,
        ["@rpath/a.dylib", "@rpath/left.dylib", "@rpath/right.dylib"],
        rpaths=("@loader_path",),
        filetype=8,
    )
    _macho(a, ["@rpath/b.dylib", "@rpath/child.dylib"], filetype=6)
    _macho(
        b,
        ["@loader_path/a.dylib"],
        rpaths=("@loader_path/redirect",),
        filetype=6,
    )
    _macho(left, ["@rpath/first.dylib", "@rpath/second.dylib"], filetype=6)
    _macho(right, ["@rpath/first.dylib", "@rpath/second.dylib"], filetype=6)
    _macho(first, ["@rpath/terminal.dylib"], filetype=6)
    _macho(second, ["@rpath/terminal.dylib"], filetype=6)
    _macho(direct_child, [], filetype=6)
    _macho(redirected_child, [], filetype=6)
    _macho(terminal, [], filetype=6)
    # The four diamond paths to terminal and the A -> B -> A cycle would
    # create ancestry-shaped states beyond this cap.  The re-entered A has a
    # distinct inherited frame and must still reach redirect/child.dylib.
    monkeypatch.setattr(runtime_identity, "_MAX_MACHO_RESOLUTION_CONTEXTS", 14)

    identity = observe_python_runtime_identity(
        executable=executable,
        prefix=prefix,
        stdlib=stdlib,
        additional_binary_paths=[root],
    )

    assert str(redirected_child) in {item["path"] for item in identity["native_files"]}


def test_owned_native_roots_do_not_expand_python_prefix_admission(
    tmp_path: Path,
) -> None:
    base_prefix, base_executable, stdlib = _python_runtime_layout(tmp_path / "base")
    venv = tmp_path / "venv"
    executable = venv / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_text("python launcher\n", encoding="utf-8")

    with pytest.raises(RuntimeIdentityError, match="python-executable-outside-prefix"):
        observe_python_runtime_identity(
            executable=executable,
            prefix=base_prefix,
            stdlib=stdlib,
            owned_native_roots=[base_prefix, venv],
        )

    outside_prefix, _, outside_stdlib = _python_runtime_layout(tmp_path / "outside")
    with pytest.raises(RuntimeIdentityError, match="python-stdlib-outside-prefix"):
        observe_python_runtime_identity(
            executable=base_executable,
            prefix=base_prefix,
            stdlib=outside_stdlib,
            owned_native_roots=[base_prefix, outside_prefix],
        )


def _fake_go_runtime(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    goroot = tmp_path / "go"
    gotooldir = goroot / "pkg" / "tool" / "darwin_arm64"
    executable = goroot / "bin" / "go"
    gotooldir.mkdir(parents=True)
    executable.parent.mkdir(parents=True)
    (gotooldir / "compile").write_bytes(b"compiler-v1")
    go_env = {
        "CC": "clang",
        "CGO_ENABLED": "0",
        "CXX": "clang++",
        "GOARCH": "arm64",
        "GOOS": "darwin",
        "GOROOT": str(goroot),
        "GOTOOLDIR": str(gotooldir),
    }
    executable.write_text(
        "#!/usr/bin/python3\n"
        f"print({json.dumps(json.dumps(go_env, sort_keys=True))})\n",
        encoding="utf-8",
    )
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "TMPDIR": str(tmp_path / "tmp"),
        "CGO_ENABLED": "0",
        "GOENV": "off",
        "GOWORK": "off",
        "GOTOOLCHAIN": "local",
        "GOPROXY": "off",
        "GOSUMDB": "off",
    }
    return executable, environment


def test_go_runtime_identity_binds_full_root_and_rejects_cgo_or_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable, environment = _fake_go_runtime(tmp_path)
    monkeypatch.setattr(
        runtime_identity,
        "_require_safe_root_owned_path",
        lambda path, *, directory: Path(path).lstat(),
    )
    real_manifest = runtime_identity.tree_manifest
    monkeypatch.setattr(
        runtime_identity,
        "tree_manifest",
        lambda root, *, reject_symlinks, require_root_owned, excluded_top_level_names=(): (
            real_manifest(
                root,
                reject_symlinks=reject_symlinks,
                require_root_owned=False,
                excluded_top_level_names=excluded_top_level_names,
            )
        ),
    )

    identity = observe_go_runtime_identity(executable, environment)
    assert identity["go_env"]["CGO_ENABLED"] == "0"
    assert identity["gotooldir_manifest"]["file_count"] == 1
    authenticate_go_runtime_identity(identity, executable, environment)

    compile_tool = Path(identity["go_env"]["GOTOOLDIR"]) / "compile"
    compile_tool.write_bytes(b"compiler-v2")
    with pytest.raises(RuntimeIdentityError, match="go-runtime-identity-mismatch"):
        authenticate_go_runtime_identity(identity, executable, environment)
    with pytest.raises(RuntimeIdentityError, match="go-runtime-cgo-not-disabled"):
        observe_go_runtime_identity(executable, {**environment, "CGO_ENABLED": "1"})


def test_go_runtime_identity_rejects_goroot_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable, environment = _fake_go_runtime(tmp_path)
    root = executable.parent.parent
    real_root = tmp_path / "real-go"
    root.rename(real_root)
    root.symlink_to(real_root, target_is_directory=True)
    executable = root / "bin" / "go"
    monkeypatch.setattr(
        runtime_identity,
        "_require_safe_root_owned_path",
        lambda path, *, directory: Path(path).lstat(),
    )
    real_manifest = runtime_identity.tree_manifest
    monkeypatch.setattr(
        runtime_identity,
        "tree_manifest",
        lambda root, *, reject_symlinks, require_root_owned, excluded_top_level_names=(): (
            real_manifest(
                root,
                reject_symlinks=reject_symlinks,
                require_root_owned=False,
                excluded_top_level_names=excluded_top_level_names,
            )
        ),
    )
    with pytest.raises(RuntimeIdentityError, match="runtime-tree-invalid"):
        observe_go_runtime_identity(executable, environment)


def test_git_identity_uses_direct_clt_binary_and_closed_environment() -> None:
    executable = trusted_clt_git_executable()
    environment = trusted_git_subprocess_environment()

    assert executable == Path("/Library/Developer/CommandLineTools/usr/bin/git")
    assert executable != Path("/usr/bin/git")
    assert environment["GIT_EXEC_PATH"] == (
        "/Library/Developer/CommandLineTools/usr/libexec/git-core"
    )
    assert "SSH_AUTH_SOCK" not in environment
    assert "PYTHONPATH" not in environment
    assert set(environment) == {
        "DEVELOPER_DIR",
        "GIT_ATTR_NOSYSTEM",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_KEY_1",
        "GIT_CONFIG_KEY_2",
        "GIT_CONFIG_KEY_3",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_VALUE_0",
        "GIT_CONFIG_VALUE_1",
        "GIT_CONFIG_VALUE_2",
        "GIT_CONFIG_VALUE_3",
        "GIT_EXEC_PATH",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_OPTIONAL_LOCKS",
        "GIT_TERMINAL_PROMPT",
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "TMPDIR",
        "XDG_CONFIG_HOME",
    }
    identity = observe_git_runtime_identity()
    assert identity["executable"]["path"] == str(executable)
    assert identity["exec_path_symlink_targets"]
    authenticate_git_runtime_identity(identity)


def test_git_required_builtins_cover_every_fixed_workspace_git_verb() -> None:
    workspace_source = Path(workspace_module.__file__).read_text(encoding="utf-8")
    syntax = ast.parse(workspace_source)
    fixed_verbs: set[str] = set()
    for node in ast.walk(syntax):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"_git_text", "_git_bytes"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            fixed_verbs.add(node.args[1].value)
        if not isinstance(node, ast.List) or not node.elts:
            continue
        values = node.elts
        if not isinstance(values[0], ast.Constant) or values[0].value != "git":
            continue
        verb_index = (
            3
            if len(values) > 1
            and isinstance(values[1], ast.Constant)
            and values[1].value == "-C"
            else 1
        )
        if (
            len(values) > verb_index
            and isinstance(values[verb_index], ast.Constant)
            and isinstance(values[verb_index].value, str)
        ):
            fixed_verbs.add(values[verb_index].value)

    assert fixed_verbs <= runtime_identity._GIT_REQUIRED_BUILTINS
