from __future__ import annotations

import os
import json
from pathlib import Path

import pytest

from graph_model.integrations.mlxcelerator_runtime import (
    MLXCELERATOR_LLAMA_ADMISSION_FORMAT,
    MLXCELERATOR_LLAMA_GENERATION_FORMAT,
    MLXCELERATOR_RUNTIME_PROBE_FORMAT,
    MlxceleratorRuntimeError,
    _model_file_identity,
    admit_mlxcelerator_llama_model,
    generate_mlxcelerator_llama_text,
    probe_mlxcelerator_runtime,
)
from graph_model.runtime_identity import (
    canonical_sha256,
    regular_file_identity,
    tree_manifest,
)


VALID_OUTPUT = """runtime=mlxcelerator-mlx-runtime/0.1.0/default
chip=Apple M5 Max
model=Mac17,6
memory_bytes=137438953472
os_version=27.0
os_build=26A5421a
admitted=true
admission_reason=qualified
mlx_version=0.32.2
mlx_gpu_smoke=true
mlx_gpu_smoke_error=none
core_ai_architecture=h17c
core_ai_units=cpu,gpu,ane
nax_gpu_path=false
lane_gpu=QualifiedOnly
lane_cpu=QualifiedOnly
lane_ane=QualifiedOnly
"""


def _probe_script(tmp_path: Path, output: str) -> tuple[Path, Path]:
    executable = tmp_path / "mlxcelerator"
    escaped = output.replace("'", "'\"'\"'")
    executable.write_text(
        f"#!/bin/sh\nprintf '%s' '{escaped}'\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    library_root = tmp_path / "mlx"
    library_root.mkdir()
    library = library_root / "libmlxc.dylib"
    library.write_bytes(b"test mlx-c")
    return executable, library


def _authorized_probe(executable: Path, library: Path) -> dict[str, object]:
    return probe_mlxcelerator_runtime(
        executable,
        library,
        expected_executable_sha256=regular_file_identity(executable)["sha256"],
        expected_mlx_c_sha256=regular_file_identity(library)["sha256"],
        expected_library_manifest_sha256=tree_manifest(
            library.parent,
            reject_symlinks=True,
            require_root_owned=False,
        )["manifest_sha256"],
    )


def test_probe_authenticates_runtime_and_capabilities(
    tmp_path: Path,
) -> None:
    executable, library = _probe_script(tmp_path, VALID_OUTPUT)

    receipt = _authorized_probe(executable, library)

    assert receipt["format"] == MLXCELERATOR_RUNTIME_PROBE_FORMAT
    assert receipt["capabilities"]["chip"] == "Apple M5 Max"
    assert receipt["capabilities"]["memory_bytes"] == 137438953472
    assert receipt["capabilities"]["lanes"] == {
        "gpu": "QualifiedOnly",
        "cpu": "QualifiedOnly",
        "ane": "QualifiedOnly",
    }
    assert receipt["executable"]["sha256"]
    assert receipt["mlx_library_manifest"]["manifest_sha256"]
    assert receipt["receipt_sha256"]
    assert receipt["capabilities"]["mlx_gpu_smoke"] is True
    assert receipt["capabilities"]["core_ai_units"] == ["cpu", "gpu", "ane"]
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == canonical_sha256(unsigned)


def _llama_script(tmp_path: Path, output: str) -> Path:
    executable = tmp_path / "mlxcelerator"
    escaped = output.replace("'", "'\"'\"'")
    executable.write_text(
        f"#!/bin/sh\nprintf '%s' '{escaped}'\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable


def _sdpa_llama_script(tmp_path: Path, output: str) -> Path:
    executable = tmp_path / "mlxcelerator"
    escaped = output.replace("'", "'\"'\"'")
    executable.write_text(
        f"#!/bin/sh\n"
        f"if [ \"$MLXC_USE_SDPA\" != \"1\" ]; then exit 91; fi\n"
        f"printf '%s' '{escaped}'\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable


def _rope_llama_script(tmp_path: Path, output: str) -> Path:
    executable = tmp_path / "mlxcelerator"
    escaped = output.replace("'", "'\"'\"'")
    executable.write_text(
        f"#!/bin/sh\n"
        f"if [ \"$MLXC_USE_ROPE\" != \"1\" ]; then exit 92; fi\n"
        f"printf '%s' '{escaped}'\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable


def test_llama_admission_authenticates_model_and_configuration(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"validated llama fixture")
    model_identity = regular_file_identity(model)
    output = json.dumps(
        {
            "schema_version": 1,
            "path": "/snapshot/model.gguf",
            "file_size": model_identity["bytes"],
            "digest_sha256": model_identity["sha256"],
            "context_length": 16,
            "embedding_length": 4,
            "block_count": 1,
            "head_count": 2,
            "head_count_kv": 1,
            "feed_forward_length": 8,
            "vocab_size": 8,
            "rms_norm_epsilon": 1e-5,
            "rope_freq_base": 10000.0,
            "tensor_count": 12,
        },
        separators=(",", ":"),
    )
    executable = _llama_script(tmp_path, output)

    receipt = admit_mlxcelerator_llama_model(
        executable,
        model,
        expected_executable_sha256=regular_file_identity(executable)["sha256"],
        expected_model_sha256=model_identity["sha256"],
    )

    assert receipt["format"] == MLXCELERATOR_LLAMA_ADMISSION_FORMAT
    assert receipt["admission"]["embedding_length"] == 4
    assert receipt["admission"]["head_count_kv"] == 1
    assert receipt["model"]["sha256"] == model_identity["sha256"]
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == canonical_sha256(unsigned)


def test_llama_admission_rejects_digest_mismatch(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"validated llama fixture")
    executable = _llama_script(tmp_path, "{}")

    with pytest.raises(
        MlxceleratorRuntimeError,
        match="mlxcelerator-llama-model-unauthorized",
    ):
        admit_mlxcelerator_llama_model(
            executable,
            model,
            expected_executable_sha256=regular_file_identity(executable)["sha256"],
            expected_model_sha256="0" * 64,
        )


def test_llama_generation_authenticates_native_backend_and_output(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"validated llama fixture")
    model_identity = regular_file_identity(model)
    library_root = tmp_path / "mlx"
    library_root.mkdir()
    library = library_root / "libmlxc.dylib"
    library.write_bytes(b"test mlx-c")
    output = json.dumps(
        {
            "schema_version": 1,
            "backend": "mlx-gpu-linear-host-attention-v1",
            "path": "/snapshot/model.gguf",
            "file_size": model_identity["bytes"],
            "digest_sha256": model_identity["sha256"],
            "prompt": "hello",
            "max_new_tokens": 2,
            "generated_text": "ok",
        },
        separators=(",", ":"),
    )
    executable = _llama_script(tmp_path, output)
    receipt = generate_mlxcelerator_llama_text(
        executable,
        model,
        library,
        "hello",
        2,
        expected_executable_sha256=regular_file_identity(executable)["sha256"],
        expected_model_sha256=model_identity["sha256"],
        expected_mlx_c_sha256=regular_file_identity(library)["sha256"],
        expected_library_manifest_sha256=tree_manifest(
            library.parent,
            reject_symlinks=True,
            require_root_owned=False,
        )["manifest_sha256"],
    )

    assert receipt["format"] == MLXCELERATOR_LLAMA_GENERATION_FORMAT
    assert receipt["generation"] == {
        "schema_version": 1,
        "backend": "mlx-gpu-linear-host-attention-v1",
        "path": "/snapshot/model.gguf",
        "file_size": model_identity["bytes"],
        "digest_sha256": model_identity["sha256"],
        "prompt": "hello",
        "max_new_tokens": 2,
        "generated_text": "ok",
    }
    assert receipt["mlx_c_library"]["identity"]["sha256"] == regular_file_identity(
        library
    )["sha256"]
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == canonical_sha256(unsigned)


def test_llama_generation_sdpa_requires_and_passes_explicit_mode_evidence(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"validated llama fixture")
    model_identity = regular_file_identity(model)
    library_root = tmp_path / "mlx"
    library_root.mkdir()
    library = library_root / "libmlxc.dylib"
    library.write_bytes(b"test mlx-c")
    output = json.dumps(
        {
            "schema_version": 1,
            "backend": "mlx-gpu-linear-sdpa-host-kv-v1",
            "attention_mode": "sdpa",
            "path": "/snapshot/model.gguf",
            "file_size": model_identity["bytes"],
            "digest_sha256": model_identity["sha256"],
            "prompt": "hello",
            "max_new_tokens": 2,
            "generated_text": "ok",
        },
        separators=(",", ":"),
    )
    executable = _sdpa_llama_script(tmp_path, output)

    receipt = generate_mlxcelerator_llama_text(
        executable,
        model,
        library,
        "hello",
        2,
        expected_executable_sha256=regular_file_identity(executable)["sha256"],
        expected_model_sha256=model_identity["sha256"],
        expected_mlx_c_sha256=regular_file_identity(library)["sha256"],
        expected_library_manifest_sha256=tree_manifest(
            library.parent,
            reject_symlinks=True,
            require_root_owned=False,
        )["manifest_sha256"],
        attention_mode="sdpa",
    )

    assert receipt["generation"]["attention_mode"] == "sdpa"
    assert receipt["generation"]["backend"] == "mlx-gpu-linear-sdpa-host-kv-v1"


def test_llama_generation_sdpa_rejects_host_backend_evidence(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"validated llama fixture")
    model_identity = regular_file_identity(model)
    library_root = tmp_path / "mlx"
    library_root.mkdir()
    library = library_root / "libmlxc.dylib"
    library.write_bytes(b"test mlx-c")
    output = json.dumps(
        {
            "schema_version": 1,
            "backend": "mlx-gpu-linear-host-attention-v1",
            "path": "/snapshot/model.gguf",
            "file_size": model_identity["bytes"],
            "digest_sha256": model_identity["sha256"],
            "prompt": "hello",
            "max_new_tokens": 2,
            "generated_text": "ok",
        },
        separators=(",", ":"),
    )
    executable = _llama_script(tmp_path, output)

    with pytest.raises(
        MlxceleratorRuntimeError,
        match="mlxcelerator-llama-generation-schema-invalid",
    ):
        generate_mlxcelerator_llama_text(
            executable,
            model,
            library,
            "hello",
            2,
            expected_executable_sha256=regular_file_identity(executable)["sha256"],
            expected_model_sha256=model_identity["sha256"],
            expected_mlx_c_sha256=regular_file_identity(library)["sha256"],
            expected_library_manifest_sha256=tree_manifest(
                library.parent,
                reject_symlinks=True,
                require_root_owned=False,
            )["manifest_sha256"],
            attention_mode="sdpa",
        )


def test_llama_generation_native_rope_requires_and_passes_explicit_mode_evidence(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"validated llama fixture")
    model_identity = regular_file_identity(model)
    library_root = tmp_path / "mlx"
    library_root.mkdir()
    library = library_root / "libmlxc.dylib"
    library.write_bytes(b"test mlx-c")
    output = json.dumps(
        {
            "schema_version": 1,
            "backend": "mlx-gpu-linear-host-attention-v1",
            "rope_mode": "mlx-fast",
            "path": "/snapshot/model.gguf",
            "file_size": model_identity["bytes"],
            "digest_sha256": model_identity["sha256"],
            "prompt": "hello",
            "max_new_tokens": 2,
            "generated_text": "ok",
        },
        separators=(",", ":"),
    )
    executable = _rope_llama_script(tmp_path, output)

    receipt = generate_mlxcelerator_llama_text(
        executable,
        model,
        library,
        "hello",
        2,
        expected_executable_sha256=regular_file_identity(executable)["sha256"],
        expected_model_sha256=model_identity["sha256"],
        expected_mlx_c_sha256=regular_file_identity(library)["sha256"],
        expected_library_manifest_sha256=tree_manifest(
            library.parent,
            reject_symlinks=True,
            require_root_owned=False,
        )["manifest_sha256"],
        rope_mode="mlx-fast",
    )

    assert receipt["generation"]["rope_mode"] == "mlx-fast"
    assert receipt["generation"]["backend"] == "mlx-gpu-linear-host-attention-v1"


def test_model_identity_streams_past_native_library_cap(tmp_path: Path) -> None:
    model = tmp_path / "large-model.gguf"
    with model.open("wb") as handle:
        handle.truncate(256 * 1024 * 1024 + 1)

    identity = _model_file_identity(model)

    assert identity["bytes"] == 256 * 1024 * 1024 + 1
    assert len(identity["sha256"]) == 64


def test_receipt_binds_the_selected_mlx_c_file(tmp_path: Path) -> None:
    executable, library = _probe_script(tmp_path, VALID_OUTPUT)
    alternate = library.with_name("libmlxc-alternate.dylib")
    alternate.write_bytes(b"different mlx-c")
    manifest_sha256 = tree_manifest(
        library.parent,
        reject_symlinks=True,
        require_root_owned=False,
    )["manifest_sha256"]

    receipts = [
        probe_mlxcelerator_runtime(
            executable,
            selected,
            expected_executable_sha256=regular_file_identity(executable)["sha256"],
            expected_mlx_c_sha256=regular_file_identity(selected)["sha256"],
            expected_library_manifest_sha256=manifest_sha256,
        )
        for selected in (library, alternate)
    ]

    assert receipts[0]["mlx_c_library"]["relative_path"] == "libmlxc.dylib"
    assert receipts[1]["mlx_c_library"]["relative_path"] == "libmlxc-alternate.dylib"
    assert receipts[0]["receipt_sha256"] != receipts[1]["receipt_sha256"]


def test_probe_rejects_inconsistent_admission(
    tmp_path: Path,
) -> None:
    invalid = VALID_OUTPUT.replace(
        "admission_reason=qualified", "admission_reason=below_m5_max_floor"
    )
    executable, library = _probe_script(tmp_path, invalid)

    with pytest.raises(
        MlxceleratorRuntimeError,
        match="mlxcelerator-runtime-admission-inconsistent",
    ):
        _authorized_probe(executable, library)


def test_probe_accepts_unavailable_lanes_below_hardware_floor(tmp_path: Path) -> None:
    output = (
        VALID_OUTPUT.replace("admitted=true", "admitted=false")
        .replace("admission_reason=qualified", "admission_reason=below_m5_max_floor")
        .replace("lane_gpu=QualifiedOnly", "lane_gpu=Unavailable")
        .replace("lane_cpu=QualifiedOnly", "lane_cpu=Unavailable")
        .replace("lane_ane=QualifiedOnly", "lane_ane=Unavailable")
    )
    executable, library = _probe_script(tmp_path, output)

    receipt = _authorized_probe(executable, library)

    assert receipt["capabilities"]["admitted"] is False
    assert set(receipt["capabilities"]["lanes"].values()) == {"Unavailable"}


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        (
            "memory_bytes=137438953472",
            f"memory_bytes={1 << 64}",
            "mlxcelerator-runtime-memory-invalid",
        ),
        (
            "admission_reason=qualified",
            "admission_reason=future_reason",
            "mlxcelerator-runtime-admission-reason-invalid",
        ),
        (
            "core_ai_units=cpu,gpu,ane",
            "core_ai_units=gpu,cpu,ane",
            "mlxcelerator-runtime-core-ai-units-invalid",
        ),
        (
            "lane_gpu=QualifiedOnly",
            "lane_gpu=Active",
            "mlxcelerator-runtime-lane-invalid:lane_gpu",
        ),
        (
            "lane_cpu=QualifiedOnly",
            "lane_cpu=Unavailable",
            "mlxcelerator-runtime-cpu-lane-inconsistent",
        ),
        (
            "lane_ane=QualifiedOnly",
            "lane_ane=Active",
            "mlxcelerator-runtime-lane-invalid:lane_ane",
        ),
        (
            "nax_gpu_path=false",
            "nax_gpu_path=true",
            "mlxcelerator-runtime-nax-unimplemented",
        ),
        (
            "mlx_version=0.32.2",
            "mlx_version=",
            "mlxcelerator-runtime-mlx-unprobed",
        ),
        (
            "mlx_gpu_smoke_error=none",
            "mlx_gpu_smoke_error=",
            "mlxcelerator-runtime-gpu-smoke-inconsistent",
        ),
    ],
)
def test_probe_rejects_impossible_capability_tuples(
    tmp_path: Path, old: str, new: str, error: str
) -> None:
    executable, library = _probe_script(tmp_path, VALID_OUTPUT.replace(old, new))

    with pytest.raises(MlxceleratorRuntimeError, match=error):
        _authorized_probe(executable, library)


def test_probe_rejects_empty_failed_gpu_smoke_error(tmp_path: Path) -> None:
    output = (
        VALID_OUTPUT.replace("mlx_gpu_smoke=true", "mlx_gpu_smoke=false")
        .replace("mlx_gpu_smoke_error=none", "mlx_gpu_smoke_error=")
    )
    executable, library = _probe_script(tmp_path, output)

    with pytest.raises(
        MlxceleratorRuntimeError,
        match="mlxcelerator-runtime-gpu-smoke-inconsistent",
    ):
        _authorized_probe(executable, library)


def test_probe_rejects_duplicate_capability_keys(
    tmp_path: Path,
) -> None:
    invalid = VALID_OUTPUT + "chip=Apple M6 Max\n"
    executable, library = _probe_script(tmp_path, invalid)

    with pytest.raises(
        MlxceleratorRuntimeError,
        match="mlxcelerator-runtime-probe-key-duplicate:chip",
    ):
        _authorized_probe(executable, library)


def test_probe_rejects_unapproved_executable(tmp_path: Path) -> None:
    executable, library = _probe_script(tmp_path, VALID_OUTPUT)
    library_manifest = tree_manifest(
        library.parent,
        reject_symlinks=True,
        require_root_owned=False,
    )

    with pytest.raises(
        MlxceleratorRuntimeError,
        match="mlxcelerator-runtime-executable-unauthorized",
    ):
        probe_mlxcelerator_runtime(
            executable,
            library,
            expected_executable_sha256="0" * 64,
            expected_mlx_c_sha256=regular_file_identity(library)["sha256"],
            expected_library_manifest_sha256=library_manifest["manifest_sha256"],
        )


def test_probe_kills_output_overflow(tmp_path: Path) -> None:
    output = VALID_OUTPUT + ("x" * (64 * 1024))
    executable, library = _probe_script(tmp_path, output)

    with pytest.raises(
        MlxceleratorRuntimeError,
        match="mlxcelerator-runtime-probe-output-exceeded",
    ):
        _authorized_probe(executable, library)


def test_probe_kills_timed_out_process_group(tmp_path: Path) -> None:
    executable, library = _probe_script(tmp_path, VALID_OUTPUT)
    executable.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")

    with pytest.raises(
        MlxceleratorRuntimeError,
        match="mlxcelerator-runtime-probe-timeout",
    ):
        probe_mlxcelerator_runtime(
            executable,
            library,
            expected_executable_sha256=regular_file_identity(executable)["sha256"],
            expected_mlx_c_sha256=regular_file_identity(library)["sha256"],
            expected_library_manifest_sha256=tree_manifest(
                library.parent,
                reject_symlinks=True,
                require_root_owned=False,
            )["manifest_sha256"],
            timeout_seconds=0.05,
        )


def test_probe_rejects_and_kills_lingering_descendant(tmp_path: Path) -> None:
    executable, library = _probe_script(tmp_path, VALID_OUTPUT)
    pid_file = tmp_path / "descendant.pid"
    escaped = VALID_OUTPUT.replace("'", "'\"'\"'")
    executable.write_text(
        f"#!/bin/sh\nsleep 30 >/dev/null 2>&1 &\nprintf '%s' \"$!\" > '{pid_file}'\nprintf '%s' '{escaped}'\n",
        encoding="utf-8",
    )

    with pytest.raises(
        MlxceleratorRuntimeError,
        match="mlxcelerator-runtime-probe-descendant-process",
    ):
        _authorized_probe(executable, library)

    descendant_pid = int(pid_file.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0.0, -1.0, True])
def test_probe_rejects_invalid_timeout_before_spawning(
    tmp_path: Path, timeout: float
) -> None:
    executable, library = _probe_script(tmp_path, VALID_OUTPUT)

    with pytest.raises(
        MlxceleratorRuntimeError,
        match="mlxcelerator-runtime-probe-timeout-invalid",
    ):
        probe_mlxcelerator_runtime(
            executable,
            library,
            expected_executable_sha256=regular_file_identity(executable)["sha256"],
            expected_mlx_c_sha256=regular_file_identity(library)["sha256"],
            expected_library_manifest_sha256=tree_manifest(
                library.parent,
                reject_symlinks=True,
                require_root_owned=False,
            )["manifest_sha256"],
            timeout_seconds=timeout,
        )


def test_probe_rejects_symlinked_executable(tmp_path: Path) -> None:
    executable, library = _probe_script(tmp_path, VALID_OUTPUT)
    linked_executable = tmp_path / "linked-mlxcelerator"
    linked_executable.symlink_to(executable)

    with pytest.raises(
        MlxceleratorRuntimeError,
        match="mlxcelerator-runtime-executable-symlink-component",
    ):
        probe_mlxcelerator_runtime(
            linked_executable,
            library,
            expected_executable_sha256="0" * 64,
            expected_mlx_c_sha256="0" * 64,
            expected_library_manifest_sha256="0" * 64,
        )
