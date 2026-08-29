"""Authenticated capability bridge for the MLXcelerator Rust runtime."""

from __future__ import annotations

import math
import json
import hashlib
import os
import selectors
import signal
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from graph_model.runtime_identity import (
    canonical_sha256,
    observe_macho_runtime_closure,
    regular_file_identity,
    tree_manifest,
)


MLXCELERATOR_RUNTIME_PROBE_FORMAT = "graph-native-mlxcelerator-runtime-probe-v1"
MLXCELERATOR_LLAMA_ADMISSION_FORMAT = "graph-native-mlxcelerator-llama-admission-v1"
MLXCELERATOR_LLAMA_GENERATION_FORMAT = "graph-native-mlxcelerator-llama-generation-v1"
_MAX_PROBE_OUTPUT_BYTES = 64 * 1024
_REQUIRED_KEYS = frozenset(
    {
        "runtime",
        "chip",
        "model",
        "memory_bytes",
        "os_version",
        "os_build",
        "admitted",
        "admission_reason",
        "mlx_version",
        "mlx_gpu_smoke",
        "mlx_gpu_smoke_error",
        "core_ai_architecture",
        "core_ai_units",
        "nax_gpu_path",
        "lane_gpu",
        "lane_cpu",
        "lane_ane",
    }
)
_LANE_STATES = frozenset({"QualifiedOnly", "Unavailable"})
_ADMISSION_REASONS = frozenset(
    {"qualified", "non_apple_silicon", "below_m5_max_floor", "unknown_hardware"}
)
_U64_MAX = (1 << 64) - 1
_MAX_MODEL_BYTES = 1 << 40
_LLAMA_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "path",
        "file_size",
        "digest_sha256",
        "context_length",
        "embedding_length",
        "block_count",
        "head_count",
        "head_count_kv",
        "feed_forward_length",
        "vocab_size",
        "rms_norm_epsilon",
        "rope_freq_base",
        "tensor_count",
    }
)
_LLAMA_GENERATION_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "backend",
        "path",
        "file_size",
        "digest_sha256",
        "prompt",
        "max_new_tokens",
        "generated_text",
    }
)
_LLAMA_GENERATION_HOST_BACKEND = "mlx-gpu-linear-host-attention-v1"
_LLAMA_GENERATION_SDPA_BACKEND = "mlx-gpu-linear-sdpa-host-kv-v1"
_LLAMA_ATTENTION_MODES = frozenset({"host", "sdpa"})
_LLAMA_ATTENTION_PROJECTION_MODES = frozenset(
    {"host", "mlx-resident-query-sdpa-v1"}
)
_LLAMA_ROPE_MODES = frozenset({"host", "mlx-fast"})
_LLAMA_FFN_MODES = frozenset({"host", "mlx-resident"})
_LLAMA_HIDDEN_STATE_MODES = frozenset({"host", "mlx-resident-hidden-v1"})
_LLAMA_QUANTIZATION_MODES = frozenset({"host", "mlx-affine-q4-v1"})


class MlxceleratorRuntimeError(RuntimeError):
    """The native runtime probe failed its identity or capability contract."""


def probe_mlxcelerator_runtime(
    executable: str | Path,
    mlx_c_library: str | Path,
    *,
    expected_executable_sha256: str,
    expected_library_manifest_sha256: str,
    expected_mlx_c_sha256: str,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Run an exact MLXcelerator binary and return authenticated capabilities.

    Source artifacts are authenticated, copied into a private per-probe
    snapshot, authenticated again, and executed from that snapshot. The local
    account remains inside the trust boundary, as it does for the rest of
    Graph-Native's user-owned runtime identities. When an MLX-C library is
    supplied, the Rust probe verifies a native F32 GPU matrix multiplication,
    not just symbol presence.
    """

    _validate_timeout(timeout_seconds)
    executable_path = _require_executable(Path(executable))
    mlx_c_path = _require_regular_file(Path(mlx_c_library), "mlx-c-library")
    mlx_library_root = mlx_c_path.parent

    executable_before = regular_file_identity(executable_path)
    mlx_c_before = regular_file_identity(mlx_c_path)
    library_before = _library_manifest(mlx_library_root)
    if executable_before["sha256"] != expected_executable_sha256:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-executable-unauthorized")
    if library_before["manifest_sha256"] != expected_library_manifest_sha256:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-library-unauthorized")
    if mlx_c_before["sha256"] != expected_mlx_c_sha256:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-mlx-c-unauthorized")
    with tempfile.TemporaryDirectory(prefix="graph-mlxcelerator-probe-") as temporary:
        snapshot_root = Path(temporary)
        snapshot_executable = snapshot_root / "mlxcelerator"
        snapshot_library_root = snapshot_root / "mlx"
        try:
            shutil.copy2(executable_path, snapshot_executable, follow_symlinks=False)
            shutil.copytree(
                mlx_library_root,
                snapshot_library_root,
                symlinks=True,
                copy_function=shutil.copy2,
            )
        except OSError as exc:
            raise MlxceleratorRuntimeError(
                "mlxcelerator-runtime-snapshot-failed"
            ) from exc

        snapshot_executable_identity = regular_file_identity(snapshot_executable)
        snapshot_library = _library_manifest(snapshot_library_root)
        if snapshot_executable_identity["sha256"] != executable_before["sha256"]:
            raise MlxceleratorRuntimeError(
                "mlxcelerator-runtime-snapshot-executable-drift"
            )
        if not _same_tree_content(library_before, snapshot_library):
            raise MlxceleratorRuntimeError(
                "mlxcelerator-runtime-snapshot-library-drift"
            )

        snapshot_mlx_c = snapshot_library_root / mlx_c_path.name
        if regular_file_identity(snapshot_mlx_c)["sha256"] != mlx_c_before["sha256"]:
            raise MlxceleratorRuntimeError("mlxcelerator-runtime-snapshot-mlx-c-drift")
        snapshot_preloads = tuple(
            candidate
            for name in ("libjaccl.dylib", "libmlx.dylib")
            if (candidate := snapshot_library_root / name).is_file()
        )
        native_closure = observe_macho_runtime_closure(
            snapshot_executable,
            required_binary_paths=(snapshot_mlx_c,),
            preloaded_binary_paths=snapshot_preloads,
            owned_native_roots=(snapshot_root,),
        )
        returncode, stdout, stderr = _run_probe_bounded(
            [str(snapshot_executable), "probe", str(snapshot_mlx_c)],
            cwd=snapshot_root,
            env={
                "PATH": "/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
            },
            timeout=timeout_seconds,
        )

        if regular_file_identity(snapshot_executable) != snapshot_executable_identity:
            raise MlxceleratorRuntimeError(
                "mlxcelerator-runtime-snapshot-executable-drift"
            )
        if _library_manifest(snapshot_library_root) != snapshot_library:
            raise MlxceleratorRuntimeError(
                "mlxcelerator-runtime-snapshot-library-drift"
            )
        if (
            observe_macho_runtime_closure(
                snapshot_executable,
                required_binary_paths=(snapshot_mlx_c,),
                preloaded_binary_paths=snapshot_preloads,
                owned_native_roots=(snapshot_root,),
            )
            != native_closure
        ):
            raise MlxceleratorRuntimeError("mlxcelerator-runtime-native-closure-drift")

    if returncode != 0:
        raise MlxceleratorRuntimeError(f"mlxcelerator-runtime-probe-exit:{returncode}")

    executable_after = regular_file_identity(executable_path)
    mlx_c_after = regular_file_identity(mlx_c_path)
    library_after = _library_manifest(mlx_library_root)
    if executable_before != executable_after:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-executable-drift")
    if library_before != library_after:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-library-drift")
    if mlx_c_before != mlx_c_after:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-mlx-c-drift")

    try:
        output = stdout.decode("utf-8", errors="strict")
        stderr.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-probe-not-utf8") from exc
    capabilities = _parse_probe_output(output)
    receipt = {
        "format": MLXCELERATOR_RUNTIME_PROBE_FORMAT,
        "executable": executable_before,
        "mlx_c_library": {
            "relative_path": mlx_c_path.name,
            "identity": mlx_c_before,
        },
        "mlx_library_manifest": library_before,
        "native_closure": native_closure,
        "capabilities": capabilities,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def admit_mlxcelerator_llama_model(
    executable: str | Path,
    model: str | Path,
    *,
    expected_executable_sha256: str,
    expected_model_sha256: str,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Authenticate and run the Rust Llama model-admission command.

    The executable and model are copied into a private snapshot before the
    command runs. The receipt proves only metadata and tensor-layout admission,
    not decoder execution or text generation.
    """

    _validate_timeout(timeout_seconds)
    executable_path = _require_executable(Path(executable))
    model_path = _require_regular_file(Path(model), "model")
    executable_before = regular_file_identity(executable_path)
    model_before = _model_file_identity(model_path)
    if executable_before["sha256"] != expected_executable_sha256:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-executable-unauthorized")
    if model_before["sha256"] != expected_model_sha256:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-model-unauthorized")

    with tempfile.TemporaryDirectory(prefix="graph-mlxcelerator-llama-") as temporary:
        snapshot_root = Path(temporary)
        snapshot_executable = snapshot_root / "mlxcelerator"
        snapshot_model = snapshot_root / "model.gguf"
        try:
            shutil.copy2(executable_path, snapshot_executable, follow_symlinks=False)
            shutil.copy2(model_path, snapshot_model, follow_symlinks=False)
            snapshot_executable.chmod(executable_before["mode"])
        except OSError as exc:
            raise MlxceleratorRuntimeError(
                "mlxcelerator-llama-snapshot-failed"
            ) from exc
        snapshot_executable_identity = regular_file_identity(snapshot_executable)
        snapshot_model_identity = _model_file_identity(snapshot_model)
        if snapshot_executable_identity["sha256"] != executable_before["sha256"]:
            raise MlxceleratorRuntimeError("mlxcelerator-llama-snapshot-executable-drift")
        if snapshot_model_identity["sha256"] != model_before["sha256"]:
            raise MlxceleratorRuntimeError("mlxcelerator-llama-snapshot-model-drift")
        returncode, stdout, stderr = _run_probe_bounded(
            [str(snapshot_executable), "llama-index", str(snapshot_model)],
            cwd=snapshot_root,
            env={
                "PATH": "/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
            },
            timeout=timeout_seconds,
        )
        if regular_file_identity(snapshot_executable) != snapshot_executable_identity:
            raise MlxceleratorRuntimeError("mlxcelerator-llama-snapshot-executable-drift")
        if _model_file_identity(snapshot_model) != snapshot_model_identity:
            raise MlxceleratorRuntimeError("mlxcelerator-llama-snapshot-model-drift")

    if returncode != 0:
        raise MlxceleratorRuntimeError(f"mlxcelerator-llama-admission-exit:{returncode}")
    try:
        output = stdout.decode("utf-8", errors="strict")
        stderr.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-admission-not-utf8") from exc
    admission = _parse_llama_index_output(output, model_before)
    executable_after = regular_file_identity(executable_path)
    model_after = _model_file_identity(model_path)
    if executable_before != executable_after:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-executable-drift")
    if model_before != model_after:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-model-drift")
    receipt = {
        "format": MLXCELERATOR_LLAMA_ADMISSION_FORMAT,
        "executable": executable_before,
        "model": model_before,
        "admission": admission,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def generate_mlxcelerator_llama_text(
    executable: str | Path,
    model: str | Path,
    mlx_c_library: str | Path,
    prompt: str,
    max_new_tokens: int,
    *,
    expected_executable_sha256: str,
    expected_model_sha256: str,
    expected_library_manifest_sha256: str,
    expected_mlx_c_sha256: str,
    attention_mode: str = "host",
    attention_projection_mode: str = "host",
    rope_mode: str = "host",
    ffn_mode: str = "host",
    hidden_state_mode: str = "host",
    quantization_mode: str = "host",
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Authenticate and run the native MLX Llama generation command.

    The executable, model, and complete MLX library directory are copied into
    a private snapshot before execution. The returned receipt binds the exact
    content identities, native Mach-O closure, prompt, token budget, backend,
    and generated text. ``attention_mode="sdpa"`` opt-in passes
    ``MLXC_USE_SDPA=1`` and requires the runtime to report the distinct SDPA
    backend and mode in its output. ``rope_mode="mlx-fast"`` passes
    ``MLXC_USE_ROPE=1`` and requires an explicit ``rope_mode`` receipt field.
    ``attention_projection_mode="mlx-resident-query-sdpa-v1"`` is valid only
    with both SDPA and native RoPE enabled. It passes
    ``MLXC_USE_SDPA_RESIDENT=1`` only for that combination and requires the
    exact versioned ``attention_projection_mode`` receipt field.
    ``ffn_mode="mlx-resident"`` passes ``MLXC_USE_DEVICE_FFN=1`` and requires
    an explicit ``ffn_mode`` receipt field. All opt-in modes fail closed
    against an older binary that silently ignores the request. The resident
    hidden-state mode requires resident attention projection and resident FFN;
    it also enables the resident residual path and requires an exact
    ``hidden_state_mode`` receipt field.
    ``quantization_mode="mlx-affine-q4-v1"`` passes ``MLXC_USE_QUANTIZED=1``
    and requires the exact quantization receipt field. The mode is an explicit
    MLX affine representation, not a claim that GGUF block formats are
    losslessly interchangeable with it.
    """

    _validate_timeout(timeout_seconds)
    if (
        not isinstance(attention_mode, str)
        or attention_mode not in _LLAMA_ATTENTION_MODES
    ):
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-attention-mode-invalid"
        )
    if (
        not isinstance(attention_projection_mode, str)
        or attention_projection_mode not in _LLAMA_ATTENTION_PROJECTION_MODES
    ):
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-attention-projection-mode-invalid"
        )
    if not isinstance(rope_mode, str) or rope_mode not in _LLAMA_ROPE_MODES:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-rope-mode-invalid")
    if not isinstance(ffn_mode, str) or ffn_mode not in _LLAMA_FFN_MODES:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-ffn-mode-invalid")
    if (
        not isinstance(hidden_state_mode, str)
        or hidden_state_mode not in _LLAMA_HIDDEN_STATE_MODES
    ):
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-hidden-state-mode-invalid"
        )
    if (
        not isinstance(quantization_mode, str)
        or quantization_mode not in _LLAMA_QUANTIZATION_MODES
    ):
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-quantization-mode-invalid"
        )
    if (
        attention_projection_mode == "mlx-resident-query-sdpa-v1"
        and (attention_mode != "sdpa" or rope_mode != "mlx-fast")
    ):
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-attention-projection-mode-requires-sdpa-rope"
        )
    if hidden_state_mode == "mlx-resident-hidden-v1" and (
        attention_projection_mode != "mlx-resident-query-sdpa-v1"
        or ffn_mode != "mlx-resident"
    ):
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-hidden-state-mode-requires-resident-primitives"
        )
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 0 <= max_new_tokens <= 1_000_000
    ):
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-token-budget-invalid")
    if not isinstance(prompt, str) or len(prompt.encode("utf-8")) > _MAX_PROBE_OUTPUT_BYTES:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-prompt-invalid")
    executable_path = _require_executable(Path(executable))
    model_path = _require_regular_file(Path(model), "model")
    mlx_c_path = _require_regular_file(Path(mlx_c_library), "mlx-c-library")
    mlx_library_root = mlx_c_path.parent
    executable_before = regular_file_identity(executable_path)
    model_before = _model_file_identity(model_path)
    mlx_c_before = regular_file_identity(mlx_c_path)
    library_before = _library_manifest(mlx_library_root)
    if executable_before["sha256"] != expected_executable_sha256:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-executable-unauthorized")
    if model_before["sha256"] != expected_model_sha256:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-model-unauthorized")
    if library_before["manifest_sha256"] != expected_library_manifest_sha256:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-library-unauthorized")
    if mlx_c_before["sha256"] != expected_mlx_c_sha256:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-mlx-c-unauthorized")

    with tempfile.TemporaryDirectory(prefix="graph-mlxcelerator-llama-generation-") as temporary:
        snapshot_root = Path(temporary)
        snapshot_executable = snapshot_root / "mlxcelerator"
        snapshot_model = snapshot_root / "model.gguf"
        snapshot_library_root = snapshot_root / "mlx"
        try:
            shutil.copy2(executable_path, snapshot_executable, follow_symlinks=False)
            shutil.copy2(model_path, snapshot_model, follow_symlinks=False)
            shutil.copytree(
                mlx_library_root,
                snapshot_library_root,
                symlinks=True,
                copy_function=shutil.copy2,
            )
        except OSError as exc:
            raise MlxceleratorRuntimeError(
                "mlxcelerator-llama-generation-snapshot-failed"
            ) from exc
        snapshot_executable.chmod(executable_before["mode"])
        snapshot_executable_identity = regular_file_identity(snapshot_executable)
        snapshot_model_identity = _model_file_identity(snapshot_model)
        snapshot_library = _library_manifest(snapshot_library_root)
        if snapshot_executable_identity["sha256"] != executable_before["sha256"]:
            raise MlxceleratorRuntimeError(
                "mlxcelerator-llama-generation-snapshot-executable-drift"
            )
        if snapshot_model_identity["sha256"] != model_before["sha256"]:
            raise MlxceleratorRuntimeError(
                "mlxcelerator-llama-generation-snapshot-model-drift"
            )
        if not _same_tree_content(library_before, snapshot_library):
            raise MlxceleratorRuntimeError(
                "mlxcelerator-llama-generation-snapshot-library-drift"
            )
        snapshot_mlx_c = snapshot_library_root / mlx_c_path.name
        if regular_file_identity(snapshot_mlx_c)["sha256"] != mlx_c_before["sha256"]:
            raise MlxceleratorRuntimeError(
                "mlxcelerator-llama-generation-snapshot-mlx-c-drift"
            )
        snapshot_preloads = tuple(
            candidate
            for name in ("libjaccl.dylib", "libmlx.dylib")
            if (candidate := snapshot_library_root / name).is_file()
        )
        native_closure = observe_macho_runtime_closure(
            snapshot_executable,
            required_binary_paths=(snapshot_mlx_c,),
            preloaded_binary_paths=snapshot_preloads,
            owned_native_roots=(snapshot_root,),
        )
        runtime_env = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
        }
        if attention_mode == "sdpa":
            # Both attention paths are opt-in. The receipt parser below
            # requires explicit mode evidence from the runtime before
            # admitting the result.
            runtime_env["MLXC_USE_SDPA"] = "1"
        if attention_projection_mode == "mlx-resident-query-sdpa-v1":
            runtime_env["MLXC_USE_SDPA_RESIDENT"] = "1"
        if rope_mode == "mlx-fast":
            runtime_env["MLXC_USE_ROPE"] = "1"
        if ffn_mode == "mlx-resident":
            runtime_env["MLXC_USE_DEVICE_FFN"] = "1"
        if hidden_state_mode == "mlx-resident-hidden-v1":
            runtime_env["MLXC_USE_DEVICE_RESIDUAL"] = "1"
        if quantization_mode == "mlx-affine-q4-v1":
            runtime_env["MLXC_USE_QUANTIZED"] = "1"
        returncode, stdout, stderr = _run_probe_bounded(
            [
                str(snapshot_executable),
                "llama-generate-mlx",
                str(snapshot_model),
                str(snapshot_mlx_c),
                str(max_new_tokens),
                prompt,
            ],
            cwd=snapshot_root,
            env=runtime_env,
            timeout=timeout_seconds,
        )
        if regular_file_identity(snapshot_executable) != snapshot_executable_identity:
            raise MlxceleratorRuntimeError(
                "mlxcelerator-llama-generation-snapshot-executable-drift"
            )
        if _model_file_identity(snapshot_model) != snapshot_model_identity:
            raise MlxceleratorRuntimeError(
                "mlxcelerator-llama-generation-snapshot-model-drift"
            )
        if _library_manifest(snapshot_library_root) != snapshot_library:
            raise MlxceleratorRuntimeError(
                "mlxcelerator-llama-generation-snapshot-library-drift"
            )
        if (
            observe_macho_runtime_closure(
                snapshot_executable,
                required_binary_paths=(snapshot_mlx_c,),
                preloaded_binary_paths=snapshot_preloads,
                owned_native_roots=(snapshot_root,),
            )
            != native_closure
        ):
            raise MlxceleratorRuntimeError(
                "mlxcelerator-llama-generation-native-closure-drift"
            )

    if returncode != 0:
        raise MlxceleratorRuntimeError(
            f"mlxcelerator-llama-generation-exit:{returncode}"
        )
    try:
        output = stdout.decode("utf-8", errors="strict")
        stderr.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-not-utf8"
        ) from exc
    generation = _parse_llama_generation_output(
        output,
        model_before,
        prompt,
        max_new_tokens,
        attention_mode,
        attention_projection_mode,
        rope_mode,
        ffn_mode,
        hidden_state_mode,
        quantization_mode,
    )
    executable_after = regular_file_identity(executable_path)
    model_after = _model_file_identity(model_path)
    mlx_c_after = regular_file_identity(mlx_c_path)
    library_after = _library_manifest(mlx_library_root)
    if executable_before != executable_after:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-executable-drift")
    if model_before != model_after:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-model-drift")
    if mlx_c_before != mlx_c_after:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-mlx-c-drift")
    if library_before != library_after:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-library-drift")
    receipt = {
        "format": MLXCELERATOR_LLAMA_GENERATION_FORMAT,
        "executable": executable_before,
        "model": model_before,
        "mlx_c_library": {
            "relative_path": mlx_c_path.name,
            "identity": mlx_c_before,
        },
        "mlx_library_manifest": library_before,
        "native_closure": native_closure,
        "generation": generation,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _run_probe_bounded(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
) -> tuple[int, bytes, bytes]:
    _validate_timeout(timeout)
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-probe-failed") from exc
    selector: selectors.BaseSelector | None = None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout
    try:
        if process.stdout is None or process.stderr is None:
            raise MlxceleratorRuntimeError("mlxcelerator-runtime-probe-pipe-failed")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MlxceleratorRuntimeError("mlxcelerator-runtime-probe-timeout")
            for key, _mask in selector.select(min(remaining, 0.1)):
                chunk = os.read(key.fd, 64 * 1024)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffer = buffers[key.data]
                buffer.extend(chunk)
                if len(buffer) > _MAX_PROBE_OUTPUT_BYTES:
                    raise MlxceleratorRuntimeError(
                        "mlxcelerator-runtime-probe-output-exceeded"
                    )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MlxceleratorRuntimeError("mlxcelerator-runtime-probe-timeout")
        returncode = process.wait(timeout=remaining)
        if _kill_process_group(process):
            raise MlxceleratorRuntimeError(
                "mlxcelerator-runtime-probe-descendant-process"
            )
    except subprocess.TimeoutExpired as exc:
        _kill_process_group(process)
        process.wait()
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-probe-timeout") from exc
    except MlxceleratorRuntimeError:
        _kill_process_group(process)
        process.wait()
        raise
    except (OSError, TypeError, ValueError) as exc:
        _kill_process_group(process)
        process.wait()
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-probe-failed") from exc
    finally:
        if selector is not None:
            selector.close()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    return returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"])


def _validate_timeout(timeout: float) -> None:
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-probe-timeout-invalid")


def _kill_process_group(process: subprocess.Popen[bytes]) -> bool:
    """Kill any process still in the probe's isolated process group.

    Returns whether the group still existed. Once the direct child has been
    reaped, a surviving group means the probe detached a descendant.
    """

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return False
    except PermissionError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        return True
    return True


def _library_manifest(root: Path) -> dict[str, Any]:
    return tree_manifest(
        root,
        reject_symlinks=True,
        require_root_owned=False,
    )


def _same_tree_content(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left[field] == right[field]
        for field in ("entries", "file_count", "total_bytes")
    )


def _require_executable(path: Path) -> Path:
    resolved = _require_regular_file(path, "executable")
    if not resolved.stat().st_mode & stat.S_IXUSR:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-not-executable")
    return resolved


def _require_regular_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise MlxceleratorRuntimeError(f"mlxcelerator-runtime-{label}-not-absolute")
    absolute = path.absolute()
    _reject_symlinked_components(absolute, label)
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise MlxceleratorRuntimeError(
            f"mlxcelerator-runtime-{label}-unavailable"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise MlxceleratorRuntimeError(f"mlxcelerator-runtime-{label}-invalid")
    return absolute


def _model_file_identity(path: Path) -> dict[str, Any]:
    """Hash a stable model file without applying the native-library size cap."""

    target = path.absolute()
    try:
        before = target.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise MlxceleratorRuntimeError("mlxcelerator-runtime-model-invalid")
        if before.st_size > _MAX_MODEL_BYTES:
            raise MlxceleratorRuntimeError("mlxcelerator-runtime-model-too-large")
        descriptor = os.open(
            target,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except MlxceleratorRuntimeError:
        raise
    except OSError as exc:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-model-unavailable") from exc
    try:
        opened = os.fstat(descriptor)
        if opened.st_size != before.st_size or opened.st_size > _MAX_MODEL_BYTES:
            raise MlxceleratorRuntimeError("mlxcelerator-runtime-model-replaced")
        digest = hashlib.sha256()
        bytes_read = 0
        while bytes_read < opened.st_size:
            chunk = os.read(descriptor, min(1024 * 1024, opened.st_size - bytes_read))
            if not chunk:
                raise MlxceleratorRuntimeError("mlxcelerator-runtime-model-replaced")
            digest.update(chunk)
            bytes_read += len(chunk)
        after_read = os.fstat(descriptor)
    except MlxceleratorRuntimeError:
        raise
    except OSError as exc:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-model-read-failed") from exc
    finally:
        os.close(descriptor)
    try:
        after_path = target.lstat()
    except OSError as exc:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-model-replaced") from exc
    if _stable_file_metadata(before) != _stable_file_metadata(after_read) or _stable_file_metadata(
        before
    ) != _stable_file_metadata(after_path):
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-model-replaced")
    return {
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


def _stable_file_metadata(metadata: os.stat_result) -> tuple[int, ...]:
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


def _reject_symlinked_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise MlxceleratorRuntimeError(
                f"mlxcelerator-runtime-{label}-unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise MlxceleratorRuntimeError(
                f"mlxcelerator-runtime-{label}-symlink-component"
            )


def _parse_probe_output(output: str) -> dict[str, Any]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if not line or "=" not in line:
            raise MlxceleratorRuntimeError("mlxcelerator-runtime-probe-line-invalid")
        key, value = line.split("=", 1)
        if key not in _REQUIRED_KEYS:
            raise MlxceleratorRuntimeError(
                f"mlxcelerator-runtime-probe-key-unknown:{key}"
            )
        if key in values:
            raise MlxceleratorRuntimeError(
                f"mlxcelerator-runtime-probe-key-duplicate:{key}"
            )
        values[key] = value
    missing = sorted(_REQUIRED_KEYS - values.keys())
    if missing:
        raise MlxceleratorRuntimeError(
            "mlxcelerator-runtime-probe-key-missing:" + ",".join(missing)
        )

    admitted = _parse_bool(values["admitted"], "admitted")
    mlx_gpu_smoke = _parse_bool(values["mlx_gpu_smoke"], "mlx_gpu_smoke")
    nax_gpu_path = _parse_bool(values["nax_gpu_path"], "nax_gpu_path")
    if values["admission_reason"] not in _ADMISSION_REASONS:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-admission-reason-invalid")
    if admitted != (values["admission_reason"] == "qualified"):
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-admission-inconsistent")
    if mlx_gpu_smoke != (values["mlx_gpu_smoke_error"] == "none"):
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-gpu-smoke-inconsistent")
    if not mlx_gpu_smoke and not values["mlx_gpu_smoke_error"]:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-gpu-smoke-inconsistent")
    for key in ("lane_gpu", "lane_cpu", "lane_ane"):
        if values[key] not in _LANE_STATES:
            raise MlxceleratorRuntimeError(f"mlxcelerator-runtime-lane-invalid:{key}")
    try:
        memory_bytes = int(values["memory_bytes"])
    except ValueError as exc:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-memory-invalid") from exc
    if not 0 < memory_bytes <= _U64_MAX:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-memory-invalid")
    if not values["runtime"].startswith("mlxcelerator-mlx-runtime/"):
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-identity-invalid")
    if not values["mlx_version"] or values["mlx_version"] == "not_probed":
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-mlx-unprobed")
    for key in ("model", "os_version", "os_build"):
        if not values[key]:
            raise MlxceleratorRuntimeError(
                f"mlxcelerator-runtime-capability-empty:{key}"
            )
    core_ai_units = (
        values["core_ai_units"].split(",") if values["core_ai_units"] else []
    )
    if len(core_ai_units) != len(set(core_ai_units)) or not set(core_ai_units) <= {
        "cpu",
        "gpu",
        "ane",
    }:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-core-ai-units-invalid")
    canonical_core_ai_units = [
        unit for unit in ("cpu", "gpu", "ane") if unit in core_ai_units
    ]
    if core_ai_units != canonical_core_ai_units:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-core-ai-units-invalid")
    if not values["core_ai_architecture"] or (
        values["core_ai_architecture"] == "not_available"
    ) == bool(core_ai_units):
        raise MlxceleratorRuntimeError(
            "mlxcelerator-runtime-core-ai-architecture-inconsistent"
        )
    expected_gpu_lane = (
        "QualifiedOnly"
        if admitted and (mlx_gpu_smoke or "gpu" in core_ai_units)
        else "Unavailable"
    )
    expected_cpu_lane = "QualifiedOnly" if admitted else "Unavailable"
    expected_ane_lane = (
        "QualifiedOnly" if admitted and "ane" in core_ai_units else "Unavailable"
    )
    if values["lane_gpu"] != expected_gpu_lane:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-gpu-lane-inconsistent")
    if values["lane_cpu"] != expected_cpu_lane:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-cpu-lane-inconsistent")
    if values["lane_ane"] != expected_ane_lane:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-ane-lane-inconsistent")
    if nax_gpu_path:
        raise MlxceleratorRuntimeError("mlxcelerator-runtime-nax-unimplemented")

    return {
        "runtime_identity": values["runtime"],
        "chip": values["chip"],
        "model": values["model"],
        "memory_bytes": memory_bytes,
        "os_version": values["os_version"],
        "os_build": values["os_build"],
        "admitted": admitted,
        "admission_reason": values["admission_reason"],
        "mlx_version": values["mlx_version"],
        "mlx_gpu_smoke": mlx_gpu_smoke,
        "mlx_gpu_smoke_error": values["mlx_gpu_smoke_error"],
        "core_ai_architecture": values["core_ai_architecture"],
        "core_ai_units": core_ai_units,
        "nax_gpu_path": nax_gpu_path,
        "lanes": {
            "gpu": values["lane_gpu"],
            "cpu": values["lane_cpu"],
            "ane": values["lane_ane"],
        },
    }


def _parse_llama_index_output(
    output: str, model_identity: dict[str, Any]
) -> dict[str, Any]:
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-admission-json-invalid") from exc
    if not isinstance(value, dict) or set(value) != _LLAMA_REQUIRED_KEYS:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-admission-schema-invalid")
    if (
        isinstance(value["schema_version"], bool)
        or not isinstance(value["schema_version"], int)
        or value["schema_version"] != 1
    ):
        raise MlxceleratorRuntimeError("mlxcelerator-llama-admission-schema-version-invalid")
    if not isinstance(value["path"], str) or not value["path"]:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-admission-path-invalid")
    if (
        isinstance(value["file_size"], bool)
        or not isinstance(value["file_size"], int)
        or value["file_size"] != model_identity["bytes"]
    ):
        raise MlxceleratorRuntimeError("mlxcelerator-llama-admission-size-mismatch")
    if (
        not isinstance(value["digest_sha256"], str)
        or value["digest_sha256"] != model_identity["sha256"]
    ):
        raise MlxceleratorRuntimeError("mlxcelerator-llama-admission-digest-mismatch")
    dimensions = {}
    for key in (
        "context_length",
        "embedding_length",
        "block_count",
        "head_count",
        "head_count_kv",
        "feed_forward_length",
        "vocab_size",
        "tensor_count",
    ):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise MlxceleratorRuntimeError(
                f"mlxcelerator-llama-admission-{key}-invalid"
            )
        dimensions[key] = item
    for key in ("rms_norm_epsilon", "rope_freq_base"):
        item = value[key]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or item <= 0
        ):
            raise MlxceleratorRuntimeError(
                f"mlxcelerator-llama-admission-{key}-invalid"
            )
    if dimensions["head_count_kv"] > dimensions["head_count"]:
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-admission-head-count-kv-invalid"
        )
    if dimensions["head_count"] % dimensions["head_count_kv"]:
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-admission-head-count-kv-invalid"
        )
    if dimensions["embedding_length"] % dimensions["head_count"]:
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-admission-embedding-length-invalid"
        )
    return {
        **dimensions,
        "path": value["path"],
        "file_size": value["file_size"],
        "digest_sha256": value["digest_sha256"],
        "rms_norm_epsilon": value["rms_norm_epsilon"],
        "rope_freq_base": value["rope_freq_base"],
    }


def _parse_llama_generation_output(
    output: str,
    model_identity: dict[str, Any],
    prompt: str,
    max_new_tokens: int,
    attention_mode: str = "host",
    attention_projection_mode: str = "host",
    rope_mode: str = "host",
    ffn_mode: str = "host",
    hidden_state_mode: str = "host",
    quantization_mode: str = "host",
) -> dict[str, Any]:
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-json-invalid"
        ) from exc
    required_keys = set(_LLAMA_GENERATION_REQUIRED_KEYS)
    if attention_mode == "sdpa":
        required_keys.add("attention_mode")
    if attention_projection_mode == "mlx-resident-query-sdpa-v1":
        required_keys.add("attention_projection_mode")
    if rope_mode == "mlx-fast":
        required_keys.add("rope_mode")
    if ffn_mode == "mlx-resident":
        required_keys.add("ffn_mode")
    if hidden_state_mode == "mlx-resident-hidden-v1":
        required_keys.add("hidden_state_mode")
    if quantization_mode == "mlx-affine-q4-v1":
        required_keys.add("quantization_mode")
    if not isinstance(value, dict) or set(value) != required_keys:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-schema-invalid")
    if (
        isinstance(value["schema_version"], bool)
        or not isinstance(value["schema_version"], int)
        or value["schema_version"] != 1
    ):
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-schema-version-invalid"
        )
    expected_backend = (
        _LLAMA_GENERATION_SDPA_BACKEND
        if attention_mode == "sdpa"
        else _LLAMA_GENERATION_HOST_BACKEND
    )
    if value["backend"] != expected_backend:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-backend-invalid")
    if attention_mode == "sdpa" and value["attention_mode"] != "sdpa":
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-attention-mode-mismatch"
        )
    if (
        attention_projection_mode == "mlx-resident-query-sdpa-v1"
        and value["attention_projection_mode"] != "mlx-resident-query-sdpa-v1"
    ):
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-attention-projection-mode-mismatch"
        )
    if rope_mode == "mlx-fast" and value["rope_mode"] != "mlx-fast":
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-rope-mode-mismatch"
        )
    if ffn_mode == "mlx-resident" and value["ffn_mode"] != "mlx-resident":
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-ffn-mode-mismatch"
        )
    if (
        hidden_state_mode == "mlx-resident-hidden-v1"
        and value["hidden_state_mode"] != "mlx-resident-hidden-v1"
    ):
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-hidden-state-mode-mismatch"
        )
    if (
        quantization_mode == "mlx-affine-q4-v1"
        and value["quantization_mode"] != "mlx-affine-q4-v1"
    ):
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-quantization-mode-mismatch"
        )
    if not isinstance(value["path"], str) or not value["path"]:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-path-invalid")
    if (
        isinstance(value["file_size"], bool)
        or not isinstance(value["file_size"], int)
        or value["file_size"] != model_identity["bytes"]
    ):
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-size-mismatch")
    if (
        not isinstance(value["digest_sha256"], str)
        or value["digest_sha256"] != model_identity["sha256"]
    ):
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-digest-mismatch")
    if value["prompt"] != prompt:
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-prompt-mismatch")
    if (
        isinstance(value["max_new_tokens"], bool)
        or not isinstance(value["max_new_tokens"], int)
        or value["max_new_tokens"] != max_new_tokens
    ):
        raise MlxceleratorRuntimeError(
            "mlxcelerator-llama-generation-token-budget-mismatch"
        )
    if not isinstance(value["generated_text"], str):
        raise MlxceleratorRuntimeError("mlxcelerator-llama-generation-text-invalid")
    generation = {
        "schema_version": 1,
        "backend": value["backend"],
        "path": value["path"],
        "file_size": value["file_size"],
        "digest_sha256": value["digest_sha256"],
        "prompt": value["prompt"],
        "max_new_tokens": value["max_new_tokens"],
        "generated_text": value["generated_text"],
    }
    if attention_mode == "sdpa":
        generation["attention_mode"] = value["attention_mode"]
    if attention_projection_mode == "mlx-resident-query-sdpa-v1":
        generation["attention_projection_mode"] = value["attention_projection_mode"]
    if rope_mode == "mlx-fast":
        generation["rope_mode"] = value["rope_mode"]
    if ffn_mode == "mlx-resident":
        generation["ffn_mode"] = value["ffn_mode"]
    if hidden_state_mode == "mlx-resident-hidden-v1":
        generation["hidden_state_mode"] = value["hidden_state_mode"]
    if quantization_mode == "mlx-affine-q4-v1":
        generation["quantization_mode"] = value["quantization_mode"]
    return generation


def _parse_bool(value: str, label: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise MlxceleratorRuntimeError(f"mlxcelerator-runtime-boolean-invalid:{label}")
