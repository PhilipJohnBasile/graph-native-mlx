from __future__ import annotations

import os
from importlib import metadata
import platform
from pathlib import Path
from typing import Any


def mlx_diagnostics(*, load_model: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "mlx": {"installed": False},
        "mlx_lm": {"installed": False},
        "configuration": {},
    }
    try:
        import mlx
        import mlx.core as mx

        metal = getattr(mx, "metal", None)
        metal_available = bool(
            metal is not None
            and hasattr(metal, "is_available")
            and metal.is_available()
        )
        device_info = mx.device_info() if hasattr(mx, "device_info") else {}
        mlx_version = str(getattr(mlx, "__version__", "unknown"))
        if mlx_version == "unknown":
            try:
                mlx_version = metadata.version("mlx")
            except metadata.PackageNotFoundError:
                pass
        report["mlx"] = {
            "installed": True,
            "version": mlx_version,
            "default_device": str(mx.default_device()),
            "metal_available": metal_available,
            "device_info": device_info,
        }
    except Exception as exc:
        report["mlx"]["error"] = f"{type(exc).__name__}: {exc}"

    try:
        import mlx_lm

        mlx_lm_version = str(getattr(mlx_lm, "__version__", "unknown"))
        if mlx_lm_version == "unknown":
            try:
                mlx_lm_version = metadata.version("mlx-lm")
            except metadata.PackageNotFoundError:
                pass
        report["mlx_lm"] = {
            "installed": True,
            "version": mlx_lm_version,
        }
    except Exception as exc:
        report["mlx_lm"]["error"] = f"{type(exc).__name__}: {exc}"

    model = os.getenv("GRAPH_MODEL_MLX_MODEL")
    adapter = os.getenv("GRAPH_MODEL_MLX_ADAPTER_PATH")
    revision = os.getenv("GRAPH_MODEL_MLX_REVISION")
    policy_weights = os.getenv("GRAPH_MODEL_MLX_POLICY_WEIGHTS")
    report["configuration"] = {
        "model": model,
        "model_is_local_path": bool(model and Path(model).expanduser().exists()),
        "adapter": adapter,
        "adapter_exists": bool(adapter and Path(adapter).expanduser().exists()),
        "revision": revision,
        "policy_weights": policy_weights,
        "policy_weights_exists": bool(
            policy_weights and Path(policy_weights).expanduser().exists()
        ),
    }

    if load_model:
        if not model:
            report["model_load"] = {
                "ok": False,
                "error": "GRAPH_MODEL_MLX_MODEL is not configured",
            }
        else:
            try:
                from .provider import MLXLocalProvider

                provider = MLXLocalProvider.from_env()
                provider.load()
                report["model_load"] = {"ok": True, "identity": provider.identity}
            except Exception as exc:
                report["model_load"] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    report["ready"] = bool(
        report["mlx"].get("installed")
        and report["mlx_lm"].get("installed")
        and model
    )
    if load_model:
        report["ready"] = bool(report["ready"] and report.get("model_load", {}).get("ok"))
    return report
