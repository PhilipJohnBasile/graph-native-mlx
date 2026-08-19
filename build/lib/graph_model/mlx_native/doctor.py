from __future__ import annotations

import os
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

from .hidden_state import (
    DEFAULT_HIDDEN_FEATURE_SIZE,
    DEFAULT_HIDDEN_LAYER_SPECS,
    DEFAULT_HIDDEN_MAX_INPUT_TOKENS,
    DEFAULT_HIDDEN_POOLING,
    DEFAULT_HIDDEN_PROJECTION_SEED,
    HiddenStateCaptureConfig,
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _path_report(value: str | None) -> dict[str, Any]:
    if not value:
        return {"value": None, "exists": False, "resolved": None}
    path = Path(value).expanduser()
    return {
        "value": value,
        "exists": path.exists(),
        "resolved": str(path.resolve(strict=False)),
    }


def _hidden_report(root: str, capture_enabled: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "capture_enabled": capture_enabled,
        "artifact_root": _path_report(root),
        "persists_raw_prompts": False,
        "persists_raw_hidden_tensors": False,
    }
    try:
        config = HiddenStateCaptureConfig(
            feature_size=int(
                os.getenv(
                    "GRAPH_MODEL_MLX_HIDDEN_FEATURE_SIZE",
                    str(DEFAULT_HIDDEN_FEATURE_SIZE),
                )
            ),
            max_input_tokens=int(
                os.getenv(
                    "GRAPH_MODEL_MLX_HIDDEN_MAX_INPUT_TOKENS",
                    str(DEFAULT_HIDDEN_MAX_INPUT_TOKENS),
                )
            ),
            layer_specs=os.getenv(
                "GRAPH_MODEL_MLX_POLICY_LAYERS",
                ",".join(DEFAULT_HIDDEN_LAYER_SPECS),
            ),
            pooling=os.getenv(
                "GRAPH_MODEL_MLX_POLICY_POOLING",
                DEFAULT_HIDDEN_POOLING,
            ).strip().lower(),
            projection_seed=int(
                os.getenv(
                    "GRAPH_MODEL_MLX_HIDDEN_PROJECTION_SEED",
                    str(DEFAULT_HIDDEN_PROJECTION_SEED),
                )
            ),
        )
        cache_entries = int(
            os.getenv("GRAPH_MODEL_MLX_HIDDEN_CACHE_ENTRIES", "1024")
        )
        if cache_entries < 0:
            raise ValueError("hidden-state cache entries must be >= 0")
        report.update(
            {
                "feature_size": config.feature_size,
                "max_input_tokens": config.max_input_tokens,
                "layers": list(config.layer_specs),
                "pooling": config.pooling,
                "projection_seed": config.projection_seed,
                "cache_entries": cache_entries,
                "schema_hash": config.schema_hash,
                "validation": {"ok": True},
            }
        )
    except (TypeError, ValueError) as exc:
        report["validation"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return report


def _policy_report(
    weights: str | None,
    config: str | None,
    *,
    graph: Any | None = None,
    hidden_schema_hash: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "weights": _path_report(weights),
        "config": _path_report(config),
        "configured": bool(weights),
    }
    if not weights:
        return report
    weights_path = Path(weights).expanduser()
    if not weights_path.is_file():
        report["validation"] = {
            "ok": False,
            "error": f"policy weights do not exist: {weights_path}",
        }
        return report
    config_path = (
        Path(config).expanduser()
        if config
        else weights_path.with_name("graph_policy.json")
    )
    report["config"] = _path_report(str(config_path))
    if not config_path.is_file():
        report["validation"] = {
            "ok": False,
            "error": f"policy config does not exist: {config_path}",
        }
        return report
    try:
        from graph_model.graph import load_default_graph

        from .graph_tables import compile_graph
        from .policy import GraphPolicyConfig

        selected_graph = graph if graph is not None else load_default_graph()
        policy_config = GraphPolicyConfig.load(config_path)
        policy_config.validate(compile_graph(selected_graph))
        compatibility_ok = True
        compatibility_error = None
        if (
            policy_config.requires_hidden
            and hidden_schema_hash
            and policy_config.hidden_state_schema_hash != hidden_schema_hash
        ):
            compatibility_ok = False
            compatibility_error = (
                "policy hidden-state schema does not match the configured extractor: "
                f"policy={policy_config.hidden_state_schema_hash}, "
                f"configured={hidden_schema_hash}"
            )
        report["validation"] = {
            "ok": compatibility_ok,
            "format_version": policy_config.format_version,
            "requires_hidden": policy_config.requires_hidden,
            "explicit_feature_size": policy_config.input_size,
            "hidden_feature_size": policy_config.backbone_feature_size,
            "hidden_state_schema_hash": policy_config.hidden_state_schema_hash,
            "model_fingerprint": policy_config.model_fingerprint,
            "graph_schema_hash": policy_config.graph_schema_hash,
        }
        if compatibility_error is not None:
            report["validation"]["error"] = compatibility_error
    except Exception as exc:  # diagnostics must report rather than terminate
        report["validation"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return report


def mlx_diagnostics(
    *,
    load_model: bool = False,
    graph: Any | None = None,
) -> dict[str, Any]:
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
    policy_config = os.getenv("GRAPH_MODEL_MLX_POLICY_CONFIG")
    hidden_root = os.getenv("GRAPH_MODEL_MLX_HIDDEN_ROOT", ".graph-model/hidden-states")
    hidden_capture = _env_bool("GRAPH_MODEL_MLX_CAPTURE_HIDDEN", False)
    hidden_report = _hidden_report(hidden_root, hidden_capture)
    policy_report = _policy_report(
        policy_weights,
        policy_config,
        graph=graph,
        hidden_schema_hash=hidden_report.get("schema_hash"),
    )
    report["configuration"] = {
        # Raw compatibility fields retained for scripts written against v0.3.
        "model": model,
        "adapter": adapter,
        "revision": revision,
        "model_path": _path_report(model),
        "adapter_path": _path_report(adapter),
        "execution": "dedicated-single-worker",
        "hidden_state": hidden_report,
        "policy": policy_report,
        "model_is_local_path": bool(model and Path(model).expanduser().exists()),
        "adapter_exists": bool(adapter and Path(adapter).expanduser().exists()),
        "policy_weights": policy_weights,
        "policy_weights_exists": bool(
            policy_weights and Path(policy_weights).expanduser().is_file()
        ),
    }

    if load_model:
        if not model:
            report["model_load"] = {
                "ok": False,
                "error": "GRAPH_MODEL_MLX_MODEL is not configured",
            }
        else:
            provider = None
            try:
                from .provider import MLXLocalProvider

                provider = MLXLocalProvider.from_env()
                provider.load()
                from graph_model.graph import load_default_graph

                from .controller import MLXGraphController

                selected_graph = graph if graph is not None else load_default_graph()
                controller = MLXGraphController.from_env(
                    selected_graph,
                    hidden_state_source=provider,
                )
                report["model_load"] = {
                    "ok": True,
                    "identity": provider.identity,
                    "hidden_state_identity": provider.hidden_state_identity,
                    "controller_identity": controller.identity,
                }
            except Exception as exc:
                report["model_load"] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                if provider is not None:
                    provider.close()

    policy_ready = (
        not policy_report.get("configured")
        or bool(policy_report.get("validation", {}).get("ok"))
    )
    report["ready"] = bool(
        report["mlx"].get("installed")
        and report["mlx_lm"].get("installed")
        and model
        and hidden_report.get("validation", {}).get("ok")
        and policy_ready
    )
    if load_model:
        report["ready"] = bool(
            report["ready"] and report.get("model_load", {}).get("ok")
        )
    return report
