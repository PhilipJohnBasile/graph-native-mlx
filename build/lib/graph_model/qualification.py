from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .models import GraphSpec, RunState
from .mlx_native.graph_tables import graph_schema_hash


@dataclass(frozen=True)
class QualificationStage:
    name: str
    ok: bool
    elapsed_seconds: float
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "elapsed_seconds": self.elapsed_seconds,
            "details": self.details,
        }


def _memory_snapshot() -> dict[str, float | None]:
    try:
        import mlx.core as mx
    except ImportError:
        return {"active_gib": None, "peak_gib": None, "cache_gib": None}

    def value(name: str) -> float | None:
        function = getattr(mx, name, None)
        if not callable(function):
            metal = getattr(mx, "metal", None)
            function = getattr(metal, name, None) if metal is not None else None
        if not callable(function):
            return None
        try:
            return float(function()) / (1024.0**3)
        except Exception:  # noqa: BLE001 - diagnostics must remain best-effort
            return None

    return {
        "active_gib": value("get_active_memory"),
        "peak_gib": value("get_peak_memory"),
        "cache_gib": value("get_cache_memory"),
    }


def _reset_peak_memory() -> bool:
    try:
        import mlx.core as mx
    except ImportError:
        return False
    function = getattr(mx, "reset_peak_memory", None)
    if not callable(function):
        metal = getattr(mx, "metal", None)
        function = getattr(metal, "reset_peak_memory", None) if metal is not None else None
    if not callable(function):
        return False
    try:
        function()
    except Exception:  # noqa: BLE001 - best-effort diagnostics
        return False
    return True


def _timed_stage(name: str, function: Callable[[], dict[str, Any]]) -> QualificationStage:
    started = time.perf_counter()
    try:
        details = function()
        ok = bool(details.pop("ok", True))
    except Exception as exc:  # diagnostics report instead of hiding partial evidence
        details = {"error": f"{type(exc).__name__}: {exc}"}
        ok = False
    return QualificationStage(
        name=name,
        ok=ok,
        elapsed_seconds=time.perf_counter() - started,
        details=details,
    )


def _markdown_report(report: Mapping[str, Any]) -> str:
    status = "PASS" if report.get("passed") else "FAIL"
    lines = [
        "# Graph-Native MLX M5 Qualification",
        "",
        f"**Overall:** {status}",
        "",
        f"- Graph: `{report.get('graph', {}).get('name')}@{report.get('graph', {}).get('version')}`",
        f"- Graph schema: `{report.get('graph', {}).get('schema_hash')}`",
        f"- Model: `{report.get('model_identity', {}).get('model', '')}`",
        f"- Revision: `{report.get('model_identity', {}).get('revision', '') or 'unpinned'}`",
        "",
        "## Stages",
        "",
        "| Stage | Result | Seconds |",
        "|---|---:|---:|",
    ]
    for stage in report.get("stages", []):
        lines.append(
            f"| {stage.get('name')} | {'PASS' if stage.get('ok') else 'FAIL'} | "
            f"{float(stage.get('elapsed_seconds', 0.0)):.3f} |"
        )
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "The machine-readable JSON report contains model/controller identities, hidden-feature "
            "artifact hashes, token counts, and MLX memory telemetry. Raw hidden tensors are not "
            "included.",
            "",
        ]
    )
    return "\n".join(lines)


async def qualify_mlx_host(
    *,
    graph: GraphSpec,
    output_dir: str | Path,
    provider_factory: Callable[[], Any] | None = None,
    controller_factory: Callable[..., Any] | None = None,
    diagnostics_factory: Callable[..., Mapping[str, Any]] | None = None,
    include_generation: bool = True,
    include_hidden: bool = True,
    require_apple_silicon: bool = True,
) -> dict[str, Any]:
    """Run the exact model/controller qualification path and persist a structured report.

    The factories exist for portable tests. Production callers omit them and use the model,
    revision, adapter, sampler, hidden-state, policy, and diagnostics configuration from the
    environment.
    """

    if provider_factory is None:
        from .mlx_native.provider import MLXLocalProvider

        provider_factory = MLXLocalProvider.from_env
    if controller_factory is None:
        from .mlx_native.controller import MLXGraphController

        controller_factory = MLXGraphController.from_env
    if diagnostics_factory is None:
        from .mlx_native.doctor import mlx_diagnostics

        diagnostics_factory = mlx_diagnostics

    root = Path(output_dir).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    stages: list[QualificationStage] = []
    warnings: list[str] = []

    system_name = platform.system()
    machine = platform.machine()

    def platform_stage() -> dict[str, Any]:
        doctor = dict(diagnostics_factory(load_model=False, graph=graph))
        apple_silicon = system_name == "Darwin" and machine == "arm64"
        metal_available = bool(doctor.get("mlx", {}).get("metal_available"))
        ok = (
            apple_silicon and metal_available
            if require_apple_silicon
            else bool(doctor.get("mlx", {}).get("installed"))
        )
        return {
            "ok": ok,
            "platform": doctor.get("platform"),
            "mlx": doctor.get("mlx"),
            "mlx_lm": doctor.get("mlx_lm"),
            "configuration": doctor.get("configuration"),
        }

    stages.append(_timed_stage("platform-and-configuration", platform_stage))
    provider = None
    controller = None
    hidden_reference: dict[str, Any] | None = None
    generation_evidence: dict[str, Any] | None = None
    model_identity: dict[str, str] = {}
    try:
        provider = provider_factory()

        def load_stage() -> dict[str, Any]:
            peak_reset = _reset_peak_memory()
            before = _memory_snapshot()
            provider.load()
            after = _memory_snapshot()
            nonlocal model_identity
            model_identity = dict(provider.identity)
            return {
                "ok": bool(getattr(provider, "loaded", True)),
                "identity": model_identity,
                "hidden_state_identity": getattr(provider, "hidden_state_identity", ""),
                "peak_memory_reset": peak_reset,
                "memory_before": before,
                "memory_after": after,
            }

        stages.append(_timed_stage("model-load", load_stage))

        if include_generation:
            async def generate() -> QualificationStage:
                started = time.perf_counter()
                before = _memory_snapshot()
                try:
                    payload, prompt_tokens, completion_tokens = await provider.complete_json(
                        system=(
                            "Return only a compact JSON object with keys ok (boolean), "
                            "runtime (string), and purpose (string). Set ok to true."
                        ),
                        user=(
                            "Qualify structured JSON generation for a graph-controlled MLX runtime. "
                            "Do not include markdown or explanatory text."
                        ),
                        temperature=0.0,
                    )
                    ok = payload.get("ok") is True
                    details = {
                        "payload": payload,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "memory_before": before,
                        "memory_after": _memory_snapshot(),
                    }
                    nonlocal generation_evidence
                    generation_evidence = details
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    details = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "memory_before": before,
                        "memory_after": _memory_snapshot(),
                    }
                return QualificationStage(
                    name="structured-generation",
                    ok=ok,
                    elapsed_seconds=time.perf_counter() - started,
                    details=details,
                )

            stages.append(await generate())

        state = RunState.new(
            graph=graph,
            task=(
                "Inspect a repository failure, select the smallest valid graph path, and stop once "
                "deterministic verification and semantic review pass."
            ),
            run_id="mlx-host-qualification",
        )

        if include_hidden:
            def hidden_stage() -> dict[str, Any]:
                before = _memory_snapshot()
                observation = provider.capture_policy_hidden(
                    state=state,
                    node_id=graph.start,
                    decision_type="route",
                )
                reference = observation.reference.as_dict()
                nonlocal hidden_reference
                hidden_reference = reference
                return {
                    "ok": len(observation.features) > 0,
                    "feature_size": len(observation.features),
                    "feature_l2": sum(value * value for value in observation.features) ** 0.5,
                    "cache_hit": observation.cache_hit,
                    "reference": reference,
                    "memory_before": before,
                    "memory_after": _memory_snapshot(),
                }

            stages.append(_timed_stage("qwen-hidden-capture", hidden_stage))

        def controller_stage() -> dict[str, Any]:
            nonlocal controller
            from .runtime import valid_outgoing_edges

            controller = controller_factory(
                graph,
                hidden_state_source=provider,
                capture_hidden_override=include_hidden,
            )
            route = controller.select_route(state.task, state)
            state.data["route"] = route.route
            candidates = valid_outgoing_edges(graph, state, graph.start)
            stop = controller.select_stop(
                graph=graph,
                state=state,
                node_id=graph.start,
                candidates=candidates,
            )
            edge = controller.select_edge(
                graph=graph,
                state=state,
                node_id=graph.start,
                candidates=candidates,
                stop=stop,
            )
            return {
                "ok": (
                    route.route in {"fast", "deep", "repair"}
                    and edge.edge.key in {candidate.key for candidate in candidates}
                ),
                "identity": controller.identity,
                "route_decision": route.as_dict(),
                "stop_decision": stop.as_dict(),
                "edge_decision": edge.as_dict(),
                "memory": _memory_snapshot(),
            }

        stages.append(_timed_stage("mlx-hard-masked-controller", controller_stage))
    finally:
        if provider is not None:
            close = getattr(provider, "close", None)
            if callable(close):
                stages.append(_timed_stage("provider-close", lambda: _close_provider(close)))

    if model_identity and not model_identity.get("revision"):
        warnings.append(
            "The model repository revision is not pinned; qualification is not reproducible across future Hub updates."
        )
    if include_hidden and hidden_reference is None:
        warnings.append("No hidden-feature artifact was produced.")
    if include_generation and generation_evidence is None:
        warnings.append("No structured-generation evidence was produced.")

    passed = all(stage.ok for stage in stages)
    report: dict[str, Any] = {
        "format_version": 1,
        "created_at_unix": time.time(),
        "passed": passed,
        "graph": {
            "name": graph.name,
            "version": graph.version,
            "schema_hash": graph_schema_hash(graph),
        },
        "model_identity": model_identity,
        "controller_identity": (
            dict(controller.identity) if controller is not None else None
        ),
        "hidden_reference": hidden_reference,
        "stages": [stage.as_dict() for stage in stages],
        "warnings": warnings,
        "security": {
            "raw_hidden_tensors_persisted": False,
            "raw_policy_prompts_persisted": False,
        },
    }
    json_path = root / "mlx-m5-qualification.json"
    markdown_path = root / "mlx-m5-qualification.md"
    report["artifacts"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return report


def _close_provider(close: Callable[[], Any]) -> dict[str, Any]:
    close()
    return {"ok": True}
