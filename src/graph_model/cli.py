from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .benchmark import run_graph_vs_loop_benchmark
from .controller import DeterministicGraphController, GraphController
from .graph import load_default_graph, load_graph
from .models import GraphSpec
from .provider import MockProvider, ModelProvider, OpenAICompatibleProvider, ProviderError
from .router_policy import HashedLinearRouter, train_router_file
from .runtime import BudgetExceeded, GraphRuntime, GraphRuntimeError
from .store import SQLiteRunStore
from .trace_export import export_router_training_data
from .workspace import (
    DEFAULT_ALLOWED_COMMANDS,
    RepositoryWorkspace,
    WorkspaceError,
    workspace_initial_data,
)


def _store(path: str | None) -> SQLiteRunStore:
    return SQLiteRunStore(path or os.getenv("GRAPH_MODEL_DB", ".graph-model/runs.sqlite3"))


def _provider(name: str) -> ModelProvider:
    if name == "mock":
        return MockProvider()
    if name == "openai":
        return OpenAICompatibleProvider.from_env()
    if name == "mlx":
        from .mlx_native.provider import MLXLocalProvider

        return MLXLocalProvider.from_env()
    raise ValueError(f"unknown provider {name!r}")


def _controller(name: str, graph: GraphSpec, *, provider_name: str) -> GraphController:
    selected = "mlx" if name == "auto" and provider_name == "mlx" else name
    if selected == "auto":
        selected = "deterministic"
    if selected == "deterministic":
        return DeterministicGraphController()
    if selected == "mlx":
        from .mlx_native.controller import MLXGraphController

        return MLXGraphController.from_env(graph)
    raise ValueError(f"unknown controller {name!r}")


def _load_selected_graph(path: str | None) -> GraphSpec:
    return load_graph(path) if path else load_default_graph()


async def _run_command(args: argparse.Namespace) -> int:
    try:
        graph = _load_selected_graph(args.graph)
        runtime = GraphRuntime(
            graph=graph,
            store=_store(args.db),
            provider=_provider(args.provider),
            controller=_controller(args.controller, graph, provider_name=args.provider),
        )
        initial_data = None
        if args.repo:
            allowed_commands = args.allowed_command or list(DEFAULT_ALLOWED_COMMANDS)
            initial_data = workspace_initial_data(
                source_root=args.repo,
                mode=args.workspace_mode,
                base_ref=args.base_ref,
                workspace_home=args.workspace_home,
                artifact_root=args.artifact_root,
                test_commands=args.test_command or (),
                allowed_commands=allowed_commands,
                command_timeout_seconds=args.command_timeout,
                max_command_output_bytes=args.max_command_output_bytes,
                max_context_files=args.max_context_files,
                max_context_file_bytes=args.max_context_file_bytes,
                max_context_bytes=args.max_context_bytes,
                max_patch_bytes=args.max_patch_bytes,
                max_patch_files=args.max_patch_files,
                allow_sensitive_paths=args.allow_sensitive_paths,
            )
        state = await runtime.run(
            args.task,
            run_id=args.run_id,
            initial_data=initial_data,
            stop_after_steps=args.stop_after_steps,
        )
    except BudgetExceeded as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 2
    except (ProviderError, GraphRuntimeError, ValueError, RuntimeError) as exc:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                indent=2,
            )
        )
        return 2
    print(state.model_dump_json(indent=2))
    return 0 if state.status != "failed" else 1


async def _benchmark_command(args: argparse.Namespace) -> int:
    try:
        graph = load_default_graph()
        report = await run_graph_vs_loop_benchmark(
            input_path=args.input,
            provider=_provider(args.provider),
            output_path=args.output,
            loop_attempts=args.loop_attempts,
            controller=_controller(args.controller, graph, provider_name=args.provider),
        )
    except (ProviderError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


async def _resume_command(args: argparse.Namespace) -> int:
    try:
        graph = _load_selected_graph(args.graph)
        runtime = GraphRuntime(
            graph=graph,
            store=_store(args.db),
            provider=_provider(args.provider),
            controller=_controller(args.controller, graph, provider_name=args.provider),
        )
        state = await runtime.run(run_id=args.run_id)
    except BudgetExceeded as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 2
    except (ProviderError, GraphRuntimeError, ValueError, RuntimeError) as exc:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                indent=2,
            )
        )
        return 2
    print(state.model_dump_json(indent=2))
    return 0 if state.status != "failed" else 1


def _apply_result_command(args: argparse.Namespace) -> int:
    store = _store(args.db)
    state = store.load_run(args.run_id)
    if state is None:
        print(json.dumps({"status": "failed", "error": "run not found"}, indent=2))
        return 2
    if state.status != "completed":
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": f"run must be completed before promotion; status={state.status}",
                },
                indent=2,
            )
        )
        return 2
    workspace = RepositoryWorkspace.from_state_data(state.data, run_id=state.run_id)
    if workspace is None:
        print(json.dumps({"status": "failed", "error": "run has no repository workspace"}, indent=2))
        return 2
    manifest = state.artifacts.get("verified-patch.json")
    if not isinstance(manifest, dict):
        output_workspace = (state.output or {}).get("workspace") if isinstance(state.output, dict) else None
        manifest = (
            output_workspace.get("verified_patch")
            if isinstance(output_workspace, dict)
            else None
        )
    if not isinstance(manifest, dict) or not manifest.get("path") or not manifest.get("sha256"):
        print(json.dumps({"status": "failed", "error": "verified patch manifest is missing"}, indent=2))
        return 2
    try:
        report = workspace.promote_verified_patch(
            str(manifest["path"]),
            str(manifest["sha256"]),
        )
    except (WorkspaceError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                indent=2,
            )
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0



def _cleanup_command(args: argparse.Namespace) -> int:
    store = _store(args.db)
    try:
        with store.run_lock(args.run_id):
            state = store.load_run(args.run_id)
            if state is None:
                print(json.dumps({"status": "failed", "error": "run not found"}, indent=2))
                return 2
            if state.status == "running" and not args.force:
                print(
                    json.dumps(
                        {
                            "status": "failed",
                            "error": "run is still active or resumable; pass --force to discard its worktree",
                        },
                        indent=2,
                    )
                )
                return 2
            workspace = RepositoryWorkspace.from_state_data(
                state.data, run_id=state.run_id
            )
            if workspace is None:
                print(
                    json.dumps(
                        {"status": "failed", "error": "run has no repository workspace"},
                        indent=2,
                    )
                )
                return 2
            retained_patch = False
            for key in ("verified-patch.json", "failed-patch.json"):
                manifest = state.artifacts.get(key)
                if isinstance(manifest, dict) and manifest.get("path"):
                    retained_patch = Path(str(manifest["path"])).is_file()
                    if retained_patch:
                        break
            report = workspace.cleanup_worktree(
                force=args.force or retained_patch
            )
    except (WorkspaceError, OSError, ValueError, RuntimeError) as exc:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                indent=2,
            )
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _trace_command(args: argparse.Namespace) -> int:
    events = _store(args.db).events(args.run_id)
    print(json.dumps(events, indent=2, sort_keys=True))
    return 0 if events else 1


def _export_command(args: argparse.Namespace) -> int:
    count = export_router_training_data(_store(args.db), args.output)
    print(json.dumps({"records": count, "output": str(Path(args.output).resolve())}, indent=2))
    return 0


def _export_mlx_policy_command(args: argparse.Namespace) -> int:
    from .mlx_native.training_data import export_mlx_policy_training_data

    graph = _load_selected_graph(args.graph)
    count = export_mlx_policy_training_data(
        _store(args.db),
        args.output,
        graph=graph,
        success_only=args.success_only,
    )
    print(json.dumps({"records": count, "output": str(Path(args.output).resolve())}, indent=2))
    return 0


def _train_mlx_policy_command(args: argparse.Namespace) -> int:
    from .mlx_native.trainer import train_mlx_policy_file

    try:
        summary = train_mlx_policy_file(
            input_path=args.input,
            output_dir=args.output_dir,
            graph=_load_selected_graph(args.graph),
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    except (ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


def _train_router_command(args: argparse.Namespace) -> int:
    summary = train_router_file(
        input_path=args.input,
        output_path=args.output,
        dimension=args.dimension,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        success_only=args.success_only,
        seed=args.seed,
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


def _predict_route_command(args: argparse.Namespace) -> int:
    prediction = HashedLinearRouter.load(args.model).predict(args.task)
    print(
        json.dumps(
            {
                "route": prediction.route,
                "confidence": prediction.confidence,
                "probabilities": prediction.probabilities,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    from .mlx_native.graph_tables import graph_schema_hash

    graph = _load_selected_graph(args.graph)
    print(
        json.dumps(
            {
                "name": graph.name,
                "version": graph.version,
                "schema_hash": graph_schema_hash(graph),
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "terminals": sorted(graph.terminals),
            },
            indent=2,
        )
    )
    return 0


def _compile_graph_command(args: argparse.Namespace) -> int:
    from .mlx_native.graph_tables import compile_graph, write_generated_module

    graph = _load_selected_graph(args.graph)
    tables = compile_graph(graph)
    output = write_generated_module(tables, args.output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "graph": f"{tables.name}@{tables.version}",
                "schema_hash": tables.schema_hash,
                "nodes": len(tables.node_ids),
                "edges": len(tables.edge_keys),
            },
            indent=2,
        )
    )
    return 0


def _policy_config_command(args: argparse.Namespace) -> int:
    from .mlx_native.graph_tables import compile_graph
    from .mlx_native.policy import GraphPolicyConfig

    graph = _load_selected_graph(args.graph)
    config = GraphPolicyConfig.for_graph(compile_graph(graph), hidden_size=args.hidden_size)
    output = config.save(args.output)
    print(json.dumps({"output": str(output.resolve()), **config.__dict__}, indent=2))
    return 0


def _mlx_doctor_command(args: argparse.Namespace) -> int:
    from .mlx_native.doctor import mlx_diagnostics

    report = mlx_diagnostics(load_model=args.load_model)
    print(json.dumps(report, indent=2, default=str, sort_keys=True))
    return 0 if report.get("ready") else 1


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph")
    parser.add_argument("--db")
    parser.add_argument("--provider", choices=("mock", "openai", "mlx"), default="mock")
    parser.add_argument(
        "--controller",
        choices=("auto", "deterministic", "mlx"),
        default="auto",
        help="auto selects the MLX controller when --provider mlx, otherwise deterministic",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graph-model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="start a graph execution")
    run.add_argument("--task", required=True)
    run.add_argument("--run-id")
    _add_runtime_options(run)
    run.add_argument("--stop-after-steps", type=int)
    run.add_argument("--repo", help="Git repository root for real coding-agent mode")
    run.add_argument(
        "--workspace-mode",
        choices=("worktree", "in-place"),
        default="worktree",
        help="worktree leaves the source checkout untouched; in-place mutates it directly",
    )
    run.add_argument("--base-ref", default="HEAD")
    run.add_argument("--workspace-home")
    run.add_argument("--artifact-root")
    run.add_argument(
        "--test-command",
        action="append",
        help="bounded verifier command; repeat for multiple commands (auto-detected when omitted)",
    )
    run.add_argument(
        "--allowed-command",
        action="append",
        help="replace the default verifier executable allowlist; repeat per executable",
    )
    run.add_argument("--command-timeout", type=float, default=300.0)
    run.add_argument("--max-command-output-bytes", type=int, default=200_000)
    run.add_argument("--max-context-files", type=int, default=18)
    run.add_argument("--max-context-file-bytes", type=int, default=40_000)
    run.add_argument("--max-context-bytes", type=int, default=180_000)
    run.add_argument("--max-patch-bytes", type=int, default=500_000)
    run.add_argument("--max-patch-files", type=int, default=32)
    run.add_argument("--allow-sensitive-paths", action="store_true")

    apply_result = subparsers.add_parser(
        "apply-result",
        help="apply a completed worktree run's hash-verified patch to its clean source checkout",
    )
    apply_result.add_argument("--run-id", required=True)
    apply_result.add_argument("--db")

    cleanup = subparsers.add_parser(
        "cleanup",
        help="remove a run's detached worktree while retaining patches, traces, and artifacts",
    )
    cleanup.add_argument("--run-id", required=True)
    cleanup.add_argument("--db")
    cleanup.add_argument(
        "--force",
        action="store_true",
        help="discard a running or unexported dirty worktree",
    )

    benchmark = subparsers.add_parser("benchmark", help="compare graph execution with a retry loop")
    benchmark.add_argument("--input", required=True)
    benchmark.add_argument("--output")
    benchmark.add_argument("--provider", choices=("mock", "openai", "mlx"), default="mock")
    benchmark.add_argument(
        "--controller", choices=("auto", "deterministic", "mlx"), default="auto"
    )
    benchmark.add_argument("--loop-attempts", type=int, default=3)

    resume = subparsers.add_parser("resume", help="resume a checkpointed execution")
    resume.add_argument("--run-id", required=True)
    _add_runtime_options(resume)

    trace = subparsers.add_parser("trace", help="inspect an execution trace")
    trace.add_argument("--run-id", required=True)
    trace.add_argument("--db")

    export = subparsers.add_parser("export", help="export router training records")
    export.add_argument("--output", required=True)
    export.add_argument("--db")

    export_mlx = subparsers.add_parser(
        "export-mlx-policy",
        help="export MLX route/edge/stop/value/cost training records from traces",
    )
    export_mlx.add_argument("--output", required=True)
    export_mlx.add_argument("--graph")
    export_mlx.add_argument("--db")
    export_mlx.add_argument("--success-only", action="store_true")

    train_mlx = subparsers.add_parser(
        "train-mlx-policy",
        help="train the MLX graph-policy sidecar from exported decision traces",
    )
    train_mlx.add_argument("--input", required=True)
    train_mlx.add_argument("--output-dir", required=True)
    train_mlx.add_argument("--graph")
    train_mlx.add_argument("--hidden-size", type=int, default=128)
    train_mlx.add_argument("--epochs", type=int, default=100)
    train_mlx.add_argument("--learning-rate", type=float, default=1e-3)
    train_mlx.add_argument("--seed", type=int, default=42)

    train_router = subparsers.add_parser("train-router", help="train the constrained route policy")
    train_router.add_argument("--input", required=True)
    train_router.add_argument("--output", required=True)
    train_router.add_argument("--dimension", type=int, default=1024)
    train_router.add_argument("--epochs", type=int, default=40)
    train_router.add_argument("--learning-rate", type=float, default=0.12)
    train_router.add_argument("--seed", type=int, default=42)
    train_router.add_argument("--success-only", action="store_true")

    predict_route = subparsers.add_parser("predict-route", help="inspect a trained route decision")
    predict_route.add_argument("--model", required=True)
    predict_route.add_argument("--task", required=True)

    validate = subparsers.add_parser("validate", help="validate a graph specification")
    validate.add_argument("--graph")

    compile_graph_parser = subparsers.add_parser(
        "compile-graph",
        help="compile a validated YAML graph into immutable Python tables and masks",
    )
    compile_graph_parser.add_argument("--graph")
    compile_graph_parser.add_argument("--output", required=True)

    policy_config = subparsers.add_parser(
        "policy-config",
        help="write a graph-bound config for MLX route/edge/stop/value/cost heads",
    )
    policy_config.add_argument("--graph")
    policy_config.add_argument("--output", required=True)
    policy_config.add_argument("--hidden-size", type=int, default=128)

    mlx_doctor = subparsers.add_parser(
        "mlx-doctor", help="inspect MLX, MLX-LM, model, adapter, and policy configuration"
    )
    mlx_doctor.add_argument("--load-model", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        code = asyncio.run(_run_command(args))
    elif args.command == "apply-result":
        code = _apply_result_command(args)
    elif args.command == "cleanup":
        code = _cleanup_command(args)
    elif args.command == "benchmark":
        code = asyncio.run(_benchmark_command(args))
    elif args.command == "resume":
        code = asyncio.run(_resume_command(args))
    elif args.command == "trace":
        code = _trace_command(args)
    elif args.command == "export":
        code = _export_command(args)
    elif args.command == "export-mlx-policy":
        code = _export_mlx_policy_command(args)
    elif args.command == "train-mlx-policy":
        code = _train_mlx_policy_command(args)
    elif args.command == "train-router":
        code = _train_router_command(args)
    elif args.command == "predict-route":
        code = _predict_route_command(args)
    elif args.command == "validate":
        code = _validate_command(args)
    elif args.command == "compile-graph":
        code = _compile_graph_command(args)
    elif args.command == "policy-config":
        code = _policy_config_command(args)
    elif args.command == "mlx-doctor":
        code = _mlx_doctor_command(args)
    else:
        parser.error(f"unknown command: {args.command}")
        return
    raise SystemExit(code)
