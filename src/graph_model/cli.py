from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .benchmark import run_graph_vs_loop_benchmark
from .graph import load_default_graph, load_graph
from .provider import MockProvider, OpenAICompatibleProvider
from .router_policy import HashedLinearRouter, train_router_file
from .runtime import BudgetExceeded, GraphRuntime
from .store import SQLiteRunStore
from .trace_export import export_router_training_data


def _store(path: str | None) -> SQLiteRunStore:
    return SQLiteRunStore(path or os.getenv("GRAPH_MODEL_DB", ".graph-model/runs.sqlite3"))


def _provider(name: str):
    return MockProvider() if name == "mock" else OpenAICompatibleProvider.from_env()


async def _run_command(args: argparse.Namespace) -> int:
    graph = load_graph(args.graph) if args.graph else load_default_graph()
    runtime = GraphRuntime(
        graph=graph,
        store=_store(args.db),
        provider=_provider(args.provider),
    )
    try:
        state = await runtime.run(
            args.task,
            run_id=args.run_id,
            stop_after_steps=args.stop_after_steps,
        )
    except BudgetExceeded as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 2
    print(state.model_dump_json(indent=2))
    return 0 if state.status != "failed" else 1


async def _benchmark_command(args: argparse.Namespace) -> int:
    report = await run_graph_vs_loop_benchmark(
        input_path=args.input,
        provider=_provider(args.provider),
        output_path=args.output,
        loop_attempts=args.loop_attempts,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


async def _resume_command(args: argparse.Namespace) -> int:
    graph = load_graph(args.graph) if args.graph else load_default_graph()
    runtime = GraphRuntime(
        graph=graph,
        store=_store(args.db),
        provider=_provider(args.provider),
    )
    try:
        state = await runtime.run(run_id=args.run_id)
    except BudgetExceeded as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 2
    print(state.model_dump_json(indent=2))
    return 0 if state.status != "failed" else 1


def _trace_command(args: argparse.Namespace) -> int:
    events = _store(args.db).events(args.run_id)
    print(json.dumps(events, indent=2, sort_keys=True))
    return 0 if events else 1


def _export_command(args: argparse.Namespace) -> int:
    count = export_router_training_data(_store(args.db), args.output)
    print(json.dumps({"records": count, "output": str(Path(args.output).resolve())}, indent=2))
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
    graph = load_graph(args.graph) if args.graph else load_default_graph()
    print(
        json.dumps(
            {
                "name": graph.name,
                "version": graph.version,
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "terminals": sorted(graph.terminals),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graph-model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="start a graph execution")
    run.add_argument("--task", required=True)
    run.add_argument("--run-id")
    run.add_argument("--graph")
    run.add_argument("--db")
    run.add_argument("--provider", choices=("mock", "openai"), default="mock")
    run.add_argument("--stop-after-steps", type=int)

    benchmark = subparsers.add_parser("benchmark", help="compare graph execution with a retry loop")
    benchmark.add_argument("--input", required=True)
    benchmark.add_argument("--output")
    benchmark.add_argument("--provider", choices=("mock", "openai"), default="mock")
    benchmark.add_argument("--loop-attempts", type=int, default=3)

    resume = subparsers.add_parser("resume", help="resume a checkpointed execution")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--graph")
    resume.add_argument("--db")
    resume.add_argument("--provider", choices=("mock", "openai"), default="mock")

    trace = subparsers.add_parser("trace", help="inspect an execution trace")
    trace.add_argument("--run-id", required=True)
    trace.add_argument("--db")

    export = subparsers.add_parser("export", help="export router training records")
    export.add_argument("--output", required=True)
    export.add_argument("--db")

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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        code = asyncio.run(_run_command(args))
    elif args.command == "benchmark":
        code = asyncio.run(_benchmark_command(args))
    elif args.command == "resume":
        code = asyncio.run(_resume_command(args))
    elif args.command == "trace":
        code = _trace_command(args)
    elif args.command == "export":
        code = _export_command(args)
    elif args.command == "train-router":
        code = _train_router_command(args)
    elif args.command == "predict-route":
        code = _predict_route_command(args)
    elif args.command == "validate":
        code = _validate_command(args)
    else:
        parser.error(f"unknown command: {args.command}")
        return
    raise SystemExit(code)
