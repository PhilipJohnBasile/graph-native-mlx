from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .controller import GraphController
from .graph_bundle import optimized_graph_version, write_graph_bundle
from .models import GraphSpec, RunState
from .optimizer import (
    GraphMutation,
    SetEdgeLimit,
    SetEdgePriority,
    SetNodeConfig,
    async_mcts_optimize,
)
from .provider import ModelProvider
from .runtime import BudgetExceeded, GraphRuntime, GraphRuntimeError
from .store import SQLiteRunStore
from .workspace import DEFAULT_ALLOWED_COMMANDS, RepositoryWorkspace, workspace_initial_data


@dataclass(frozen=True)
class ObjectiveWeights:
    quality: float = 1.0
    tokens: float = 0.02
    llm_calls: float = 0.02
    tool_calls: float = 0.01
    path_length: float = 0.01
    active_seconds: float = 0.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"objective weight {name} must be finite and non-negative")
        if self.quality <= 0:
            raise ValueError("quality weight must be positive")


@dataclass(frozen=True)
class SearchCase:
    case_id: str
    task: str
    expected_status: str = "completed"
    repo: str | None = None
    base_ref: str = "HEAD"
    test_commands: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphEvaluation:
    reward: float
    expected_outcome_rate: float
    average_tokens: float
    average_llm_calls: float
    average_tool_calls: float
    average_path_length: float
    average_active_seconds: float
    cases: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "expected_outcome_rate": self.expected_outcome_rate,
            "average_tokens": self.average_tokens,
            "average_llm_calls": self.average_llm_calls,
            "average_tool_calls": self.average_tool_calls,
            "average_path_length": self.average_path_length,
            "average_active_seconds": self.average_active_seconds,
            "cases": list(self.cases),
        }


ControllerFactory = Callable[[GraphSpec], GraphController]


def _case_identity(case: SearchCase) -> tuple[Any, ...]:
    return (
        case.task.strip(),
        str(Path(case.repo).expanduser().resolve(strict=False)) if case.repo else None,
        case.base_ref,
        case.test_commands,
    )


def validate_held_out_cases(
    train_cases: Sequence[SearchCase],
    validation_cases: Sequence[SearchCase],
) -> None:
    if not train_cases or not validation_cases:
        raise ValueError("training and validation case sets must both be non-empty")
    train_ids = {case.case_id for case in train_cases}
    validation_ids = {case.case_id for case in validation_cases}
    overlapping_ids = sorted(train_ids.intersection(validation_ids))
    if overlapping_ids:
        raise ValueError(
            f"held-out validation reuses training case IDs: {overlapping_ids[:5]}"
        )
    train_identities = {_case_identity(case) for case in train_cases}
    duplicate_tasks = [
        case.case_id for case in validation_cases if _case_identity(case) in train_identities
    ]
    if duplicate_tasks:
        raise ValueError(
            "held-out validation contains exact training tasks: "
            f"{sorted(duplicate_tasks)[:5]}"
        )


def validate_mutation_envelope(
    graph: GraphSpec,
    mutations: Sequence[GraphMutation],
) -> None:
    """Keep graph search inside the hard production safety envelope.

    Offline search may tune LLM temperatures, lower traversal caps, or reprioritize already-valid
    edges. It cannot add topology, raise retry limits, or mutate tool/verifier configuration.
    """

    edges = {edge.key: edge for edge in graph.edges}
    for mutation in mutations:
        if isinstance(mutation, SetNodeConfig):
            if mutation.node_id not in graph.nodes:
                raise ValueError(f"mutation references unknown node {mutation.node_id!r}")
            node = graph.nodes[mutation.node_id]
            if node.kind != "llm" or mutation.name != "temperature":
                raise ValueError(
                    "graph search may mutate only the temperature of existing LLM nodes"
                )
            if isinstance(mutation.value, bool) or not isinstance(
                mutation.value, (int, float)
            ):
                raise ValueError("LLM temperature mutation must be a JSON number")
            value = float(mutation.value)
            if not math.isfinite(value) or not 0.0 <= value <= 2.0:
                raise ValueError("LLM temperature mutation must be between 0.0 and 2.0")
        elif isinstance(mutation, SetEdgeLimit):
            edge = edges.get(mutation.edge_key)
            if edge is None:
                raise ValueError(f"mutation references unknown edge {mutation.edge_key!r}")
            if not 1 <= mutation.max_traversals <= edge.max_traversals:
                raise ValueError(
                    "graph search may only preserve or lower an existing edge traversal cap"
                )
        elif isinstance(mutation, SetEdgePriority):
            if mutation.edge_key not in edges:
                raise ValueError(f"mutation references unknown edge {mutation.edge_key!r}")
            if not -1_000 <= mutation.priority <= 1_000:
                raise ValueError("edge priority mutation must be between -1000 and 1000")
        else:
            raise ValueError(
                f"unsupported production graph mutation type {type(mutation).__name__!r}"
            )


def read_search_cases(path: str | Path) -> list[SearchCase]:
    cases: list[SearchCase] = []
    source = Path(path).expanduser()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid search JSONL at line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"search line {line_number} must be a JSON object")
            task = item.get("task")
            if not isinstance(task, str) or not task.strip():
                raise ValueError(f"search line {line_number} must contain a non-empty task")
            expected = item.get("expected_status", "completed")
            if expected not in {"completed", "failed"}:
                raise ValueError(
                    f"search line {line_number} has invalid expected_status {expected!r}"
                )
            raw_commands = item.get("test_commands") or []
            raw_tags = item.get("tags") or []
            if not isinstance(raw_commands, list) or not all(
                isinstance(value, str) for value in raw_commands
            ):
                raise ValueError(f"search line {line_number} test_commands must be strings")
            if not isinstance(raw_tags, list) or not all(isinstance(value, str) for value in raw_tags):
                raise ValueError(f"search line {line_number} tags must be strings")
            case_id = str(item.get("id") or item.get("run_id") or f"case-{line_number}")
            cases.append(
                SearchCase(
                    case_id=case_id,
                    task=task.strip(),
                    expected_status=str(expected),
                    repo=(str(item["repo"]) if item.get("repo") else None),
                    base_ref=str(item.get("base_ref") or "HEAD"),
                    test_commands=tuple(raw_commands),
                    tags=tuple(raw_tags),
                )
            )
    if not cases:
        raise ValueError("search input contains no cases")
    identifiers = [case.case_id for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("search case IDs must be unique")
    return cases


def default_graph_mutations(graph: GraphSpec) -> list[GraphMutation]:
    mutations: list[GraphMutation] = []
    node_temperature_space = {
        "plan": (0.0, (0.1,)),
        "implement": (0.1, (0.0, 0.2)),
        "review": (0.0, (0.1,)),
        "diagnose": (0.0, (0.1,)),
        "repair": (0.1, (0.0, 0.2)),
    }
    for node_id, (operator_default, alternatives) in node_temperature_space.items():
        if node_id not in graph.nodes:
            continue
        current = graph.nodes[node_id].config.get("temperature", operator_default)
        for value in alternatives:
            if current != value:
                mutations.append(SetNodeConfig(node_id, "temperature", value))

    edge_values = {
        "plan->plan_check:always": (1, 2),
        "apply->tests:data.verdict == 'pass'": (2, 3),
        "tests->review:data.verdict == 'pass'": (2, 3),
        "apply->diagnose:apply-repair": (1, 2),
        "tests->diagnose:test-repair": (1, 2),
        "review->diagnose:review-repair": (1, 2),
        "diagnose->repair:always": (1, 2),
        "repair->apply:always": (1, 2),
    }
    existing = {edge.key: edge for edge in graph.edges}
    for edge_key, values in edge_values.items():
        edge = existing.get(edge_key)
        if edge is None:
            continue
        for value in values:
            if edge.max_traversals != value:
                mutations.append(SetEdgeLimit(edge_key, value))
    return mutations


def read_mutations(path: str | Path, graph: GraphSpec) -> list[GraphMutation]:
    source = Path(path).expanduser()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid mutation configuration: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("mutation configuration must be a JSON array")
    mutations: list[GraphMutation] = []
    edge_keys = {edge.key for edge in graph.edges}
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"mutation {index} must be an object")
        kind = item.get("kind")
        if kind == "node_config":
            node_id = str(item.get("node") or "")
            name = str(item.get("name") or "")
            if node_id not in graph.nodes or not name:
                raise ValueError(f"mutation {index} references an invalid node or config name")
            mutations.append(SetNodeConfig(node_id, name, item.get("value")))
        elif kind == "edge_limit":
            edge_key = str(item.get("edge_key") or "")
            if edge_key not in edge_keys:
                raise ValueError(f"mutation {index} references unknown edge {edge_key!r}")
            max_traversals = item.get("max_traversals")
            if isinstance(max_traversals, bool) or not isinstance(max_traversals, int):
                raise ValueError(f"mutation {index} requires an integer max_traversals")
            mutations.append(SetEdgeLimit(edge_key, max_traversals))
        elif kind == "edge_priority":
            edge_key = str(item.get("edge_key") or "")
            if edge_key not in edge_keys:
                raise ValueError(f"mutation {index} references unknown edge {edge_key!r}")
            priority = item.get("priority")
            if isinstance(priority, bool) or not isinstance(priority, int):
                raise ValueError(f"mutation {index} requires an integer priority")
            mutations.append(SetEdgePriority(edge_key, priority))
        else:
            raise ValueError(f"mutation {index} has unsupported kind {kind!r}")
    if not mutations:
        raise ValueError("mutation configuration contains no mutations")
    validate_mutation_envelope(graph, mutations)
    return mutations


def _case_run_id(case: SearchCase, schema_hash: str, index: int) -> str:
    digest = hashlib.sha256(
        f"{case.case_id}:{schema_hash}:{index}".encode("utf-8")
    ).hexdigest()[:16]
    return f"graph-search-{index}-{digest}"


async def evaluate_graph(
    *,
    graph: GraphSpec,
    cases: Sequence[SearchCase],
    provider: ModelProvider,
    controller_factory: ControllerFactory,
    weights: ObjectiveWeights,
) -> GraphEvaluation:
    weights.validate()
    if not cases:
        raise ValueError("graph evaluation requires at least one case")
    from .mlx_native.graph_tables import graph_schema_hash

    schema_hash = graph_schema_hash(graph)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="graph-model-search-") as directory:
        root = Path(directory)
        store = SQLiteRunStore(root / "runs.sqlite3")
        runtime = GraphRuntime(
            graph=graph,
            store=store,
            provider=provider,
            controller=controller_factory(graph),
        )
        for index, case in enumerate(cases):
            initial_data = None
            if case.repo:
                initial_data = workspace_initial_data(
                    source_root=case.repo,
                    mode="worktree",
                    base_ref=case.base_ref,
                    workspace_home=root / "worktrees",
                    artifact_root=root / "artifacts",
                    test_commands=case.test_commands,
                    allowed_commands=DEFAULT_ALLOWED_COMMANDS,
                )
            error: str | None = None
            try:
                state = await runtime.run(
                    case.task,
                    run_id=_case_run_id(case, schema_hash, index),
                    initial_data=initial_data,
                )
            except (BudgetExceeded, GraphRuntimeError, RuntimeError, ValueError) as exc:
                state = store.load_run(_case_run_id(case, schema_hash, index))
                error = f"{type(exc).__name__}: {exc}"
                if state is None:
                    state = RunState.new(
                        graph=graph,
                        task=case.task,
                        run_id=_case_run_id(case, schema_hash, index),
                    )
                    state.status = "failed"
                    state.error = error
                elif state.status not in {"completed", "failed"}:
                    state.status = "failed"
                    state.error = error
            matches = state.status == case.expected_status
            row = {
                "id": case.case_id,
                "tags": list(case.tags),
                "status": state.status,
                "expected_status": case.expected_status,
                "matches_expected": matches,
                "error": error or state.error,
                "path": list(state.completed_nodes),
                "metrics": {
                    **state.metrics.model_dump(),
                    "total_tokens": state.metrics.total_tokens,
                    "path_length": len(state.completed_nodes),
                },
            }
            if case.repo:
                workspace = RepositoryWorkspace.from_state_data(
                    state.data, run_id=state.run_id
                )
                if workspace is not None:
                    try:
                        row["workspace_cleanup"] = workspace.cleanup_worktree(force=True)
                    except Exception as cleanup_exc:  # noqa: BLE001 - retain benchmark evidence
                        row["workspace_cleanup"] = {
                            "status": "failed",
                            "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                        }
            rows.append(row)

    count = len(rows)
    outcome_rate = sum(bool(row["matches_expected"]) for row in rows) / count
    average_tokens = sum(row["metrics"]["total_tokens"] for row in rows) / count
    average_llm = sum(row["metrics"]["llm_calls"] for row in rows) / count
    average_tool = sum(row["metrics"]["tool_calls"] for row in rows) / count
    average_path = sum(row["metrics"]["path_length"] for row in rows) / count
    average_seconds = sum(row["metrics"]["elapsed_seconds"] for row in rows) / count
    reward = (
        weights.quality * outcome_rate
        - weights.tokens * (average_tokens / 10_000.0)
        - weights.llm_calls * (average_llm / 10.0)
        - weights.tool_calls * (average_tool / 10.0)
        - weights.path_length * (average_path / 20.0)
        - weights.active_seconds * (average_seconds / 60.0)
    )
    return GraphEvaluation(
        reward=float(reward),
        expected_outcome_rate=float(outcome_rate),
        average_tokens=float(average_tokens),
        average_llm_calls=float(average_llm),
        average_tool_calls=float(average_tool),
        average_path_length=float(average_path),
        average_active_seconds=float(average_seconds),
        cases=tuple(rows),
    )


async def optimize_graph(
    *,
    base_graph: GraphSpec,
    train_cases: Sequence[SearchCase],
    validation_cases: Sequence[SearchCase] | None,
    provider: ModelProvider,
    controller_factory: ControllerFactory,
    output_dir: str | Path,
    mutations: list[GraphMutation] | None = None,
    weights: ObjectiveWeights = ObjectiveWeights(),
    iterations: int = 8,
    exploration: float = 1.2,
    seed: int = 42,
    min_validation_improvement: float = 0.0,
    allow_in_sample_promotion: bool = False,
) -> dict[str, Any]:
    if min_validation_improvement < 0:
        raise ValueError("min_validation_improvement must be non-negative")
    selected_mutations = mutations or default_graph_mutations(base_graph)
    if not selected_mutations:
        raise ValueError("no graph mutations are available")
    validate_mutation_envelope(base_graph, selected_mutations)
    if validation_cases is not None:
        validate_held_out_cases(train_cases, validation_cases)

    evaluation_by_schema: dict[str, GraphEvaluation] = {}
    from .mlx_native.graph_tables import graph_schema_hash

    async def evaluator(candidate: GraphSpec) -> float:
        schema = graph_schema_hash(candidate)
        evaluation = await evaluate_graph(
            graph=candidate,
            cases=train_cases,
            provider=provider,
            controller_factory=controller_factory,
            weights=weights,
        )
        evaluation_by_schema[schema] = evaluation
        return evaluation.reward

    result = await async_mcts_optimize(
        base_graph=base_graph,
        mutations=selected_mutations,
        evaluator=evaluator,
        iterations=iterations,
        exploration=exploration,
        seed=seed,
    )
    baseline_train = evaluation_by_schema[graph_schema_hash(base_graph)]
    candidate_train = evaluation_by_schema[graph_schema_hash(result.graph)]

    validation_source = validation_cases if validation_cases else train_cases
    independent_validation = bool(validation_cases)
    baseline_validation = await evaluate_graph(
        graph=base_graph,
        cases=validation_source,
        provider=provider,
        controller_factory=controller_factory,
        weights=weights,
    )
    candidate_validation = await evaluate_graph(
        graph=result.graph,
        cases=validation_source,
        provider=provider,
        controller_factory=controller_factory,
        weights=weights,
    )

    improvement = candidate_validation.reward - baseline_validation.reward
    quality_not_worse = (
        candidate_validation.expected_outcome_rate
        >= baseline_validation.expected_outcome_rate
    )
    promotion_allowed = independent_validation or allow_in_sample_promotion
    promoted = bool(
        promotion_allowed
        and quality_not_worse
        and improvement >= min_validation_improvement
        and result.mutation_path
    )
    status = "promoted" if promoted else ("candidate" if quality_not_worse else "rejected")
    versioned = optimized_graph_version(result.graph, mutation_path=result.mutation_path)

    report: dict[str, Any] = {
        "format_version": 1,
        "status": status,
        "independent_validation": independent_validation,
        "promotion_gate": {
            "promotion_allowed": promotion_allowed,
            "quality_not_worse": quality_not_worse,
            "minimum_improvement": min_validation_improvement,
            "actual_improvement": improvement,
            "has_mutation": bool(result.mutation_path),
        },
        "search": {
            "iterations_requested": iterations,
            "unique_evaluations": result.evaluations,
            "exploration": exploration,
            "seed": seed,
            "mutation_count": len(selected_mutations),
            "winning_mutation_path": list(result.mutation_path),
            "evaluated_candidates": [asdict(item) for item in result.evaluated_candidates],
        },
        "objective_weights": asdict(weights),
        "runtime_identity": {
            "provider": dict(provider.identity),
            "controller": dict(controller_factory(base_graph).identity),
        },
        "train": {
            "baseline": baseline_train.as_dict(),
            "candidate": candidate_train.as_dict(),
        },
        "validation": {
            "baseline": baseline_validation.as_dict(),
            "candidate": candidate_validation.as_dict(),
        },
    }
    bundle = write_graph_bundle(
        graph=versioned,
        output_dir=output_dir,
        benchmark_report=report,
        baseline_reward=baseline_validation.reward,
        candidate_reward=candidate_validation.reward,
        mutation_path=result.mutation_path,
        promotion_status=status,
        optimizer_config={
            "kind": "constrained-mcts",
            "iterations": iterations,
            "exploration": exploration,
            "seed": seed,
            "independent_validation": independent_validation,
            "objective_weights": asdict(weights),
        },
    )
    report["bundle"] = {
        "root": str(bundle.root),
        **bundle.identity,
        "promotion_status": status,
    }
    # The benchmark file inside the bundle intentionally excludes this self-reference. Write a
    # convenient outer summary next to the bundle for CLI consumers.
    summary_path = bundle.root / "optimization-summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    return report
