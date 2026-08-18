from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from .conditions import ConditionError, evaluate_condition
from .models import Budget, EdgeSpec, GraphSpec, NodeKind, NodeResult, RunState
from .operators import ExecutionContext, OperatorRegistry
from .provider import MockProvider, ModelProvider
from .store import SQLiteRunStore


class GraphRuntimeError(RuntimeError):
    pass


class BudgetExceeded(GraphRuntimeError):
    pass


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _merge_dict(target: dict[str, Any], delta: dict[str, Any]) -> None:
    for key, value in delta.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_dict(target[key], value)
        else:
            target[key] = value


def _condition_context(state: RunState) -> dict[str, Any]:
    return {
        "data": state.data,
        "artifacts": state.artifacts,
        "attempts": state.attempts,
        "metrics": state.metrics.model_dump(),
        "step_count": state.step_count,
        "status": state.status,
        "True": True,
        "False": False,
        "None": None,
    }


def select_edge(graph: GraphSpec, state: RunState, node_id: str) -> EdgeSpec:
    context = _condition_context(state)
    exhausted: list[str] = []
    condition_errors: list[str] = []
    for edge in graph.outgoing(node_id):
        count = int(state.edge_counts.get(edge.key, 0))
        if count >= edge.max_traversals:
            exhausted.append(edge.key)
            continue
        try:
            matches = evaluate_condition(edge.when, context)
        except ConditionError as exc:
            condition_errors.append(f"{edge.key}: {exc}")
            continue
        if matches:
            return edge
    detail = []
    if exhausted:
        detail.append(f"exhausted={exhausted}")
    if condition_errors:
        detail.append(f"condition_errors={condition_errors}")
    raise GraphRuntimeError(
        f"no valid outgoing edge from {node_id!r}; " + ("; ".join(detail) or "no condition matched")
    )


def apply_node_result(
    *,
    graph: GraphSpec,
    state: RunState,
    node_id: str,
    node_kind: NodeKind,
    result: NodeResult,
    cached: bool,
    now: float | None = None,
) -> str | None:
    _merge_dict(state.data, result.delta)
    _merge_dict(state.artifacts, result.artifacts)
    state.attempts[node_id] = int(state.attempts.get(node_id, 0)) + 1
    state.completed_nodes.append(node_id)
    state.step_count += 1
    state.updated_at = time.time() if now is None else now

    if cached:
        state.metrics.cached_steps += 1
    elif node_kind == NodeKind.LLM:
        state.metrics.llm_calls += 1
    elif node_kind in {NodeKind.TOOL, NodeKind.VERIFIER}:
        state.metrics.tool_calls += 1
    state.metrics.prompt_tokens += result.prompt_tokens
    state.metrics.completion_tokens += result.completion_tokens
    state.metrics.elapsed_seconds = state.updated_at - state.started_at

    progress_key = result.progress_key or _canonical_hash(
        {"data": state.data, "artifacts": state.artifacts, "output": result.output}
    )
    if progress_key == state.last_progress_key:
        state.no_progress_count += 1
    else:
        state.no_progress_count = 0
    state.last_progress_key = progress_key

    if node_id in graph.terminals:
        state.output = result.output
        state.status = "completed" if node_id != "abort" else "failed"
        if state.status == "failed":
            state.error = "workflow reached the bounded abort node"
        return None

    edge = select_edge(graph, state, node_id)
    state.edge_counts[edge.key] = int(state.edge_counts.get(edge.key, 0)) + 1
    state.current_node = edge.target
    return edge.target


def budget_violations(
    state: RunState,
    *,
    now: float | None = None,
    next_kind: NodeKind | None = None,
) -> list[str]:
    current_time = time.time() if now is None else now
    elapsed = current_time - state.started_at
    metrics = state.metrics
    budget = state.budget
    violations: list[str] = []
    if state.step_count >= budget.max_steps:
        violations.append(f"steps {state.step_count}/{budget.max_steps}")
    if (
        next_kind == NodeKind.LLM
        and metrics.llm_calls >= budget.max_llm_calls
        and budget.max_llm_calls >= 0
    ):
        violations.append(f"llm_calls {metrics.llm_calls}/{budget.max_llm_calls}")
    if (
        next_kind in {NodeKind.TOOL, NodeKind.VERIFIER}
        and metrics.tool_calls >= budget.max_tool_calls
        and budget.max_tool_calls >= 0
    ):
        violations.append(f"tool_calls {metrics.tool_calls}/{budget.max_tool_calls}")
    if (
        next_kind == NodeKind.LLM
        and metrics.total_tokens >= budget.max_tokens
        and budget.max_tokens >= 0
    ):
        violations.append(f"tokens {metrics.total_tokens}/{budget.max_tokens}")
    if elapsed >= budget.max_seconds:
        violations.append(f"seconds {elapsed:.1f}/{budget.max_seconds:.1f}")
    if state.no_progress_count >= budget.max_no_progress_steps:
        violations.append(
            f"no_progress {state.no_progress_count}/{budget.max_no_progress_steps}"
        )
    return violations


class GraphRuntime:
    def __init__(
        self,
        *,
        graph: GraphSpec,
        store: SQLiteRunStore,
        registry: OperatorRegistry | None = None,
        provider: ModelProvider | None = None,
    ) -> None:
        self.graph = graph
        self.store = store
        self.registry = registry or OperatorRegistry.defaults()
        self.provider = provider or MockProvider()

    async def run(
        self,
        task: str | None = None,
        *,
        run_id: str | None = None,
        budget: Budget | None = None,
        initial_data: dict[str, Any] | None = None,
        stop_after_steps: int | None = None,
    ) -> RunState:
        state = self.store.load_run(run_id) if run_id else None
        if state is None:
            if task is None:
                raise ValueError("task is required for a new run")
            state = RunState.new(
                graph=self.graph,
                task=task,
                budget=budget,
                run_id=run_id,
                initial_data=initial_data,
            )
            runtime_metadata = state.data.setdefault("_runtime", {})
            if not isinstance(runtime_metadata, dict):
                raise ValueError("initial_data._runtime must be an object when provided")
            runtime_metadata["provider"] = self.provider.identity
            self.store.create_run(state)
        else:
            self._validate_resume(state)
            if state.status in {"completed", "failed"}:
                return state

        while state.status == "running":
            if stop_after_steps is not None and state.step_count >= stop_after_steps:
                return state

            node = self.graph.nodes[state.current_node]
            self._check_budget(state, node.kind)
            input_hash = _canonical_hash(
                {
                    "node": node.id,
                    "task": state.task,
                    "data": state.data,
                    "artifacts": state.artifacts,
                    "attempt": state.attempts.get(node.id, 0),
                    "graph_version": state.graph_version,
                }
            )
            cached_result = (
                self.store.get_step_result(state.run_id, node.id, input_hash)
                if node.cacheable
                else None
            )
            cached = cached_result is not None
            result = cached_result
            if result is None:
                operator = self.registry.get(node.operator)
                idempotency_key = f"{state.run_id}:{node.id}:{input_hash[:20]}"
                result = await operator(
                    ExecutionContext(
                        state=state.model_copy(deep=True),
                        node=node,
                        provider=self.provider,
                        idempotency_key=idempotency_key,
                    )
                )

            previous_node = node.id
            next_node = apply_node_result(
                graph=self.graph,
                state=state,
                node_id=node.id,
                node_kind=node.kind,
                result=result,
                cached=cached,
            )
            self.store.commit_step(
                state=state,
                node_id=previous_node,
                input_hash=input_hash,
                result=result,
                cached=cached,
                event_payload={"next_node": next_node, "status": state.status},
            )

            if state.status != "running":
                self.store.save_terminal_event(
                    state,
                    "run_completed" if state.status == "completed" else "run_failed",
                )
                return state

        return state

    def _validate_resume(self, state: RunState) -> None:
        if state.graph_name != self.graph.name or state.graph_version != self.graph.version:
            raise GraphRuntimeError(
                "run graph does not match loaded graph: "
                f"run={state.graph_name}@{state.graph_version}, "
                f"loaded={self.graph.name}@{self.graph.version}"
            )
        runtime_metadata = state.data.get("_runtime")
        expected_provider = (
            runtime_metadata.get("provider") if isinstance(runtime_metadata, dict) else None
        )
        if expected_provider and expected_provider != self.provider.identity:
            raise GraphRuntimeError(
                "resume provider does not match the provider used to start the run: "
                f"run={expected_provider!r}, loaded={self.provider.identity!r}"
            )

    def _check_budget(self, state: RunState, next_kind: NodeKind) -> None:
        violations = budget_violations(state, next_kind=next_kind)
        if violations:
            state.status = "failed"
            state.error = "budget exceeded: " + ", ".join(violations)
            state.updated_at = time.time()
            self.store.save_terminal_event(state, "budget_exceeded")
            raise BudgetExceeded(state.error)
