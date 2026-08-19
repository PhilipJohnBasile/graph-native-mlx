from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .controller import DeterministicGraphController
from .graph import load_default_graph
from .models import RunMetrics, RunState
from .operators import ExecutionContext, OperatorRegistry
from .provider import ModelProvider


@dataclass(frozen=True)
class LoopResult:
    status: str
    output: dict[str, Any]
    metrics: RunMetrics
    path: tuple[str, ...]
    attempts: int


def _merge(target: dict[str, Any], delta: dict[str, Any]) -> None:
    for key, value in delta.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


class RalphLoopBaseline:
    """A bounded version of the plan-execute-check-retry-from-the-top pattern.

    The cap exists so the benchmark itself is safe. Unlike GraphRuntime, this baseline has no
    durable checkpoints, edge-level state, or reuse guarantee for successful upstream work.
    """

    def __init__(self, *, provider: ModelProvider, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.provider = provider
        self.max_attempts = max_attempts
        self.registry = OperatorRegistry.defaults()
        self.graph = load_default_graph()
        self.controller = DeterministicGraphController()

    async def run(self, task: str) -> LoopResult:
        state = RunState.new(graph=self.graph, task=task)
        path: list[str] = []
        metrics = RunMetrics()
        started_at = time.time()

        # Give the baseline the same explicit task context, but not graph-controlled path reuse.
        await self._invoke(state, "intake", path, metrics)
        await self._invoke(state, "context", path, metrics)

        for attempt in range(1, self.max_attempts + 1):
            # Ralph-style retry: rebuild the plan and execution from the top on every attempt.
            state.data.pop("plan", None)
            state.data["verdict"] = "pending"
            await self._invoke(state, "plan", path, metrics)
            await self._invoke(state, "implement", path, metrics)
            await self._invoke(state, "apply", path, metrics)
            test_result = await self._invoke(state, "tests", path, metrics)

            if test_result.verdict == "pass":
                review_result = await self._invoke(state, "review", path, metrics)
                if review_result.verdict == "pass":
                    final_result = await self._invoke(state, "finish", path, metrics)
                    metrics.elapsed_seconds = time.time() - started_at
                    return LoopResult(
                        status="completed",
                        output=final_result.output or {},
                        metrics=metrics,
                        path=tuple(path),
                        attempts=attempt,
                    )

            await self._invoke(state, "diagnose", path, metrics)
            state.data["repair_count"] = attempt

        abort_result = await self._invoke(state, "abort", path, metrics)
        metrics.elapsed_seconds = time.time() - started_at
        return LoopResult(
            status="failed",
            output=abort_result.output or {},
            metrics=metrics,
            path=tuple(path),
            attempts=self.max_attempts,
        )

    async def _invoke(
        self,
        state: RunState,
        node_id: str,
        path: list[str],
        metrics: RunMetrics,
    ):
        node = self.graph.nodes[node_id]
        operator = self.registry.get(node.operator)
        invocation_index = len(path)
        result = await operator(
            ExecutionContext(
                state=state.model_copy(deep=True),
                node=node,
                provider=self.provider,
                controller=self.controller,
                idempotency_key=f"loop:{node_id}:{invocation_index}",
            )
        )
        _merge(state.data, result.delta)
        _merge(state.artifacts, result.artifacts)
        path.append(node_id)
        if node.kind.value == "llm":
            metrics.llm_calls += 1
        elif node.kind.value in {"tool", "verifier"}:
            metrics.tool_calls += 1
        metrics.prompt_tokens += result.prompt_tokens
        metrics.completion_tokens += result.completion_tokens
        return result
