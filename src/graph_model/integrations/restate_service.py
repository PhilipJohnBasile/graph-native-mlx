"""Optional Restate production adapter.

Install with: uv sync --extra restate
Run with:     uv run python -m graph_model.integrations.restate_service

This adapter journals every node execution through ctx.run_typed. Successful upstream node results
are replayed rather than reissued after a crash. External tools still receive an idempotency key.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from pydantic import BaseModel, Field

from graph_model.controller import DeterministicGraphController
from graph_model.graph import load_default_graph
from graph_model.models import Budget, NodeResult, RunState
from graph_model.operators import ExecutionContext, OperatorRegistry
from graph_model.provider import OpenAICompatibleProvider
from graph_model.runtime import apply_node_result, budget_violations

try:
    import restate
except ImportError:  # pragma: no cover - optional dependency
    restate = None  # type: ignore[assignment]


class GraphRequest(BaseModel):
    task: str
    initial_data: dict[str, Any] = Field(default_factory=dict)
    budget: Budget = Field(default_factory=Budget)


async def _execute_node(
    *,
    node_id: str,
    state_payload: dict[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    graph = load_default_graph()
    state = RunState.model_validate(state_payload)
    node = graph.nodes[node_id]
    operator = OperatorRegistry.defaults().get(node.operator)
    result = await operator(
        ExecutionContext(
            state=state,
            node=node,
            provider=OpenAICompatibleProvider.from_env(),
            controller=DeterministicGraphController(),
            idempotency_key=idempotency_key,
        )
    )
    return result.model_dump()


if restate is not None:  # pragma: no branch
    graph_workflow = restate.Workflow("GraphNativeModel")

    @graph_workflow.main()
    async def run(ctx: "restate.WorkflowContext", req: GraphRequest) -> dict[str, Any]:
        graph = load_default_graph()
        started_at = float(await ctx.time()) / 1000.0
        state = RunState.new(
            graph=graph,
            task=req.task,
            budget=req.budget,
            run_id=ctx.key(),
            initial_data=req.initial_data,
        )
        state.started_at = started_at
        state.updated_at = started_at
        while state.status == "running":
            node = graph.nodes[state.current_node]
            durable_now = float(await ctx.time()) / 1000.0
            violations = budget_violations(state, now=durable_now, next_kind=node.kind)
            if violations:
                state.status = "failed"
                state.error = "budget exceeded: " + ", ".join(violations)
                state.updated_at = durable_now
                break
            invocation_key = f"{state.run_id}:{node.id}:{state.step_count}"
            payload = await ctx.run_typed(
                f"node:{state.step_count}:{node.id}",
                _execute_node,
                node_id=node.id,
                state_payload=state.model_dump(),
                idempotency_key=invocation_key,
            )
            result = NodeResult.model_validate(payload)
            apply_node_result(
                graph=graph,
                state=state,
                node_id=node.id,
                node_kind=node.kind,
                result=result,
                cached=False,
                now=float(await ctx.time()) / 1000.0,
            )
        return state.model_dump()

    app = restate.app([graph_workflow])
else:
    app = None


def main() -> None:
    if app is None:
        raise SystemExit("Install the optional Restate dependencies: uv sync --extra restate")
    import hypercorn.asyncio
    import hypercorn.config

    config = hypercorn.config.Config()
    config.bind = [os.getenv("GRAPH_RESTATE_BIND", "0.0.0.0:9080")]
    asyncio.run(hypercorn.asyncio.serve(app, config))


if __name__ == "__main__":
    main()
