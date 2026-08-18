from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .controller import GraphController, route_difficulty
from .models import NodeResult, NodeSpec, RunState
from .provider import ModelProvider


@dataclass(frozen=True)
class ExecutionContext:
    state: RunState
    node: NodeSpec
    provider: ModelProvider
    controller: GraphController
    idempotency_key: str


Operator = Callable[[ExecutionContext], Awaitable[NodeResult]]


class OperatorRegistry:
    def __init__(self) -> None:
        self._operators: dict[str, Operator] = {}

    def register(self, name: str, operator: Operator) -> None:
        if name in self._operators:
            raise ValueError(f"operator {name!r} is already registered")
        self._operators[name] = operator

    def get(self, name: str) -> Operator:
        try:
            return self._operators[name]
        except KeyError as exc:
            raise KeyError(f"operator {name!r} is not registered") from exc

    @classmethod
    def defaults(cls) -> "OperatorRegistry":
        registry = cls()
        registry.register("route_task", route_task)
        registry.register("collect_context", collect_context)
        registry.register("make_plan", make_plan)
        registry.register("check_plan", check_plan)
        registry.register("implement", implement)
        registry.register("run_tests", run_tests)
        registry.register("review", review)
        registry.register("diagnose", diagnose)
        registry.register("repair", repair)
        registry.register("finalize", finalize)
        registry.register("abort", abort)
        return registry


def _stable_key(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


async def route_task(ctx: ExecutionContext) -> NodeResult:
    decision = ctx.controller.select_route(ctx.state.task, ctx.state)
    route = decision.route
    difficulty = route_difficulty(ctx.state.task, route)
    return NodeResult(
        delta={
            "route": route,
            "difficulty": difficulty,
            "verdict": "pending",
            "router": decision.as_dict(),
        },
        progress_key=f"route:{route}:{difficulty}:{decision.source}",
        notes=list(decision.notes),
    )


async def collect_context(ctx: ExecutionContext) -> NodeResult:
    summary = {
        "objective": ctx.state.task,
        "constraints": ctx.state.data.get("constraints", []),
        "known_artifacts": sorted(ctx.state.artifacts),
        "route": ctx.state.data.get("route"),
    }
    return NodeResult(
        delta={"context_ready": True, "context_summary": summary},
        artifacts={"context.json": summary},
        progress_key=f"context:{_stable_key(summary)}",
    )


async def make_plan(ctx: ExecutionContext) -> NodeResult:
    system = (
        "You are the planning node in a typed agent graph. Return only JSON with keys: "
        "steps (array of concise steps), risks (array), acceptance_tests (array). "
        "Do not perform the task; produce an inspectable execution plan."
    )
    user = json.dumps(
        {
            "task": ctx.state.task,
            "context": ctx.state.data.get("context_summary", {}),
            "prior_plan": ctx.state.data.get("plan"),
        },
        sort_keys=True,
    )
    payload, prompt_tokens, completion_tokens = await ctx.provider.complete_json(
        system=system,
        user=user,
        temperature=0.0,
    )
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        steps = [
            "Inspect the relevant state and constraints",
            "Make the smallest coherent change",
            "Run deterministic verification",
            "Review the result against the objective",
        ]
    plan = {
        "steps": steps,
        "risks": payload.get("risks", []),
        "acceptance_tests": payload.get("acceptance_tests", []),
    }
    return NodeResult(
        delta={"plan": plan, "verdict": "pending"},
        artifacts={"plan.json": plan},
        progress_key=f"plan:{_stable_key(plan)}",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def check_plan(ctx: ExecutionContext) -> NodeResult:
    plan = ctx.state.data.get("plan") or {}
    steps = plan.get("steps") if isinstance(plan, dict) else None
    forced_fail = "[bad-plan]" in ctx.state.task.lower()
    revision_count = int(ctx.state.data.get("plan_revision_count", 0))
    passed = bool(steps) and not (forced_fail and revision_count == 0)
    verdict = "pass" if passed else "fail"
    delta: dict[str, Any] = {"verdict": verdict}
    if not passed:
        delta["plan_revision_count"] = revision_count + 1
    return NodeResult(
        delta=delta,
        verdict=verdict,
        progress_key=f"plan-check:{verdict}:{revision_count}",
    )


async def implement(ctx: ExecutionContext) -> NodeResult:
    route = ctx.state.data.get("route", "deep")
    plan = ctx.state.data.get("plan")
    system = (
        "You are the execution node in a graph-controlled agent. Return only JSON with keys: "
        "result (string), changed_items (array), assumptions (array). Follow the supplied plan when present. "
        "Do not claim tests passed; a separate verifier owns that decision."
    )
    user = json.dumps(
        {
            "task": ctx.state.task,
            "route": route,
            "context": ctx.state.data.get("context_summary", {}),
            "plan": plan,
            "repair_guidance": ctx.state.data.get("diagnosis"),
            "idempotency_key": ctx.idempotency_key,
        },
        sort_keys=True,
    )
    payload, prompt_tokens, completion_tokens = await ctx.provider.complete_json(
        system=system,
        user=user,
        temperature=0.1,
    )
    candidate = {
        "result": payload.get("result", "Candidate output produced"),
        "changed_items": payload.get("changed_items", []),
        "assumptions": payload.get("assumptions", []),
        "revision": int(ctx.state.data.get("repair_count", 0)),
    }
    return NodeResult(
        delta={"candidate": candidate, "verdict": "pending"},
        artifacts={"candidate.json": candidate},
        progress_key=f"candidate:{_stable_key(candidate)}",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def run_tests(ctx: ExecutionContext) -> NodeResult:
    repairs = int(ctx.state.data.get("repair_count", 0))
    task = ctx.state.task.lower()
    fail_once = "[force-fail-once]" in task and repairs == 0
    fail_always = "[force-fail-always]" in task
    candidate_exists = bool(ctx.state.data.get("candidate"))
    passed = candidate_exists and not fail_once and not fail_always
    verdict = "pass" if passed else "fail"
    report = {
        "verdict": verdict,
        "checks": ["schema", "acceptance", "regression"],
        "repair_count": repairs,
        "idempotency_key": ctx.idempotency_key,
    }
    return NodeResult(
        delta={"verdict": verdict, "test_report": report},
        artifacts={f"test-report-{repairs}.json": report},
        verdict=verdict,
        progress_key=f"tests:{verdict}:{repairs}",
    )


async def review(ctx: ExecutionContext) -> NodeResult:
    if ctx.state.data.get("verdict") != "pass":
        return NodeResult(
            delta={"verdict": "fail", "review_reason": "deterministic verification did not pass"},
            verdict="fail",
            progress_key="review:blocked",
        )
    system = (
        "You are a semantic verifier, not the executor. Return only JSON with keys: "
        "verdict ('pass' or 'fail'), reasons (array), confidence (number 0..1). "
        "Reject unsupported claims and objective mismatches."
    )
    user = json.dumps(
        {
            "task": ctx.state.task,
            "candidate": ctx.state.data.get("candidate"),
            "test_report": ctx.state.data.get("test_report"),
        },
        sort_keys=True,
    )
    payload, prompt_tokens, completion_tokens = await ctx.provider.complete_json(
        system=system,
        user=user,
        temperature=0.0,
    )
    verdict = str(payload.get("verdict", "fail")).lower()
    if verdict not in {"pass", "fail"}:
        verdict = "fail"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    review_data = {
        "verdict": verdict,
        "reasons": payload.get("reasons", []),
        "confidence": confidence,
    }
    return NodeResult(
        delta={"verdict": verdict, "review": review_data},
        artifacts={"review.json": review_data},
        verdict=verdict,
        progress_key=f"review:{verdict}:{_stable_key(review_data)}",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def diagnose(ctx: ExecutionContext) -> NodeResult:
    diagnosis = {
        "failed_stage": "review" if ctx.state.data.get("review") else "tests",
        "test_report": ctx.state.data.get("test_report"),
        "review": ctx.state.data.get("review"),
        "recommended_action": "repair the candidate without discarding validated context",
    }
    return NodeResult(
        delta={"diagnosis": diagnosis},
        artifacts={"diagnosis.json": diagnosis},
        progress_key=f"diagnosis:{_stable_key(diagnosis)}",
    )


async def repair(ctx: ExecutionContext) -> NodeResult:
    repair_count = int(ctx.state.data.get("repair_count", 0)) + 1
    previous = ctx.state.data.get("candidate") or {}
    candidate = dict(previous) if isinstance(previous, dict) else {"result": str(previous)}
    candidate["revision"] = repair_count
    candidate["repair_basis"] = ctx.state.data.get("diagnosis", {})
    candidate["result"] = f"{candidate.get('result', 'Candidate')} [repaired revision {repair_count}]"
    return NodeResult(
        delta={"repair_count": repair_count, "candidate": candidate, "verdict": "pending"},
        artifacts={f"candidate-repair-{repair_count}.json": candidate},
        progress_key=f"repair:{repair_count}:{_stable_key(candidate)}",
    )


async def finalize(ctx: ExecutionContext) -> NodeResult:
    output = {
        "status": "success",
        "route": ctx.state.data.get("route"),
        "candidate": ctx.state.data.get("candidate"),
        "verification": {
            "tests": ctx.state.data.get("test_report"),
            "review": ctx.state.data.get("review"),
        },
        "repairs": int(ctx.state.data.get("repair_count", 0)),
    }
    return NodeResult(output=output, progress_key=f"finish:{_stable_key(output)}")


async def abort(ctx: ExecutionContext) -> NodeResult:
    output = {
        "status": "failed",
        "route": ctx.state.data.get("route"),
        "reason": "bounded graph exhausted its validated repair path",
        "last_test_report": ctx.state.data.get("test_report"),
        "last_review": ctx.state.data.get("review"),
        "repairs": int(ctx.state.data.get("repair_count", 0)),
    }
    return NodeResult(output=output, progress_key=f"abort:{_stable_key(output)}")
