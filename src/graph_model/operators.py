from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .controller import GraphController, route_difficulty
from .models import NodeResult, NodeSpec, RunState
from .provider import ModelProvider
from .workspace import PatchError, RepositoryWorkspace, WorkspaceError


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
        registry.register("apply_candidate", apply_candidate)
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


def _workspace(ctx: ExecutionContext) -> RepositoryWorkspace | None:
    return RepositoryWorkspace.from_state_data(ctx.state.data, run_id=ctx.state.run_id)


def _node_temperature(ctx: ExecutionContext, default: float) -> float:
    raw = ctx.node.config.get("temperature", default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"node {ctx.node.id!r} has invalid temperature {raw!r}"
        ) from exc
    if not 0.0 <= value <= 2.0:
        raise ValueError(
            f"node {ctx.node.id!r} temperature must be in [0, 2], found {value}"
        )
    return value


def _truncate_middle(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    head = max(1, limit // 3)
    tail = max(1, limit - head)
    return text[:head] + "\n... evidence truncated ...\n" + text[-tail:]


def _compact_test_report(value: Any, *, max_output_chars: int = 60_000) -> Any:
    if not isinstance(value, dict):
        return value
    compact = {key: item for key, item in value.items() if key != "commands"}
    commands = value.get("commands")
    if not isinstance(commands, list):
        return compact
    selected: list[dict[str, Any]] = []
    remaining = max_output_chars
    for raw in commands[:8]:
        if not isinstance(raw, dict):
            continue
        command = dict(raw)
        per_stream = max(1_000, min(12_000, remaining // 2))
        stdout = _truncate_middle(command.get("stdout", ""), per_stream)
        stderr = _truncate_middle(command.get("stderr", ""), per_stream)
        command["stdout"] = stdout
        command["stderr"] = stderr
        remaining -= len(stdout) + len(stderr)
        selected.append(command)
        if remaining <= 0:
            break
    compact["commands"] = selected
    compact["commands_truncated"] = len(selected) < len(commands)
    return compact


def _prompt_evidence_limits(workspace: RepositoryWorkspace) -> tuple[int, int]:
    total = workspace.config.max_context_bytes
    return min(80_000, max(16_000, total // 2)), min(
        60_000, max(12_000, total // 3)
    )


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
    workspace = _workspace(ctx)
    if workspace is None:
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

    prepared = await asyncio.to_thread(workspace.ensure_prepared)
    summary = await asyncio.to_thread(workspace.collect_context, ctx.state.task)
    summary["objective"] = ctx.state.task
    summary["route"] = ctx.state.data.get("route")
    return NodeResult(
        delta={
            "workspace": prepared,
            "context_ready": True,
            "context_summary": summary,
        },
        artifacts={"context.json": summary},
        progress_key=f"workspace-context:{summary['workspace_fingerprint']}",
        notes=(
            [
                "Repository mode executes tests with local user permissions; command shape, time, paths, and output are bounded, but this is not a hostile-code sandbox."
            ]
        ),
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
        temperature=_node_temperature(ctx, 0.0),
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
    workspace = _workspace(ctx)
    if workspace is None:
        return await _implement_simulated(ctx)

    evidence_limit, _ = _prompt_evidence_limits(workspace)
    evidence = await asyncio.to_thread(
        workspace.current_evidence, max_diff_bytes=evidence_limit
    )
    system = (
        "You are the repository patch proposal node in a graph-controlled coding agent. "
        "Return a GRAPH_PATCH_V1 raw-text envelope, not prose. Use this exact shape: "
        "GRAPH_PATCH_V1, then GRAPH_PATCH_META_BEGIN, then one compact JSON metadata "
        "object with keys summary (string), assumptions (array of strings), and "
        "no_changes_needed (boolean), then GRAPH_PATCH_META_END, then "
        "GRAPH_PATCH_DIFF_BEGIN, then the raw unified diff, then GRAPH_PATCH_DIFF_END. "
        "Do not JSON-escape the diff. The diff must contain complete 'diff --git a/... "
        "b/...' headers and repository-relative paths. Leave the diff block empty only "
        "when no_changes_needed is true. Produce text-file changes only. Do not use shell "
        "commands, modify .git or secret files, or claim tests passed. A separate apply "
        "node and verifier own those decisions. Prefer the smallest coherent patch. "
        "If an external API forces JSON mode, one strict JSON object with summary, patch, "
        "assumptions, and no_changes_needed is also accepted."
    )
    user = json.dumps(
        {
            "task": ctx.state.task,
            "route": ctx.state.data.get("route", "deep"),
            "context": ctx.state.data.get("context_summary", {}),
            "plan": ctx.state.data.get("plan"),
            "current_workspace": evidence,
            "idempotency_key": ctx.idempotency_key,
        },
        sort_keys=True,
    )
    payload, prompt_tokens, completion_tokens = await ctx.provider.complete_patch(
        system=system,
        user=user,
        temperature=_node_temperature(ctx, 0.1),
    )
    proposal = _patch_proposal(payload, revision=int(ctx.state.data.get("repair_count", 0)))
    return NodeResult(
        delta={
            "pending_patch": proposal,
            "candidate_proposal": proposal,
            "verdict": "pending",
        },
        artifacts={"candidate-proposal.json": proposal},
        progress_key=f"patch-proposal:{_stable_key(proposal)}",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def _implement_simulated(ctx: ExecutionContext) -> NodeResult:
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
        temperature=_node_temperature(ctx, 0.1),
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


async def apply_candidate(ctx: ExecutionContext) -> NodeResult:
    workspace = _workspace(ctx)
    if workspace is None:
        report = {
            "verdict": "pass",
            "mode": "simulation",
            "idempotency_key": ctx.idempotency_key,
        }
        return NodeResult(
            delta={"verdict": "pass", "apply_report": report},
            artifacts={"apply-report.json": report},
            verdict="pass",
            progress_key=f"apply:simulation:{ctx.state.data.get('repair_count', 0)}",
        )

    proposal = ctx.state.data.get("pending_patch")
    if not isinstance(proposal, dict):
        error = "no pending patch proposal is present"
        report = {"verdict": "fail", "error": error}
        return NodeResult(
            delta={
                "verdict": "fail",
                "apply_report": report,
                "pending_patch": None,
                "test_report": None,
                "review": None,
            },
            artifacts={"apply-report.json": report},
            verdict="fail",
            progress_key=f"apply-fail:{_stable_key(report)}",
        )

    try:
        application = await asyncio.to_thread(
            workspace.apply_patch,
            str(proposal.get("patch") or ""),
            idempotency_key=ctx.idempotency_key,
            no_changes_needed=bool(proposal.get("no_changes_needed", False)),
        )
        evidence = await asyncio.to_thread(workspace.current_evidence)
    except (PatchError, WorkspaceError, OSError) as exc:
        report = {
            "verdict": "fail",
            "error": f"{type(exc).__name__}: {exc}",
            "proposal_sha256": hashlib.sha256(
                str(proposal.get("patch") or "").encode("utf-8")
            ).hexdigest(),
            "repair_count": int(ctx.state.data.get("repair_count", 0)),
        }
        return NodeResult(
            delta={
                "verdict": "fail",
                "apply_report": report,
                "pending_patch": None,
                "test_report": None,
                "review": None,
            },
            artifacts={
                f"apply-report-{ctx.state.data.get('repair_count', 0)}.json": report
            },
            verdict="fail",
            progress_key=f"apply-fail:{_stable_key(report)}",
        )

    report = {"verdict": "pass", **application.as_dict()}
    candidate = {
        "result": proposal.get("summary", "Repository patch applied"),
        "changed_items": evidence.get("changed_files", []),
        "assumptions": proposal.get("assumptions", []),
        "revision": int(ctx.state.data.get("repair_count", 0)),
        "patch_sha256": application.patch_sha256,
        "patch_artifact": application.patch_artifact,
        "workspace_fingerprint": evidence.get("workspace_fingerprint"),
        "diff_stat": evidence.get("diff_stat", ""),
    }
    return NodeResult(
        delta={
            "verdict": "pass",
            "apply_report": report,
            "candidate": candidate,
            "workspace_evidence": evidence,
            "pending_patch": None,
            "test_report": None,
            "review": None,
            "diagnosis": None,
        },
        artifacts={
            f"apply-report-{ctx.state.data.get('repair_count', 0)}.json": report,
            f"candidate-{ctx.state.data.get('repair_count', 0)}.json": candidate,
        },
        verdict="pass",
        progress_key=f"apply-pass:{application.after_fingerprint}",
    )


async def run_tests(ctx: ExecutionContext) -> NodeResult:
    workspace = _workspace(ctx)
    if workspace is None:
        return await _run_tests_simulated(ctx)

    if ctx.state.data.get("verdict") != "pass" or not ctx.state.data.get("candidate"):
        report = {
            "verdict": "fail",
            "reason": "candidate patch was not applied successfully",
            "apply_report": ctx.state.data.get("apply_report"),
        }
    else:
        before_files = set(await asyncio.to_thread(workspace.changed_files))
        report = await asyncio.to_thread(workspace.run_tests)
        before_fingerprint = str(report.get("workspace_fingerprint_before") or "")
        after_fingerprint = str(report.get("workspace_fingerprint") or "")
        after_files = set(report.get("changed_files", []))
        unexpected = sorted(after_files.difference(before_files))
        if unexpected:
            report["verdict"] = "fail"
            report["unexpected_test_side_effects"] = unexpected
        if bool(report.get("workspace_mutated")) or after_fingerprint != before_fingerprint:
            report["verdict"] = "fail"
            report["test_mutated_workspace"] = True
            report["workspace_fingerprint_before_tests"] = before_fingerprint
            report["workspace_fingerprint_after_tests"] = after_fingerprint
    verdict = str(report.get("verdict", "fail"))
    repairs = int(ctx.state.data.get("repair_count", 0))
    return NodeResult(
        delta={"verdict": verdict, "test_report": report},
        artifacts={f"test-report-{repairs}.json": report},
        verdict=verdict,
        progress_key=f"workspace-tests:{verdict}:{_stable_key(report)}",
    )


async def _run_tests_simulated(ctx: ExecutionContext) -> NodeResult:
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

    workspace = _workspace(ctx)
    current_evidence: dict[str, Any] | None = None
    if workspace is not None:
        evidence_limit, _ = _prompt_evidence_limits(workspace)
        current_evidence = await asyncio.to_thread(
            workspace.current_evidence, max_diff_bytes=evidence_limit
        )
        test_fingerprint = (ctx.state.data.get("test_report") or {}).get(
            "workspace_fingerprint"
        )
        if test_fingerprint and test_fingerprint != current_evidence.get(
            "workspace_fingerprint"
        ):
            reason = "workspace changed after deterministic verification"
            return NodeResult(
                delta={"verdict": "fail", "review_reason": reason},
                verdict="fail",
                progress_key=f"review:stale:{_stable_key(current_evidence)}",
            )

    system = (
        "You are a semantic verifier, not the executor. Return only JSON with keys: "
        "verdict ('pass' or 'fail'), reasons (array), confidence (number 0..1). "
        "Reject unsupported claims, objective mismatches, incomplete patches, and changes that "
        "only make tests pass without satisfying the requested behavior."
    )
    user = json.dumps(
        {
            "task": ctx.state.task,
            "candidate": ctx.state.data.get("candidate"),
            "test_report": _compact_test_report(ctx.state.data.get("test_report")),
            "workspace_evidence": current_evidence,
        },
        sort_keys=True,
    )
    payload, prompt_tokens, completion_tokens = await ctx.provider.complete_json(
        system=system,
        user=user,
        temperature=_node_temperature(ctx, 0.0),
    )
    verdict = str(payload.get("verdict", "fail")).lower()
    if verdict not in {"pass", "fail"}:
        verdict = "fail"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    if verdict == "pass" and confidence < 0.5:
        verdict = "fail"
    review_data = {
        "verdict": verdict,
        "reasons": payload.get("reasons", []),
        "confidence": confidence,
        "workspace_fingerprint": (
            current_evidence.get("workspace_fingerprint") if current_evidence else None
        ),
    }
    return NodeResult(
        delta={"verdict": verdict, "review": review_data},
        artifacts={f"review-{ctx.state.data.get('repair_count', 0)}.json": review_data},
        verdict=verdict,
        progress_key=f"review:{verdict}:{_stable_key(review_data)}",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def diagnose(ctx: ExecutionContext) -> NodeResult:
    workspace = _workspace(ctx)
    failed_stage = (
        "review"
        if ctx.state.data.get("review")
        and ctx.state.data.get("review", {}).get("verdict") == "fail"
        else "apply"
        if ctx.state.data.get("apply_report", {}).get("verdict") == "fail"
        else "tests"
    )
    if workspace is None:
        diagnosis = {
            "failed_stage": failed_stage,
            "test_report": ctx.state.data.get("test_report"),
            "review": ctx.state.data.get("review"),
            "apply_report": ctx.state.data.get("apply_report"),
            "recommended_action": "repair the candidate without discarding validated context",
        }
        return NodeResult(
            delta={"diagnosis": diagnosis},
            artifacts={"diagnosis.json": diagnosis},
            progress_key=f"diagnosis:{_stable_key(diagnosis)}",
        )

    evidence_limit, changed_limit = _prompt_evidence_limits(workspace)
    evidence = await asyncio.to_thread(
        workspace.current_evidence, max_diff_bytes=evidence_limit
    )
    changed_context = await asyncio.to_thread(
        workspace.read_changed_file_context, max_bytes=changed_limit
    )
    system = (
        "You are the failure diagnosis node in a graph-controlled coding agent. Return only JSON "
        "with keys: root_causes (array), repair_steps (array), files_to_change (array), "
        "evidence (array). Diagnose from the exact patch, command output, and semantic review. "
        "Do not propose restarting the whole task and do not claim a repair has already worked."
    )
    user = json.dumps(
        {
            "task": ctx.state.task,
            "failed_stage": failed_stage,
            "candidate_proposal": ctx.state.data.get("candidate_proposal"),
            "apply_report": ctx.state.data.get("apply_report"),
            "test_report": _compact_test_report(ctx.state.data.get("test_report")),
            "review": ctx.state.data.get("review"),
            "workspace_evidence": evidence,
            "changed_file_context": changed_context,
        },
        sort_keys=True,
    )
    payload, prompt_tokens, completion_tokens = await ctx.provider.complete_json(
        system=system,
        user=user,
        temperature=_node_temperature(ctx, 0.0),
    )
    diagnosis = {
        "failed_stage": failed_stage,
        "root_causes": payload.get("root_causes", []),
        "repair_steps": payload.get("repair_steps", []),
        "files_to_change": payload.get("files_to_change", []),
        "evidence": payload.get("evidence", []),
        "workspace_fingerprint": evidence.get("workspace_fingerprint"),
    }
    return NodeResult(
        delta={"diagnosis": diagnosis},
        artifacts={f"diagnosis-{ctx.state.data.get('repair_count', 0)}.json": diagnosis},
        progress_key=f"diagnosis:{_stable_key(diagnosis)}",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def repair(ctx: ExecutionContext) -> NodeResult:
    workspace = _workspace(ctx)
    repair_count = int(ctx.state.data.get("repair_count", 0)) + 1
    if workspace is None:
        previous = ctx.state.data.get("candidate") or {}
        candidate = dict(previous) if isinstance(previous, dict) else {"result": str(previous)}
        candidate["revision"] = repair_count
        candidate["repair_basis"] = ctx.state.data.get("diagnosis", {})
        candidate["result"] = (
            f"{candidate.get('result', 'Candidate')} [repaired revision {repair_count}]"
        )
        return NodeResult(
            delta={"repair_count": repair_count, "candidate": candidate, "verdict": "pending"},
            artifacts={f"candidate-repair-{repair_count}.json": candidate},
            progress_key=f"repair:{repair_count}:{_stable_key(candidate)}",
        )

    evidence_limit, changed_limit = _prompt_evidence_limits(workspace)
    evidence = await asyncio.to_thread(
        workspace.current_evidence, max_diff_bytes=evidence_limit
    )
    changed_context = await asyncio.to_thread(
        workspace.read_changed_file_context, max_bytes=changed_limit
    )
    system = (
        "You are the repair patch node in a bounded graph-controlled coding agent. "
        "Return a GRAPH_PATCH_V1 raw-text envelope with compact JSON metadata between "
        "GRAPH_PATCH_META_BEGIN and GRAPH_PATCH_META_END, followed by the raw unified "
        "diff between GRAPH_PATCH_DIFF_BEGIN and GRAPH_PATCH_DIFF_END. Metadata keys are "
        "summary (string), assumptions (array of strings), and no_changes_needed "
        "(boolean). Do not JSON-escape the diff. The diff must be complete against the "
        "CURRENT working tree and contain 'diff --git a/... b/...' headers. Leave the "
        "diff block empty only when no_changes_needed is true. Repair only the diagnosed "
        "failure. Do not repeat already-applied hunks, modify .git or secret files, or "
        "claim tests passed. If an external API forces JSON mode, one strict JSON object "
        "with summary, patch, assumptions, and no_changes_needed is also accepted."
    )
    user = json.dumps(
        {
            "task": ctx.state.task,
            "repair_number": repair_count,
            "diagnosis": ctx.state.data.get("diagnosis"),
            "candidate_proposal": ctx.state.data.get("candidate_proposal"),
            "apply_report": ctx.state.data.get("apply_report"),
            "test_report": _compact_test_report(ctx.state.data.get("test_report")),
            "review": ctx.state.data.get("review"),
            "current_workspace": evidence,
            "changed_file_context": changed_context,
            "original_context": ctx.state.data.get("context_summary", {}),
        },
        sort_keys=True,
    )
    payload, prompt_tokens, completion_tokens = await ctx.provider.complete_patch(
        system=system,
        user=user,
        temperature=_node_temperature(ctx, 0.1),
    )
    proposal = _patch_proposal(payload, revision=repair_count)
    proposal["repair_basis"] = ctx.state.data.get("diagnosis", {})
    return NodeResult(
        delta={
            "repair_count": repair_count,
            "pending_patch": proposal,
            "candidate_proposal": proposal,
            "verdict": "pending",
        },
        artifacts={f"repair-proposal-{repair_count}.json": proposal},
        progress_key=f"repair-proposal:{repair_count}:{_stable_key(proposal)}",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def finalize(ctx: ExecutionContext) -> NodeResult:
    workspace = _workspace(ctx)
    patch_artifact: dict[str, Any] | None = None
    if workspace is not None:
        patch_artifact = await asyncio.to_thread(workspace.export_patch, "verified.patch")
    output = {
        "status": "success",
        "route": ctx.state.data.get("route"),
        "candidate": ctx.state.data.get("candidate"),
        "verification": {
            "tests": ctx.state.data.get("test_report"),
            "review": ctx.state.data.get("review"),
        },
        "repairs": int(ctx.state.data.get("repair_count", 0)),
        "workspace": (
            {
                "mode": workspace.config.mode.value,
                "source_root": str(workspace.source_root),
                "active_root": str(workspace.active_root),
                "base_commit": workspace.config.base_commit,
                "verified_patch": patch_artifact,
                "promotion_required": workspace.config.mode.value == "worktree",
            }
            if workspace is not None
            else None
        ),
    }
    artifacts = {"verified-patch.json": patch_artifact} if patch_artifact else {}
    return NodeResult(
        output=output,
        artifacts=artifacts,
        progress_key=f"finish:{_stable_key(output)}",
    )


async def abort(ctx: ExecutionContext) -> NodeResult:
    workspace = _workspace(ctx)
    failed_patch: dict[str, Any] | None = None
    if workspace is not None:
        try:
            failed_patch = await asyncio.to_thread(workspace.export_patch, "failed.patch")
        except WorkspaceError:
            failed_patch = None
    output = {
        "status": "failed",
        "route": ctx.state.data.get("route"),
        "reason": "bounded graph exhausted its validated repair path",
        "last_apply_report": ctx.state.data.get("apply_report"),
        "last_test_report": ctx.state.data.get("test_report"),
        "last_review": ctx.state.data.get("review"),
        "repairs": int(ctx.state.data.get("repair_count", 0)),
        "workspace": (
            {
                "mode": workspace.config.mode.value,
                "source_root": str(workspace.source_root),
                "active_root": str(workspace.active_root),
                "base_commit": workspace.config.base_commit,
                "failed_patch": failed_patch,
            }
            if workspace is not None and workspace.config.active_root
            else None
        ),
    }
    artifacts = {"failed-patch.json": failed_patch} if failed_patch else {}
    return NodeResult(
        output=output,
        artifacts=artifacts,
        progress_key=f"abort:{_stable_key(output)}",
    )


def _patch_proposal(payload: dict[str, Any], *, revision: int) -> dict[str, Any]:
    assumptions = payload.get("assumptions", [])
    if not isinstance(assumptions, list):
        assumptions = [str(assumptions)]
    return {
        "summary": str(payload.get("summary") or payload.get("result") or "Patch proposed"),
        "patch": str(payload.get("patch") or ""),
        "assumptions": [str(value) for value in assumptions],
        "no_changes_needed": bool(payload.get("no_changes_needed", False)),
        "revision": revision,
    }
