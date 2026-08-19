from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .models import RunState
from .store import SQLiteRunStore


def _utc_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _task_preview(task: str, *, limit: int = 180) -> str:
    normalized = " ".join(task.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def summarize_run(state: RunState) -> dict[str, Any]:
    verified_patch = state.artifacts.get("verified-patch.json")
    failed_patch = state.artifacts.get("failed-patch.json")
    patch = verified_patch if isinstance(verified_patch, dict) else failed_patch
    route = state.data.get("route")
    return {
        "run_id": state.run_id,
        "status": state.status,
        "current_node": state.current_node,
        "route": route if isinstance(route, str) else None,
        "steps": state.step_count,
        "completed_nodes": list(state.completed_nodes),
        "repairs": int(state.data.get("repair_count", 0) or 0),
        "plan_revisions": int(state.data.get("plan_revision_count", 0) or 0),
        "llm_calls": state.metrics.llm_calls,
        "tool_calls": state.metrics.tool_calls,
        "policy_calls": state.metrics.policy_calls,
        "tokens": state.metrics.total_tokens,
        "active_seconds": state.metrics.elapsed_seconds,
        "started_at": _utc_timestamp(state.started_at),
        "updated_at": _utc_timestamp(state.updated_at),
        "task": _task_preview(state.task),
        "verified_patch": bool(isinstance(verified_patch, dict)),
        "patch_path": str(patch.get("path")) if isinstance(patch, dict) and patch.get("path") else None,
        "error": state.error,
    }


def list_run_summaries(
    store: SQLiteRunStore,
    *,
    limit: int = 20,
    status: str | None = None,
) -> list[dict[str, Any]]:
    return [summarize_run(state) for state in store.list_runs(limit=limit, status=status)]


def _decision_payloads(event: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        return ()
    decisions: list[Mapping[str, Any]] = []
    for key in ("edge_decision", "stop_decision"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            decisions.append(candidate)
    result = payload.get("result")
    if isinstance(result, Mapping):
        delta = result.get("delta")
        if isinstance(delta, Mapping):
            router = delta.get("router")
            if isinstance(router, Mapping):
                decisions.append(router)
    return tuple(decisions)


def _hidden_summary(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    artifacts: dict[str, Mapping[str, Any]] = {}
    sources: Counter[str] = Counter()
    cache_hits = 0
    decisions = 0
    for event in events:
        for decision in _decision_payloads(event):
            decisions += 1
            source = decision.get("source")
            if isinstance(source, str) and source:
                sources[source] += 1
            metrics = decision.get("policy_metrics")
            if not isinstance(metrics, Mapping):
                continue
            hidden = metrics.get("hidden_state")
            if not isinstance(hidden, Mapping):
                continue
            digest = hidden.get("sha256")
            if isinstance(digest, str) and digest:
                artifacts[digest] = hidden
            if metrics.get("hidden_state_cache_hit") is True:
                cache_hits += 1
    feature_sizes = sorted(
        {
            int(item["feature_size"])
            for item in artifacts.values()
            if isinstance(item.get("feature_size"), int)
        }
    )
    raw_sizes = sorted(
        {
            int(item["raw_hidden_size"])
            for item in artifacts.values()
            if isinstance(item.get("raw_hidden_size"), int)
        }
    )
    prompt_tokens = sum(
        int(item.get("prompt_tokens", 0) or 0) for item in artifacts.values()
    )
    return {
        "decision_count": decisions,
        "decision_sources": dict(sorted(sources.items())),
        "unique_artifacts": len(artifacts),
        "cache_hits": cache_hits,
        "feature_sizes": feature_sizes,
        "raw_hidden_sizes": raw_sizes,
        "unique_prompt_tokens": prompt_tokens,
        "artifacts": [
            {
                "sha256": digest,
                "path": str(item.get("path")) if item.get("path") else None,
                "feature_size": item.get("feature_size"),
                "raw_hidden_size": item.get("raw_hidden_size"),
                "prompt_tokens": item.get("prompt_tokens"),
                "layer_labels": item.get("layer_labels"),
                "pooling": item.get("pooling"),
                "model_fingerprint": item.get("model_fingerprint"),
            }
            for digest, item in sorted(artifacts.items())
        ],
    }


def _command_summary(command: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "command": command.get("command"),
        "exit_code": command.get("exit_code"),
        "passed": command.get("passed"),
        "timed_out": command.get("timed_out"),
        "duration_seconds": command.get("duration_seconds"),
    }


def _verification_summary(state: RunState) -> dict[str, Any]:
    test_report = state.data.get("test_report")
    review = state.data.get("review")
    tests: dict[str, Any] | None = None
    if isinstance(test_report, Mapping):
        commands = test_report.get("commands")
        tests = {
            "verdict": test_report.get("verdict"),
            "workspace_mutated": test_report.get("workspace_mutated"),
            "changed_files": list(test_report.get("changed_files") or []),
            "commands": [
                _command_summary(command)
                for command in commands
                if isinstance(command, Mapping)
            ]
            if isinstance(commands, list)
            else [],
        }
    semantic: dict[str, Any] | None = None
    if isinstance(review, Mapping):
        semantic = {
            "verdict": review.get("verdict"),
            "confidence": review.get("confidence"),
            "reasons": list(review.get("reasons") or []),
        }
    return {"tests": tests, "review": semantic}



def _terminal_output_summary(output: Any) -> Any:
    if output is None:
        return None
    if not isinstance(output, Mapping):
        text = str(output)
        return text if len(text) <= 500 else text[:499] + "…"
    candidate = output.get("candidate")
    candidate_summary = None
    if isinstance(candidate, Mapping):
        candidate_summary = {
            "result": candidate.get("result"),
            "changed_items": list(candidate.get("changed_items") or []),
            "revision": candidate.get("revision"),
        }
    verification = output.get("verification")
    verification_summary = None
    if isinstance(verification, Mapping):
        tests = verification.get("tests")
        review = verification.get("review")
        verification_summary = {
            "tests": tests.get("verdict") if isinstance(tests, Mapping) else None,
            "review": review.get("verdict") if isinstance(review, Mapping) else None,
        }
    workspace = output.get("workspace")
    workspace_summary = None
    if isinstance(workspace, Mapping):
        workspace_summary = {
            "mode": workspace.get("mode"),
            "promotion_required": workspace.get("promotion_required"),
        }
    return {
        "status": output.get("status"),
        "route": output.get("route"),
        "repairs": output.get("repairs"),
        "candidate": candidate_summary,
        "verification": verification_summary,
        "workspace": workspace_summary,
    }

def build_run_report(store: SQLiteRunStore, run_id: str) -> dict[str, Any]:
    state = store.load_run(run_id)
    if state is None:
        raise KeyError(f"run {run_id!r} was not found in {store.path}")
    events = store.events(run_id)
    verified_patch = state.artifacts.get("verified-patch.json")
    failed_patch = state.artifacts.get("failed-patch.json")
    patch = verified_patch if isinstance(verified_patch, Mapping) else failed_patch
    workspace = state.data.get("workspace")
    workspace_summary = None
    if isinstance(workspace, Mapping):
        workspace_summary = {
            "mode": workspace.get("mode"),
            "source_root": workspace.get("source_root"),
            "active_root": workspace.get("active_root"),
            "base_commit": workspace.get("base_commit"),
        }
    return {
        "run": summarize_run(state),
        "path": list(state.completed_nodes),
        "hidden_policy": _hidden_summary(events),
        "verification": _verification_summary(state),
        "patch": {
            "kind": "verified" if isinstance(verified_patch, Mapping) else (
                "failed" if isinstance(failed_patch, Mapping) else None
            ),
            "path": str(patch.get("path")) if isinstance(patch, Mapping) and patch.get("path") else None,
            "sha256": patch.get("sha256") if isinstance(patch, Mapping) else None,
            "bytes": patch.get("bytes") if isinstance(patch, Mapping) else None,
            "changed_files": list(patch.get("changed_files") or [])
            if isinstance(patch, Mapping)
            else [],
            "exists": bool(
                isinstance(patch, Mapping)
                and patch.get("path")
                and Path(str(patch["path"])).is_file()
            ),
        },
        "workspace": workspace_summary,
        "terminal": {
            "status": state.status,
            "error": state.error,
            "output": _terminal_output_summary(state.output),
        },
    }
