#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from graph_model.run_reporting import build_run_report
from graph_model.store import SQLiteRunStore

ARMS = ("static", "shadow", "route-only", "full")


def git(repo: str, *args: str) -> str:
    return subprocess.check_output(["git", "-C", repo, *args], text=True).strip()


def initial_patch_sha(state) -> str | None:
    proposal = state.artifacts.get("candidate-proposal.json")
    if not isinstance(proposal, Mapping):
        return None
    patch = proposal.get("patch")
    if not isinstance(patch, str):
        return None
    import hashlib

    return hashlib.sha256(patch.encode("utf-8")).hexdigest()


def prompt_audits(state) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, value in sorted(state.artifacts.items()):
        if not str(name).startswith("prompt-audit-") or not isinstance(value, Mapping):
            continue
        records.append(
            {
                "artifact": str(name),
                "node_id": value.get("node_id"),
                "call_kind": value.get("call_kind"),
                "revision": value.get("revision"),
                "system_sha256": value.get("system_sha256"),
                "user_sha256": value.get("user_sha256"),
                "combined_sha256": value.get("combined_sha256"),
                "generation_seed": value.get("generation_seed"),
                "raw_prompts_persisted": bool(
                    value.get("raw_system_persisted") or value.get("raw_user_persisted")
                ),
            }
        )
    return records


def hidden_prompt_hashes(store: SQLiteRunStore, run_id: str) -> list[str]:
    values: list[str] = []
    for event in store.events(run_id):
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        result = payload.get("result")
        delta = result.get("delta") if isinstance(result, Mapping) else None
        decisions: list[Any] = []
        if isinstance(delta, Mapping) and isinstance(delta.get("router"), Mapping):
            decisions.append(delta["router"])
        decisions.extend(
            item
            for item in (payload.get("stop_decision"), payload.get("edge_decision"))
            if isinstance(item, Mapping)
        )
        seen_step: set[str] = set()
        for decision in decisions:
            metrics = decision.get("policy_metrics")
            hidden = metrics.get("hidden_state") if isinstance(metrics, Mapping) else None
            digest = hidden.get("prompt_sha256") if isinstance(hidden, Mapping) else None
            if isinstance(digest, str) and digest and digest not in seen_step:
                values.append(digest)
                seen_step.add(digest)
    return values


def telemetry(store: SQLiteRunStore, run_id: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    choices: list[dict[str, Any]] = []
    for event in store.events(run_id):
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        result = payload.get("result")
        delta = result.get("delta") if isinstance(result, Mapping) else None
        decisions: list[tuple[str, Any]] = []
        if isinstance(delta, Mapping) and isinstance(delta.get("router"), Mapping):
            decisions.append(("route", delta["router"]))
        if isinstance(payload.get("stop_decision"), Mapping):
            decisions.append(("stop", payload["stop_decision"]))
        if isinstance(payload.get("edge_decision"), Mapping):
            decisions.append(("edge", payload["edge_decision"]))
        for kind, decision in decisions:
            source = str(decision.get("source") or "")
            if source:
                counts[source] += 1
            metrics = decision.get("policy_metrics")
            if not isinstance(metrics, Mapping):
                continue
            choices.append(
                {
                    "node": event.get("node_id"),
                    "kind": kind,
                    "source": source,
                    "valid_choice_count": metrics.get("valid_choice_count"),
                    "policy_context_evaluated": metrics.get("policy_context_evaluated"),
                    "policy_evaluated": metrics.get("policy_evaluated"),
                    "policy_could_change_choice": metrics.get("policy_could_change_choice"),
                    "static_choice": metrics.get("static_choice"),
                    "learned_choice": metrics.get("learned_choice"),
                    "choice_changed": metrics.get("choice_changed"),
                }
            )
    return {
        "decision_sources": dict(sorted(counts.items())),
        "meaningful_choices": sum(
            item.get("policy_could_change_choice") is True for item in choices
        ),
        "choice_changes": sum(item.get("choice_changed") is True for item in choices),
        "forced_choices": sum(item.get("valid_choice_count") == 1 for item in choices),
        "skipped_contexts": sum(
            item.get("policy_context_evaluated") is False for item in choices
        ),
        "choices": choices,
    }


def arm_result(
    arm: str,
    *,
    cases: list[dict[str, Any]],
    root: Path,
) -> dict[str, Any]:
    summary_path = root / arm / "summary.json"
    db_path = root / arm / "runs.sqlite3"
    if not summary_path.is_file() or not db_path.is_file():
        raise SystemExit(f"{arm}: missing summary or database")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    store = SQLiteRunStore(db_path)
    rows: list[dict[str, Any]] = []
    for case in cases:
        run_id = str(case["run_id"])
        state = store.load_run(run_id)
        if state is None:
            rows.append(
                {
                    "id": case["id"],
                    "run_id": run_id,
                    "status": "missing",
                    "expected_status": case["expected_status"],
                    "correct": False,
                    "false_success": False,
                }
            )
            continue
        report = build_run_report(store, run_id)
        tests = report["verification"].get("tests")
        review = report["verification"].get("review")
        patch = report["patch"]
        expected = case["expected_status"]
        verified_completion = (
            state.status == "completed"
            and isinstance(tests, Mapping)
            and tests.get("verdict") == "pass"
            and tests.get("workspace_mutated") is False
            and isinstance(review, Mapping)
            and review.get("verdict") == "pass"
            and patch.get("kind") == "verified"
            and patch.get("exists") is True
        )
        bounded_failure = (
            state.status == "failed"
            and state.current_node == "abort"
            and patch.get("kind") != "verified"
        )
        correct = verified_completion if expected == "completed" else bounded_failure
        false_success = state.status == "completed" and (
            expected == "failed" or not verified_completion
        )
        rows.append(
            {
                "id": case["id"],
                "run_id": run_id,
                "expected_status": expected,
                "status": state.status,
                "current_node": state.current_node,
                "route": state.data.get("route"),
                "path": list(state.completed_nodes),
                "correct": correct,
                "false_success": false_success,
                "tests": tests.get("verdict") if isinstance(tests, Mapping) else None,
                "review": review.get("verdict") if isinstance(review, Mapping) else None,
                "initial_patch_sha256": initial_patch_sha(state),
                "verified_patch_sha256": patch.get("sha256"),
                "prompt_audits": prompt_audits(state),
                "hidden_prompt_hashes": hidden_prompt_hashes(store, run_id),
                "telemetry": telemetry(store, run_id),
                "metrics": {
                    "steps": state.step_count,
                    "repairs": int(state.data.get("repair_count", 0)),
                    "llm_calls": state.metrics.llm_calls,
                    "tool_calls": state.metrics.tool_calls,
                    "policy_calls": state.metrics.policy_calls,
                    "tokens": state.metrics.total_tokens,
                    "active_seconds": state.metrics.elapsed_seconds,
                },
                "error": state.error,
            }
        )
    totals = {
        key: sum(float(row["metrics"][key]) for row in rows if "metrics" in row)
        for key in (
            "steps",
            "repairs",
            "llm_calls",
            "tool_calls",
            "policy_calls",
            "tokens",
            "active_seconds",
        )
    }
    return {
        "controller": summary.get("controller") or {},
        "provider": summary.get("provider") or {},
        "status_counts": summary.get("status_counts") or {},
        "collector_errors": int((summary.get("status_counts") or {}).get("collector_error", 0)),
        "rows": rows,
        "correct": sum(bool(row.get("correct")) for row in rows),
        "false_successes": sum(bool(row.get("false_success")) for row in rows),
        "raw_prompt_leaks": sum(
            any(item.get("raw_prompts_persisted") for item in row.get("prompt_audits", []))
            for row in rows
        ),
        "totals": totals,
    }


def equivalence_row(static: Mapping[str, Any], shadow: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "status": static.get("status") == shadow.get("status"),
        "route": static.get("route") == shadow.get("route"),
        "path": static.get("path") == shadow.get("path"),
        "tests": static.get("tests") == shadow.get("tests"),
        "review": static.get("review") == shadow.get("review"),
        "initial_patch": static.get("initial_patch_sha256") == shadow.get("initial_patch_sha256"),
        "verified_patch": static.get("verified_patch_sha256") == shadow.get("verified_patch_sha256"),
        "generation_prompts": static.get("prompt_audits") == shadow.get("prompt_audits"),
        "hidden_prompts": static.get("hidden_prompt_hashes") == shadow.get("hidden_prompt_hashes"),
    }
    return {
        "id": static.get("id"),
        "checks": fields,
        "equivalent": all(fields.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--candidate-sha", required=True)
    args = parser.parse_args()

    case_bundle = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = list(case_bundle["cases"])
    root = Path(args.root).expanduser().resolve()
    arms = {arm: arm_result(arm, cases=cases, root=root) for arm in ARMS}
    static_by_id = {row["id"]: row for row in arms["static"]["rows"]}
    shadow_by_id = {row["id"]: row for row in arms["shadow"]["rows"]}
    equivalence = [
        equivalence_row(static_by_id[case["id"]], shadow_by_id[case["id"]])
        for case in cases
    ]

    repository_integrity: list[dict[str, Any]] = []
    for case in cases:
        repo = str(case["repo"])
        head = git(repo, "rev-parse", "HEAD")
        status = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
        repository_integrity.append(
            {
                "id": case["id"],
                "head_unchanged": head == case["base_commit"],
                "clean": not bool(status),
                "status": status,
            }
        )

    candidate_prefix = args.candidate_sha[:16]
    identities = {
        "static_hardcoded": arms["static"]["controller"].get("policy") == "hardcoded-priors-only",
        "shadow_loaded": str(arms["shadow"]["controller"].get("policy", "")).startswith(
            f"mlx-policy:{candidate_prefix}"
        ),
        "shadow_route_scale_zero": arms["shadow"]["controller"].get("route_policy_scale") == "0",
        "shadow_transition_scale_zero": arms["shadow"]["controller"].get("transition_policy_scale") == "0",
        "route_only_route_scale_one": arms["route-only"]["controller"].get("route_policy_scale") == "1",
        "route_only_transition_scale_zero": arms["route-only"]["controller"].get("transition_policy_scale") == "0",
        "full_scales_one": (
            arms["full"]["controller"].get("route_policy_scale") == "1"
            and arms["full"]["controller"].get("transition_policy_scale") == "1"
        ),
        "all_skip_forced": all(
            arm["controller"].get("skip_forced_policy") == "true"
            for arm in arms.values()
        ),
    }
    safety_checks = {
        **identities,
        "all_arms_no_collector_error": all(arm["collector_errors"] == 0 for arm in arms.values()),
        "all_arms_expected": all(arm["correct"] == len(cases) for arm in arms.values()),
        "all_arms_zero_false_success": all(arm["false_successes"] == 0 for arm in arms.values()),
        "static_shadow_exact": all(item["equivalent"] for item in equivalence),
        "no_raw_prompt_leaks": all(arm["raw_prompt_leaks"] == 0 for arm in arms.values()),
        "repositories_clean": all(item["clean"] for item in repository_integrity),
        "repository_heads_unchanged": all(item["head_unchanged"] for item in repository_integrity),
    }
    failed = [name for name, passed in safety_checks.items() if not passed]
    passed = not failed

    report = {
        "format_version": 2,
        "purpose": "causal paired four-arm policy canary",
        "candidate": {
            "sha256": args.candidate_sha,
            "id": args.candidate_sha[:12],
            "status": "disabled-pending-larger-evaluation",
        },
        "cases": cases,
        "arms": arms,
        "static_shadow_equivalence": equivalence,
        "repository_integrity": repository_integrity,
        "safety_checks": safety_checks,
        "failed_checks": failed,
        "gate_passed": passed,
        "interpretation": {
            "static_vs_shadow": "instrumentation effect",
            "shadow_vs_route_only": "route-head intervention",
            "route_only_vs_full": "transition-head intervention",
            "promotion": "a four-case pass authorizes a larger evaluation, not activation",
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Graph-Native MLX paired policy canary v2",
        "",
        f"- Candidate: `{args.candidate_sha[:12]}`",
        f"- Gate: **{'PASS' if passed else 'FAIL'}**",
        "- Policy remains globally disabled",
        "",
        "## Outcomes",
        "",
        "| Arm | Correct | False successes | Tokens | LLM calls | Repairs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm_name in ARMS:
        arm = arms[arm_name]
        lines.append(
            f"| {arm_name} | {arm['correct']}/{len(cases)} | "
            f"{arm['false_successes']} | {arm['totals']['tokens']:.0f} | "
            f"{arm['totals']['llm_calls']:.0f} | {arm['totals']['repairs']:.0f} |"
        )
    lines.extend(["", "## Static versus shadow", ""])
    for item in equivalence:
        lines.append(
            f"- {'PASS' if item['equivalent'] else 'FAIL'}: `{item['id']}`"
        )
    lines.extend(["", "## Safety checks", ""])
    for name, value in safety_checks.items():
        lines.append(f"- {'PASS' if value else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Static versus shadow isolates instrumentation and prompt contamination.",
            "- Shadow versus route-only isolates the route head.",
            "- Route-only versus full isolates transition-head influence.",
            "- A pass does not activate or promote the candidate.",
            "",
        ]
    )
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")

    print("Static correct:", f"{arms['static']['correct']}/{len(cases)}")
    print("Shadow correct:", f"{arms['shadow']['correct']}/{len(cases)}")
    print("Route-only correct:", f"{arms['route-only']['correct']}/{len(cases)}")
    print("Full correct:", f"{arms['full']['correct']}/{len(cases)}")
    print("Static/shadow exact:", all(item["equivalent"] for item in equivalence))
    print("False successes:", {arm: arms[arm]["false_successes"] for arm in ARMS})
    print("Report:", output_json)
    print()
    print("=" * 60)
    print(" PAIRED POLICY CANARY v2 PASSED" if passed else " PAIRED POLICY CANARY v2 FAILED")
    print("=" * 60)
    if failed:
        print("Failed checks:")
        for name in failed:
            print(" -", name)
        return 1
    print("The candidate remains disabled. Next gate: larger counterbalanced evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
