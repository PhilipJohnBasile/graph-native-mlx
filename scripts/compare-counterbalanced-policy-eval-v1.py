#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from graph_model.run_reporting import build_run_report
from graph_model.store import SQLiteRunStore

ARMS = ("static", "full")
PRIMARY_METRICS = ("tokens", "llm_calls", "repairs", "steps")
ALL_METRICS = (
    "steps",
    "repairs",
    "llm_calls",
    "tool_calls",
    "policy_calls",
    "tokens",
    "active_seconds",
)


def git(repo: str, *args: str) -> str:
    return subprocess.check_output(["git", "-C", repo, *args], text=True).strip()


def initial_patch_sha(state: Any) -> str | None:
    candidates: list[tuple[str, str]] = []
    for name, value in state.artifacts.items():
        if not str(name).startswith("candidate-proposal") or not isinstance(value, Mapping):
            continue
        patch = value.get("patch")
        if isinstance(patch, str):
            candidates.append((str(name), patch))
    if not candidates:
        return None
    _, patch = sorted(candidates)[0]
    return hashlib.sha256(patch.encode("utf-8")).hexdigest()


def prompt_audits(state: Any) -> list[dict[str, Any]]:
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


def telemetry(store: SQLiteRunStore, run_id: str) -> dict[str, Any]:
    sources: Counter[str] = Counter()
    choices: list[dict[str, Any]] = []
    for event in store.events(run_id):
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        result = payload.get("result")
        delta = result.get("delta") if isinstance(result, Mapping) else None
        decisions: list[tuple[str, Mapping[str, Any]]] = []
        if isinstance(delta, Mapping) and isinstance(delta.get("router"), Mapping):
            decisions.append(("route", delta["router"]))
        if isinstance(payload.get("stop_decision"), Mapping):
            decisions.append(("stop", payload["stop_decision"]))
        if isinstance(payload.get("edge_decision"), Mapping):
            decisions.append(("edge", payload["edge_decision"]))
        for kind, decision in decisions:
            source = str(decision.get("source") or "")
            if source:
                sources[source] += 1
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
        "decision_sources": dict(sorted(sources.items())),
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


def load_block_summaries(root: Path, arm: str) -> dict[str, Any]:
    summary_dir = root / arm / "block-summaries"
    paths = sorted(summary_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"{arm}: no block summaries found in {summary_dir}")
    values = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    controllers = [value.get("controller") or {} for value in values]
    providers = [value.get("provider") or {} for value in values]
    first_controller = controllers[0]
    first_provider = providers[0]
    if any(value != first_controller for value in controllers[1:]):
        raise SystemExit(f"{arm}: controller identity changed between counterbalanced blocks")
    if any(value != first_provider for value in providers[1:]):
        raise SystemExit(f"{arm}: provider identity changed between counterbalanced blocks")
    return {
        "paths": [str(path) for path in paths],
        "controller": first_controller,
        "provider": first_provider,
        "collector_errors": sum(
            int((value.get("status_counts") or {}).get("collector_error", 0))
            for value in values
        ),
    }


def arm_result(
    arm: str,
    *,
    cases: list[dict[str, Any]],
    root: Path,
) -> dict[str, Any]:
    db_path = root / arm / "runs.sqlite3"
    if not db_path.is_file():
        raise SystemExit(f"{arm}: missing database: {db_path}")
    summaries = load_block_summaries(root, arm)
    store = SQLiteRunStore(db_path)
    rows: list[dict[str, Any]] = []

    for case in cases:
        run_id = str(case["run_id"])
        state = store.load_run(run_id)
        if state is None:
            rows.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "pair_order": case["pair_order"],
                    "run_id": run_id,
                    "status": "missing",
                    "expected_status": case["expected_status"],
                    "correct": False,
                    "false_success": False,
                    "missing": True,
                }
            )
            continue

        report = build_run_report(store, run_id)
        tests = report["verification"].get("tests")
        review = report["verification"].get("review")
        patch = report["patch"]
        expected = str(case["expected_status"])
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
                "category": case["category"],
                "pair_order": case["pair_order"],
                "run_id": run_id,
                "expected_status": expected,
                "status": state.status,
                "current_node": state.current_node,
                "route": state.data.get("route"),
                "path": list(state.completed_nodes),
                "correct": correct,
                "false_success": false_success,
                "missing": False,
                "tests": tests.get("verdict") if isinstance(tests, Mapping) else None,
                "review": review.get("verdict") if isinstance(review, Mapping) else None,
                "initial_patch_sha256": initial_patch_sha(state),
                "verified_patch_sha256": patch.get("sha256"),
                "prompt_audits": prompt_audits(state),
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
        for key in ALL_METRICS
    }
    return {
        "controller": summaries["controller"],
        "provider": summaries["provider"],
        "block_summaries": summaries["paths"],
        "collector_errors": summaries["collector_errors"],
        "rows": rows,
        "correct": sum(bool(row.get("correct")) for row in rows),
        "false_successes": sum(bool(row.get("false_success")) for row in rows),
        "missing_runs": sum(bool(row.get("missing")) for row in rows),
        "raw_prompt_leaks": sum(
            any(item.get("raw_prompts_persisted") for item in row.get("prompt_audits", []))
            for row in rows
        ),
        "totals": totals,
    }


def percent_reduction(static: float, full: float) -> float | None:
    if static == 0:
        return None
    return ((static - full) / static) * 100.0


def grouped_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    result: dict[str, Any] = {}
    for name, values in sorted(groups.items()):
        totals = {
            metric: sum(float(row.get("metrics", {}).get(metric, 0)) for row in values)
            for metric in ALL_METRICS
        }
        result[name] = {
            "cases": len(values),
            "correct": sum(bool(row.get("correct")) for row in values),
            "false_successes": sum(bool(row.get("false_success")) for row in values),
            "totals": totals,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--canary-report", required=True)
    args = parser.parse_args()

    case_bundle = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = list(case_bundle["cases"])
    root = Path(args.root).expanduser().resolve()
    canary_report = json.loads(Path(args.canary_report).read_text(encoding="utf-8"))
    arms = {arm: arm_result(arm, cases=cases, root=root) for arm in ARMS}
    by_arm = {
        arm: {row["id"]: row for row in result["rows"]}
        for arm, result in arms.items()
    }

    paired_rows: list[dict[str, Any]] = []
    for case in cases:
        static = by_arm["static"][case["id"]]
        full = by_arm["full"][case["id"]]
        same_route = static.get("route") == full.get("route")
        same_path = static.get("path") == full.get("path")
        generation_prompt_equivalent = (
            static.get("prompt_audits") == full.get("prompt_audits")
        )
        paired_rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "pair_order": case["pair_order"],
                "expected_status": case["expected_status"],
                "static_correct": static.get("correct"),
                "full_correct": full.get("correct"),
                "static_status": static.get("status"),
                "full_status": full.get("status"),
                "static_route": static.get("route"),
                "full_route": full.get("route"),
                "route_changed": not same_route,
                "path_changed": not same_path,
                "generation_prompt_equivalent_when_same_route_and_path": (
                    generation_prompt_equivalent if same_route and same_path else None
                ),
                "static_metrics": static.get("metrics") or {},
                "full_metrics": full.get("metrics") or {},
                "metric_deltas": {
                    metric: float((full.get("metrics") or {}).get(metric, 0))
                    - float((static.get("metrics") or {}).get(metric, 0))
                    for metric in ALL_METRICS
                },
                "full_telemetry": full.get("telemetry") or {},
            }
        )

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

    static_categories = grouped_summary(arms["static"]["rows"], "category")
    full_categories = grouped_summary(arms["full"]["rows"], "category")
    category_comparison: dict[str, Any] = {}
    for category in sorted(set(static_categories) | set(full_categories)):
        static = static_categories.get(category, {})
        full = full_categories.get(category, {})
        category_comparison[category] = {
            "cases": static.get("cases", full.get("cases", 0)),
            "static_correct": static.get("correct", 0),
            "full_correct": full.get("correct", 0),
            "correct_delta": full.get("correct", 0) - static.get("correct", 0),
            "static_false_successes": static.get("false_successes", 0),
            "full_false_successes": full.get("false_successes", 0),
            "metric_reductions_percent": {
                metric: percent_reduction(
                    float((static.get("totals") or {}).get(metric, 0)),
                    float((full.get("totals") or {}).get(metric, 0)),
                )
                for metric in PRIMARY_METRICS
            },
        }

    order_comparison = {
        arm: grouped_summary(arms[arm]["rows"], "pair_order")
        for arm in ARMS
    }

    static_totals = arms["static"]["totals"]
    full_totals = arms["full"]["totals"]
    reductions = {
        metric: percent_reduction(float(static_totals[metric]), float(full_totals[metric]))
        for metric in ALL_METRICS
    }
    correct_gain = arms["full"]["correct"] - arms["static"]["correct"]
    route_changes = sum(bool(row["route_changed"]) for row in paired_rows)
    full_choice_changes = sum(
        int((row.get("full_telemetry") or {}).get("choice_changes", 0))
        for row in paired_rows
    )

    same_state_prompt_checks = [
        row["generation_prompt_equivalent_when_same_route_and_path"]
        for row in paired_rows
        if row["generation_prompt_equivalent_when_same_route_and_path"] is not None
    ]

    candidate_prefix = args.candidate_sha[:16]
    identity_checks = {
        "static_hardcoded": arms["static"]["controller"].get("policy") == "hardcoded-priors-only",
        "full_candidate_loaded": str(arms["full"]["controller"].get("policy", "")).startswith(
            f"mlx-policy:{candidate_prefix}"
        ),
        "full_route_scale_one": arms["full"]["controller"].get("route_policy_scale") == "1",
        "full_transition_scale_one": arms["full"]["controller"].get("transition_policy_scale") == "1",
        "both_skip_forced": all(
            arms[arm]["controller"].get("skip_forced_policy") == "true"
            for arm in ARMS
        ),
    }
    canary_anchor = {
        "gate_passed": canary_report.get("gate_passed") is True,
        "static_shadow_exact": (
            (canary_report.get("safety_checks") or {}).get("static_shadow_exact") is True
        ),
        "candidate_matches": (
            (canary_report.get("candidate") or {}).get("sha256") == args.candidate_sha
        ),
    }
    safety_checks = {
        **identity_checks,
        "canary_gate_passed": all(canary_anchor.values()),
        "exact_case_count": len(cases) == 24,
        "balanced_order": Counter(case["pair_order"] for case in cases)
        == {"static-first": 12, "full-first": 12},
        "baselines_validated": all(case.get("baseline_validated") is True for case in cases),
        "no_missing_runs": all(arms[arm]["missing_runs"] == 0 for arm in ARMS),
        "no_collector_errors": all(arms[arm]["collector_errors"] == 0 for arm in ARMS),
        "zero_false_successes": all(arms[arm]["false_successes"] == 0 for arm in ARMS),
        "full_no_completion_regression": arms["full"]["correct"] >= arms["static"]["correct"],
        "no_category_completion_regression": all(
            value["full_correct"] >= value["static_correct"]
            for value in category_comparison.values()
        ),
        "impossible_contracts_not_regressed": (
            category_comparison.get("impossible", {}).get("full_correct", 0)
            >= category_comparison.get("impossible", {}).get("static_correct", 0)
        ),
        "same_state_generation_prompts_match": all(same_state_prompt_checks),
        "no_raw_prompt_leaks": all(arms[arm]["raw_prompt_leaks"] == 0 for arm in ARMS),
        "repositories_clean": all(item["clean"] for item in repository_integrity),
        "repository_heads_unchanged": all(
            item["head_unchanged"] for item in repository_integrity
        ),
    }
    failed_safety = [name for name, passed in safety_checks.items() if not passed]
    safety_passed = not failed_safety

    primary_reductions = {
        metric: reductions[metric] for metric in PRIMARY_METRICS
    }
    material_cost_reduction = any(
        value is not None and value >= 10.0 for value in primary_reductions.values()
    )
    material_cost_regression = any(
        value is not None and value <= -10.0 for value in primary_reductions.values()
    )
    benefit_demonstrated = correct_gain > 0 or (
        correct_gain == 0 and material_cost_reduction and not material_cost_regression
    )

    if not safety_passed:
        promotion_status = "rejected-safety-regression"
    elif benefit_demonstrated:
        promotion_status = "eligible-for-guarded-shadow-rollout"
    else:
        promotion_status = "safe-but-no-demonstrated-benefit"

    report = {
        "format_version": 1,
        "purpose": "24-case counterbalanced static-versus-full policy evaluation",
        "candidate": {
            "sha256": args.candidate_sha,
            "id": args.candidate_sha[:12],
            "activation": "disabled",
            "promotion_status": promotion_status,
        },
        "canary_anchor": canary_anchor,
        "cases": cases,
        "arms": arms,
        "paired_results": paired_rows,
        "category_comparison": category_comparison,
        "order_comparison": order_comparison,
        "repository_integrity": repository_integrity,
        "aggregate": {
            "correct_gain": correct_gain,
            "route_changes": route_changes,
            "full_choice_changes": full_choice_changes,
            "metric_reductions_percent": reductions,
            "primary_metric_reductions_percent": primary_reductions,
            "material_cost_reduction": material_cost_reduction,
            "material_cost_regression": material_cost_regression,
            "benefit_demonstrated": benefit_demonstrated,
        },
        "safety_checks": safety_checks,
        "failed_safety_checks": failed_safety,
        "safety_passed": safety_passed,
        "interpretation": {
            "active_seconds": "exploratory because block order and model-cache state can affect latency",
            "promotion": (
                "the comparator never activates the candidate; eligibility still requires human review"
            ),
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Graph-Native MLX counterbalanced policy evaluation v1",
        "",
        f"- Candidate: `{args.candidate_sha[:12]}`",
        f"- Safety gate: **{'PASS' if safety_passed else 'FAIL'}**",
        f"- Promotion status: **{promotion_status}**",
        "- Candidate remains globally disabled",
        "",
        "## Aggregate outcomes",
        "",
        "| Arm | Correct | False successes | Tokens | LLM calls | Repairs | Steps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        result = arms[arm]
        lines.append(
            f"| {arm} | {result['correct']}/{len(cases)} | {result['false_successes']} | "
            f"{result['totals']['tokens']:.0f} | {result['totals']['llm_calls']:.0f} | "
            f"{result['totals']['repairs']:.0f} | {result['totals']['steps']:.0f} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate deltas",
            "",
            f"- Verified-outcome gain: `{correct_gain:+d}`",
            f"- Route changes: `{route_changes}`",
            f"- Learned meaningful choice changes: `{full_choice_changes}`",
        ]
    )
    for metric in PRIMARY_METRICS:
        value = primary_reductions[metric]
        text = "n/a" if value is None else f"{value:+.2f}%"
        lines.append(f"- {metric} reduction: `{text}`")

    lines.extend(
        [
            "",
            "## Category outcomes",
            "",
            "| Category | Cases | Static correct | Full correct | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for category, value in category_comparison.items():
        lines.append(
            f"| {category} | {value['cases']} | {value['static_correct']} | "
            f"{value['full_correct']} | {value['correct_delta']:+d} |"
        )

    lines.extend(["", "## Safety checks", ""])
    for name, passed in safety_checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The first 12 cases ran static before full; the other 12 ran full before static.",
            "- Generation prompts must match whenever both arms reach the same route and graph path.",
            "- Active-time differences are exploratory; correctness, false success, tokens, calls, and repairs are primary.",
            "- This report never changes `.graph-env` or activates the candidate.",
            "",
        ]
    )
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")

    print("Static correct:", f"{arms['static']['correct']}/{len(cases)}")
    print("Full correct:", f"{arms['full']['correct']}/{len(cases)}")
    print("False successes:", {arm: arms[arm]["false_successes"] for arm in ARMS})
    print("Correct gain:", correct_gain)
    print("Route changes:", route_changes)
    print("Full choice changes:", full_choice_changes)
    print("Primary reductions:", primary_reductions)
    print("Safety passed:", safety_passed)
    print("Benefit demonstrated:", benefit_demonstrated)
    print("Promotion status:", promotion_status)
    print("Report:", output_json)
    print()
    print("=" * 68)
    print(
        " COUNTERBALANCED POLICY EVALUATION SAFETY PASSED"
        if safety_passed
        else " COUNTERBALANCED POLICY EVALUATION SAFETY FAILED"
    )
    print("=" * 68)
    if failed_safety:
        print("Failed safety checks:")
        for name in failed_safety:
            print(" -", name)
        return 1
    print("The candidate remains disabled. Promotion status:", promotion_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
