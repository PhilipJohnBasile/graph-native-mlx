from __future__ import annotations

import math
from typing import Iterable

from graph_model.models import GraphSpec, NodeKind, RunState

from .graph_tables import CompiledGraphTables

TASK_FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "log_chars",
    "log_words",
    "question_density",
    "repair_markers",
    "deep_markers",
    "fast_markers",
    "security_marker",
    "benchmark_marker",
    "side_effect_marker",
    "multi_file_marker",
    "test_marker",
    "production_marker",
    "code_marker",
    "failure_fixture_marker",
    "digit_density",
)

STATE_FEATURE_NAMES: tuple[str, ...] = (
    "step_budget_used",
    "llm_budget_used",
    "tool_budget_used",
    "token_budget_used",
    "time_budget_used",
    "no_progress_used",
    "repair_count",
    "plan_revision_count",
    "current_node_attempts",
    "exhausted_edge_fraction",
    "route_fast",
    "route_deep",
    "route_repair",
    "verdict_pending",
    "verdict_pass",
    "verdict_fail",
    "has_context",
    "has_plan",
    "has_workspace",
    "has_pending_patch",
    "has_candidate",
    "has_apply_report",
    "apply_pass",
    "apply_fail",
    "has_test_report",
    "test_mutated_workspace",
    "has_review",
    "has_diagnosis",
    "kind_router",
    "kind_llm",
    "kind_tool",
    "kind_verifier",
    "kind_final",
    "is_terminal",
)


def _count_markers(text: str, markers: Iterable[str], *, cap: int = 4) -> float:
    count = sum(1 for marker in markers if marker in text)
    return min(1.0, count / max(1, cap))


def _ratio(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 1.0 if value > 0 else 0.0
    return min(1.0, max(0.0, value / maximum))


def task_features(task: str) -> tuple[float, ...]:
    text = task.lower()
    words = text.split()
    chars = max(1, len(text))
    repair_markers = ("failing", "failure", "bug", "error", "regression", "broken", "ci")
    deep_markers = (
        "architecture",
        "feature",
        "refactor",
        "migration",
        "design",
        "multi-file",
        "production",
        "benchmark",
    )
    fast_markers = ("typo", "rename", "one line", "small patch", "quick fix", "format")
    side_effect_markers = ("deploy", "publish", "push", "delete", "email", "database", "github")
    code_markers = ("code", "repository", "repo", "function", "class", "api", "test", "patch")
    digit_count = sum(character.isdigit() for character in text)
    return (
        1.0,
        min(1.0, math.log1p(chars) / math.log1p(2_000)),
        min(1.0, math.log1p(len(words)) / math.log1p(400)),
        min(1.0, text.count("?") / 4.0),
        _count_markers(text, repair_markers),
        _count_markers(text, deep_markers),
        _count_markers(text, fast_markers),
        float(any(marker in text for marker in ("security", "permission", "secret", "auth"))),
        float("benchmark" in text or "compare" in text),
        float(any(marker in text for marker in side_effect_markers)),
        float("multi-file" in text or "across files" in text or "repository" in text),
        float("test" in text or "verify" in text or "acceptance" in text),
        float("production" in text or "enterprise" in text),
        float(any(marker in text for marker in code_markers)),
        float("[force-" in text or "[bad-plan]" in text),
        min(1.0, digit_count / chars * 10.0),
    )


def state_features(
    graph: GraphSpec,
    tables: CompiledGraphTables,
    state: RunState,
    node_id: str,
) -> tuple[float, ...]:
    route = str(state.data.get("route", ""))
    verdict = str(state.data.get("verdict", "pending"))
    node_kind = graph.nodes[node_id].kind
    edge_counts = state.edge_counts
    exhausted = sum(
        int(edge_counts.get(edge_key, 0)) >= max_traversals
        for edge_key, max_traversals in zip(
            tables.edge_keys,
            tables.edge_max_traversals,
            strict=True,
        )
    )
    node_index = tables.node_index[node_id]
    apply_report = state.data.get("apply_report")
    apply_verdict = (
        str(apply_report.get("verdict", "")) if isinstance(apply_report, dict) else ""
    )
    test_report = state.data.get("test_report")
    return (
        _ratio(state.step_count, state.budget.max_steps),
        _ratio(state.metrics.llm_calls, state.budget.max_llm_calls),
        _ratio(state.metrics.tool_calls, state.budget.max_tool_calls),
        _ratio(state.metrics.total_tokens, state.budget.max_tokens),
        _ratio(state.metrics.elapsed_seconds, state.budget.max_seconds),
        _ratio(state.no_progress_count, state.budget.max_no_progress_steps),
        min(1.0, int(state.data.get("repair_count", 0)) / 2.0),
        min(1.0, int(state.data.get("plan_revision_count", 0)) / 2.0),
        min(1.0, int(state.attempts.get(node_id, 0)) / 3.0),
        _ratio(exhausted, len(tables.edge_keys)),
        float(route == "fast"),
        float(route == "deep"),
        float(route == "repair"),
        float(verdict == "pending"),
        float(verdict == "pass"),
        float(verdict == "fail"),
        float(bool(state.data.get("context_ready"))),
        float(bool(state.data.get("plan"))),
        float(isinstance(state.data.get("workspace"), dict)),
        float(bool(state.data.get("pending_patch"))),
        float(bool(state.data.get("candidate"))),
        float(isinstance(apply_report, dict)),
        float(apply_verdict == "pass"),
        float(apply_verdict == "fail"),
        float(isinstance(test_report, dict)),
        float(
            isinstance(test_report, dict)
            and bool(test_report.get("test_mutated_workspace"))
        ),
        float(bool(state.data.get("review"))),
        float(bool(state.data.get("diagnosis"))),
        float(node_kind == NodeKind.ROUTER),
        float(node_kind == NodeKind.LLM),
        float(node_kind == NodeKind.TOOL),
        float(node_kind == NodeKind.VERIFIER),
        float(node_kind == NodeKind.FINAL),
        float(tables.terminal_mask[node_index]),
    )


def controller_feature_vector(
    graph: GraphSpec,
    tables: CompiledGraphTables,
    state: RunState,
    node_id: str,
) -> tuple[float, ...]:
    node_vector = tuple(float(candidate == node_id) for candidate in tables.node_ids)
    return task_features(state.task) + state_features(graph, tables, state, node_id) + node_vector


def controller_input_size(tables: CompiledGraphTables) -> int:
    return len(TASK_FEATURE_NAMES) + len(STATE_FEATURE_NAMES) + len(tables.node_ids)
