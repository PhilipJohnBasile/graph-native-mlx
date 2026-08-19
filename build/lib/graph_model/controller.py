from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from .models import EdgeSpec, GraphSpec, RunState
from .router_policy import load_router_cached

ROUTES: tuple[str, ...] = ("fast", "deep", "repair")
STOP_ACTIONS: tuple[str, ...] = ("continue", "repair", "finish", "abort")


@dataclass(frozen=True)
class RouteDecision:
    route: str
    confidence: float
    probabilities: dict[str, float]
    source: str
    rule_route: str
    notes: tuple[str, ...] = ()
    policy_metrics: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "source": self.source,
            "rule_route": self.rule_route,
            "notes": list(self.notes),
            "policy_metrics": self.policy_metrics,
        }


@dataclass(frozen=True)
class StopDecision:
    action: str
    confidence: float
    probabilities: dict[str, float]
    source: str
    preferred_target: str | None = None
    allowed_actions: tuple[str, ...] = ()
    policy_metrics: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "source": self.source,
            "preferred_target": self.preferred_target,
            "allowed_actions": list(self.allowed_actions),
            "policy_metrics": self.policy_metrics,
        }


@dataclass(frozen=True)
class EdgeDecision:
    edge: EdgeSpec
    confidence: float
    probabilities: dict[str, float]
    allowed_edge_keys: tuple[str, ...]
    source: str
    policy_metrics: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "edge_key": self.edge.key,
            "source_node": self.edge.source,
            "target_node": self.edge.target,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "allowed_edge_keys": list(self.allowed_edge_keys),
            "source": self.source,
            "policy_metrics": self.policy_metrics,
        }


@runtime_checkable
class GraphController(Protocol):
    @property
    def identity(self) -> dict[str, str]: ...

    def select_route(self, task: str, state: RunState) -> RouteDecision: ...

    def select_stop(
        self,
        *,
        graph: GraphSpec,
        state: RunState,
        node_id: str,
        candidates: Sequence[EdgeSpec] = (),
    ) -> StopDecision: ...

    def select_edge(
        self,
        *,
        graph: GraphSpec,
        state: RunState,
        node_id: str,
        candidates: Sequence[EdgeSpec],
        stop: StopDecision,
    ) -> EdgeDecision: ...


def rule_route(task: str) -> str:
    normalized = task.lower()
    deep_markers = (
        "architecture",
        "feature",
        "refactor",
        "migration",
        "design",
        "implement across",
        "multi-file",
        "security",
        "production",
        "benchmark",
    )
    repair_markers = ("ci", "failing", "failure", "bug", "error", "regression", "broken")
    fast_markers = ("typo", "rename", "one line", "small patch", "quick fix", "format")

    if any(marker in normalized for marker in repair_markers):
        return "repair"
    if any(marker in normalized for marker in deep_markers) or len(normalized) > 500:
        return "deep"
    if any(marker in normalized for marker in fast_markers) or len(normalized) < 140:
        return "fast"
    return "deep"


def route_difficulty(task: str, route: str) -> str:
    normalized = task.lower()
    deep_markers = (
        "architecture",
        "feature",
        "refactor",
        "migration",
        "design",
        "multi-file",
        "security",
        "production",
        "benchmark",
    )
    if route == "fast":
        return "low"
    if route == "repair":
        return "medium"
    return "high" if len(normalized) > 300 or any(m in normalized for m in deep_markers) else "medium"


def one_hot_probabilities(choice: str, values: Sequence[str]) -> dict[str, float]:
    return {value: 1.0 if value == choice else 0.0 for value in values}


class DeterministicGraphController:
    """Safe default controller constrained to validated graph choices."""

    @property
    def identity(self) -> dict[str, str]:
        return {"kind": "deterministic-graph-controller", "version": "1"}

    def select_route(self, task: str, state: RunState) -> RouteDecision:
        del state
        fallback = rule_route(task)
        route = fallback
        source = "rule"
        confidence = 1.0
        probabilities = one_hot_probabilities(route, ROUTES)
        notes = ("Rule router selected a validated subgraph.",)

        policy_path = os.getenv("GRAPH_MODEL_ROUTER_PATH")
        if policy_path:
            try:
                prediction = load_router_cached(policy_path).predict(task)
                threshold = float(os.getenv("GRAPH_MODEL_ROUTER_MIN_CONFIDENCE", "0.55"))
                probabilities = {
                    route_name: float(prediction.probabilities.get(route_name, 0.0))
                    for route_name in ROUTES
                }
                confidence = float(prediction.confidence)
                if prediction.route in ROUTES and confidence >= threshold:
                    route = prediction.route
                    source = "learned-hashed-router"
                    notes = ("Learned route policy selected among graph-validated paths.",)
                else:
                    notes = (
                        "Learned route confidence was below threshold; deterministic fallback used.",
                    )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                notes = (f"Learned router unavailable; deterministic fallback used: {exc}",)

        return RouteDecision(
            route=route,
            confidence=confidence,
            probabilities=probabilities,
            source=source,
            rule_route=fallback,
            notes=notes,
        )

    def select_stop(
        self,
        *,
        graph: GraphSpec,
        state: RunState,
        node_id: str,
        candidates: Sequence[EdgeSpec] = (),
    ) -> StopDecision:
        if node_id in graph.terminals:
            action = "abort" if node_id == "abort" else "finish"
        elif state.no_progress_count >= state.budget.max_no_progress_steps:
            action = "abort"
        else:
            targets = {edge.target for edge in candidates}
            if "finish" in targets:
                action = "finish"
            elif "abort" in targets and len(targets) == 1:
                action = "abort"
            elif targets.intersection({"diagnose", "repair"}):
                action = "repair"
            else:
                action = "continue"

        preferred_target = {
            "repair": "diagnose",
            "finish": "finish",
            "abort": "abort",
        }.get(action)
        return StopDecision(
            action=action,
            confidence=1.0,
            probabilities=one_hot_probabilities(action, STOP_ACTIONS),
            source="deterministic-stop-policy",
            preferred_target=preferred_target,
            allowed_actions=(action,),
        )

    def select_edge(
        self,
        *,
        graph: GraphSpec,
        state: RunState,
        node_id: str,
        candidates: Sequence[EdgeSpec],
        stop: StopDecision,
    ) -> EdgeDecision:
        del graph, state, node_id
        if not candidates:
            raise ValueError("select_edge requires at least one valid candidate")
        chosen = next(
            (edge for edge in candidates if stop.preferred_target == edge.target),
            candidates[0],
        )
        probabilities = {
            edge.key: 1.0 if edge.key == chosen.key else 0.0 for edge in candidates
        }
        return EdgeDecision(
            edge=chosen,
            confidence=1.0,
            probabilities=probabilities,
            allowed_edge_keys=tuple(edge.key for edge in candidates),
            source="deterministic-edge-policy",
        )
