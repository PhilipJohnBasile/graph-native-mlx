from __future__ import annotations

import os
from pathlib import Path
from threading import RLock
from typing import Protocol, Sequence

from graph_model.controller import (
    ROUTES,
    STOP_ACTIONS,
    EdgeDecision,
    RouteDecision,
    StopDecision,
    rule_route,
)
from graph_model.models import EdgeSpec, GraphSpec, RunState

from .decision import DecisionBackend, MLXDecisionBackend
from .features import controller_feature_vector
from .graph_tables import CompiledGraphTables, compile_graph
from .policy import MLXPolicyRunner, PolicyOutput


class PolicyPredictor(Protocol):
    @property
    def identity(self) -> str: ...

    def predict(self, features: Sequence[float]) -> PolicyOutput: ...


def _add_logits(base: list[float], residual: Sequence[float], scale: float) -> list[float]:
    if len(base) != len(residual):
        raise ValueError(
            f"policy residual length {len(residual)} does not match decision size {len(base)}"
        )
    return [value + scale * float(delta) for value, delta in zip(base, residual, strict=True)]


def _policy_metrics(output: PolicyOutput | None) -> dict[str, object]:
    if output is None:
        return {}
    return {
        "success_value": output.success_value,
        "predicted_cost": {
            "tokens": output.cost[0],
            "latency": output.cost[1],
            "tool_calls": output.cost[2],
        },
    }




def _resolve_tables(
    graph: GraphSpec,
    tables: CompiledGraphTables | None,
) -> CompiledGraphTables:
    if tables is not None:
        return tables
    try:
        from .generated_coding_graph import COMPILED_GRAPH

        COMPILED_GRAPH.validate_graph(graph)
        return COMPILED_GRAPH
    except ValueError:
        # Custom graphs can be compiled in memory for development. Production deployments should
        # run `graph-model compile-graph` and import the generated module instead.
        return compile_graph(graph)

def _action_for_target(target: str) -> str:
    if target == "abort":
        return "abort"
    if target == "finish":
        return "finish"
    if target in {"diagnose", "repair"}:
        return "repair"
    return "continue"


class MLXGraphController:
    """Graph controller whose choice masks and softmax/argmax execute in MLX.

    Structural validity is never learned. The graph compiler creates fixed node/edge tables;
    runtime predicates reduce them further to the current candidate set. Optional learned heads
    add residual logits only before the hard mask is applied.
    """

    def __init__(
        self,
        *,
        graph: GraphSpec,
        tables: CompiledGraphTables | None = None,
        decision_backend: DecisionBackend | None = None,
        policy: PolicyPredictor | None = None,
        policy_scale: float = 1.0,
    ) -> None:
        if policy_scale < 0:
            raise ValueError("policy_scale must be non-negative")
        self.graph = graph
        self.tables = _resolve_tables(graph, tables)
        self.tables.validate_graph(graph)
        self.backend = decision_backend or MLXDecisionBackend()
        self.policy = policy
        self.policy_scale = float(policy_scale)
        self._cache_lock = RLock()
        self._cached_policy_key: tuple[object, ...] | None = None
        self._cached_policy_output: PolicyOutput | None = None

    @classmethod
    def from_env(
        cls,
        graph: GraphSpec,
        *,
        tables: CompiledGraphTables | None = None,
        decision_backend: DecisionBackend | None = None,
    ) -> "MLXGraphController":
        compiled = _resolve_tables(graph, tables)
        weights = os.getenv("GRAPH_MODEL_MLX_POLICY_WEIGHTS")
        config = os.getenv("GRAPH_MODEL_MLX_POLICY_CONFIG")
        policy: MLXPolicyRunner | None = None
        if weights:
            weights_path = Path(weights).expanduser()
            config_path = (
                Path(config).expanduser()
                if config
                else weights_path.with_name("graph_policy.json")
            )
            policy = MLXPolicyRunner(
                tables=compiled,
                weights_path=weights_path,
                config_path=config_path,
            )
        return cls(
            graph=graph,
            tables=compiled,
            decision_backend=decision_backend,
            policy=policy,
            policy_scale=float(os.getenv("GRAPH_MODEL_MLX_POLICY_SCALE", "1.0")),
        )

    @property
    def identity(self) -> dict[str, str]:
        return {
            "kind": "mlx-graph-controller",
            "version": "1",
            "graph_schema_hash": self.tables.schema_hash,
            "decision_backend": self.backend.identity,
            "policy": self.policy.identity if self.policy is not None else "hardcoded-priors-only",
            "policy_scale": f"{self.policy_scale:g}",
        }

    def _policy_output(self, state: RunState, node_id: str) -> PolicyOutput | None:
        if self.policy is None:
            return None
        key = (
            id(state),
            state.run_id,
            state.step_count,
            node_id,
            state.updated_at,
            state.data.get("route"),
            state.data.get("verdict"),
            state.data.get("repair_count"),
            state.data.get("plan_revision_count"),
        )
        with self._cache_lock:
            if key != self._cached_policy_key:
                features = controller_feature_vector(
                    self.graph,
                    self.tables,
                    state,
                    node_id,
                )
                self._cached_policy_output = self.policy.predict(features)
                self._cached_policy_key = key
            return self._cached_policy_output

    def _decision_metrics(
        self,
        state: RunState,
        node_id: str,
        output: PolicyOutput | None,
    ) -> dict[str, object]:
        metrics = {
            "graph_schema_hash": self.tables.schema_hash,
            "feature_vector": list(
                controller_feature_vector(
                    self.graph,
                    self.tables,
                    state,
                    node_id,
                )
            ),
        }
        metrics.update(_policy_metrics(output))
        return metrics

    def select_route(self, task: str, state: RunState) -> RouteDecision:
        fallback = rule_route(task)
        logits = [0.0] * len(ROUTES)
        logits[ROUTES.index(fallback)] = 8.0
        policy_output = self._policy_output(state, self.graph.start)
        if policy_output is not None:
            logits = _add_logits(logits, policy_output.route_logits, self.policy_scale)
        choice = self.backend.masked_softmax_argmax(logits, [True] * len(ROUTES))
        route = ROUTES[choice.selected_index]
        probabilities = {
            name: float(choice.probabilities[index]) for index, name in enumerate(ROUTES)
        }
        source = "mlx-policy-residual" if policy_output is not None else "mlx-hardcoded-route"
        notes = (
            "MLX selected among the three validated route IDs; graph structure remains external.",
        )
        return RouteDecision(
            route=route,
            confidence=probabilities[route],
            probabilities=probabilities,
            source=source,
            rule_route=fallback,
            notes=notes,
            policy_metrics=self._decision_metrics(state, self.graph.start, policy_output),
        )

    def select_stop(
        self,
        *,
        graph: GraphSpec,
        state: RunState,
        node_id: str,
        candidates: Sequence[EdgeSpec] = (),
    ) -> StopDecision:
        del graph
        allowed_actions = {_action_for_target(edge.target) for edge in candidates}
        if not allowed_actions:
            allowed_actions = {"abort" if node_id == "abort" else "finish"}
        if state.no_progress_count >= state.budget.max_no_progress_steps:
            allowed_actions.add("abort")

        action_mask = [action in allowed_actions for action in STOP_ACTIONS]
        # The prior follows the highest-priority candidate. A trained policy can rank any action
        # represented by a currently valid candidate, but cannot unmask another action.
        prior_action = (
            _action_for_target(candidates[0].target)
            if candidates
            else ("abort" if node_id == "abort" else "finish")
        )
        logits = [0.0] * len(STOP_ACTIONS)
        logits[STOP_ACTIONS.index(prior_action)] = 8.0
        policy_output = self._policy_output(state, node_id)
        if policy_output is not None:
            logits = _add_logits(logits, policy_output.stop_logits, self.policy_scale)
        choice = self.backend.masked_softmax_argmax(logits, action_mask)
        action = STOP_ACTIONS[choice.selected_index]
        probabilities = {
            name: float(choice.probabilities[index])
            for index, name in enumerate(STOP_ACTIONS)
        }
        preferred_target = {
            "repair": "diagnose",
            "finish": "finish",
            "abort": "abort",
        }.get(action)
        return StopDecision(
            action=action,
            confidence=probabilities[action],
            probabilities=probabilities,
            source="mlx-policy-residual" if policy_output is not None else "mlx-hardcoded-stop",
            preferred_target=preferred_target,
            allowed_actions=tuple(
                action_name for action_name in STOP_ACTIONS if action_name in allowed_actions
            ),
            policy_metrics=self._decision_metrics(state, node_id, policy_output),
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
        del graph
        if not candidates:
            raise ValueError("select_edge requires at least one condition-valid candidate")
        node_index = self.tables.node_index[node_id]
        candidate_by_key = {edge.key: edge for edge in candidates}
        unknown = set(candidate_by_key).difference(self.tables.edge_index)
        if unknown:
            raise ValueError(f"runtime candidates are absent from compiled graph: {sorted(unknown)}")

        logits = [priority / 10.0 for priority in self.tables.edge_priorities]
        mask: list[bool] = []
        for index, edge_key in enumerate(self.tables.edge_keys):
            traversal_available = (
                int(state.edge_counts.get(edge_key, 0))
                < self.tables.edge_max_traversals[index]
            )
            allowed = (
                self.tables.allowed_edge_mask[node_index][index]
                and edge_key in candidate_by_key
                and traversal_available
            )
            mask.append(allowed)
            if allowed and stop.preferred_target == candidate_by_key[edge_key].target:
                logits[index] += 6.0

        policy_output = self._policy_output(state, node_id)
        if policy_output is not None:
            logits = _add_logits(logits, policy_output.edge_logits, self.policy_scale)
        choice = self.backend.masked_softmax_argmax(logits, mask)
        edge_key = self.tables.edge_keys[choice.selected_index]
        chosen = candidate_by_key[edge_key]
        probabilities = {
            candidate.key: float(choice.probabilities[self.tables.edge_index[candidate.key]])
            for candidate in candidates
        }
        return EdgeDecision(
            edge=chosen,
            confidence=probabilities[chosen.key],
            probabilities=probabilities,
            allowed_edge_keys=tuple(candidate.key for candidate in candidates),
            source="mlx-policy-residual" if policy_output is not None else "mlx-hard-masked-edge",
            policy_metrics=self._decision_metrics(state, node_id, policy_output),
        )
