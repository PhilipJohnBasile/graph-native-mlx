from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar

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
from .hidden_state import (
    HiddenStateObservation,
    HiddenStateSource,
    model_fingerprint,
)
from .policy import MLXPolicyRunner, PolicyOutput


class PolicyPredictor(Protocol):
    @property
    def identity(self) -> str: ...

    def predict(
        self,
        features: Sequence[float],
        *,
        hidden_features: Sequence[float] | None = None,
        hidden_metadata: Mapping[str, Any] | None = None,
    ) -> PolicyOutput: ...


_T = TypeVar("_T")


class AffinityExecutor(Protocol):
    def run_on_affinity(self, function: Callable[..., _T], *args: Any) -> _T: ...


@dataclass(frozen=True)
class _PolicyContext:
    explicit_features: tuple[float, ...]
    output: PolicyOutput | None
    observation: HiddenStateObservation | None


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
        "hidden_used": output.hidden_used,
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
        route_policy_scale: float | None = None,
        transition_policy_scale: float | None = None,
        skip_forced_policy: bool = False,
        hidden_state_source: HiddenStateSource | None = None,
        capture_hidden: bool = False,
        affinity_executor: AffinityExecutor | None = None,
    ) -> None:
        route_scale = policy_scale if route_policy_scale is None else route_policy_scale
        transition_scale = (
            policy_scale if transition_policy_scale is None else transition_policy_scale
        )
        if policy_scale < 0 or route_scale < 0 or transition_scale < 0:
            raise ValueError("policy scales must be non-negative")
        self.graph = graph
        self.tables = _resolve_tables(graph, tables)
        self.tables.validate_graph(graph)
        self.backend = decision_backend or MLXDecisionBackend()
        self.policy = policy
        self.policy_scale = float(policy_scale)
        self.route_policy_scale = float(route_scale)
        self.transition_policy_scale = float(transition_scale)
        self.skip_forced_policy = bool(skip_forced_policy)
        self.hidden_state_source = hidden_state_source
        self.capture_hidden = bool(capture_hidden)
        if bool(getattr(policy, "requires_hidden", False)):
            self.capture_hidden = True
            if hidden_state_source is None:
                raise ValueError(
                    "the configured MLX policy requires Qwen hidden-state features, "
                    "but no MLX hidden-state source was provided"
                )
        if self.capture_hidden and hidden_state_source is None:
            raise ValueError(
                "hidden-state capture was requested, but the selected provider cannot supply it"
            )
        policy_config = getattr(policy, "config", None)
        source_config = getattr(hidden_state_source, "hidden_config", None)
        source_identity = getattr(hidden_state_source, "identity", None)
        if bool(getattr(policy, "requires_hidden", False)) and policy_config is not None:
            mismatches: dict[str, tuple[object, object]] = {}
            if source_config is not None:
                actual_schema = getattr(source_config, "schema_hash", None)
                expected_schema = getattr(policy_config, "hidden_state_schema_hash", None)
                if actual_schema != expected_schema:
                    mismatches["hidden_state_schema_hash"] = (expected_schema, actual_schema)
            if isinstance(source_identity, Mapping):
                actual_model = model_fingerprint(source_identity)
                expected_model = getattr(policy_config, "model_fingerprint", None)
                if actual_model != expected_model:
                    mismatches["model_fingerprint"] = (expected_model, actual_model)
            if mismatches:
                raise ValueError(
                    "configured hidden-state policy is incompatible with the MLX provider: "
                    f"{mismatches}"
                )
        self.affinity_executor = affinity_executor
        if self.affinity_executor is None and callable(
            getattr(hidden_state_source, "run_on_affinity", None)
        ):
            self.affinity_executor = hidden_state_source  # type: ignore[assignment]
        self._cache_lock = RLock()
        self._cached_context_key: tuple[object, ...] | None = None
        self._cached_context: _PolicyContext | None = None

    @classmethod
    def from_env(
        cls,
        graph: GraphSpec,
        *,
        tables: CompiledGraphTables | None = None,
        decision_backend: DecisionBackend | None = None,
        hidden_state_source: HiddenStateSource | None = None,
        capture_hidden_override: bool | None = None,
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
            affinity = (
                hidden_state_source
                if callable(getattr(hidden_state_source, "run_on_affinity", None))
                else None
            )
            factory = lambda: MLXPolicyRunner(
                tables=compiled,
                weights_path=weights_path,
                config_path=config_path,
            )
            policy = (
                affinity.run_on_affinity(factory)
                if affinity is not None
                else factory()
            )
        capture_default = bool(
            getattr(hidden_state_source, "hidden_capture_enabled", False)
            or getattr(policy, "requires_hidden", False)
        )
        capture_raw = os.getenv("GRAPH_MODEL_MLX_CAPTURE_HIDDEN")
        capture_hidden = (
            capture_default
            if capture_raw is None
            else capture_raw.strip().lower() in {"1", "true", "yes", "on"}
        )
        if capture_hidden_override is not None:
            capture_hidden = bool(capture_hidden_override)
        return cls(
            graph=graph,
            tables=compiled,
            decision_backend=decision_backend,
            policy=policy,
            policy_scale=float(os.getenv("GRAPH_MODEL_MLX_POLICY_SCALE", "1.0")),
            route_policy_scale=float(
                os.getenv(
                    "GRAPH_MODEL_MLX_ROUTE_POLICY_SCALE",
                    os.getenv("GRAPH_MODEL_MLX_POLICY_SCALE", "1.0"),
                )
            ),
            transition_policy_scale=float(
                os.getenv(
                    "GRAPH_MODEL_MLX_TRANSITION_POLICY_SCALE",
                    os.getenv("GRAPH_MODEL_MLX_POLICY_SCALE", "1.0"),
                )
            ),
            skip_forced_policy=(
                os.getenv("GRAPH_MODEL_MLX_SKIP_FORCED_POLICY", "false")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            ),
            hidden_state_source=hidden_state_source,
            capture_hidden=capture_hidden,
            affinity_executor=(
                hidden_state_source
                if callable(getattr(hidden_state_source, "run_on_affinity", None))
                else None
            ),
        )

    @property
    def identity(self) -> dict[str, str]:
        return {
            "kind": "mlx-graph-controller",
            "version": "5",
            "graph_schema_hash": self.tables.schema_hash,
            "decision_backend": self.backend.identity,
            "policy": self.policy.identity if self.policy is not None else "hardcoded-priors-only",
            "policy_scale": f"{self.policy_scale:g}",
            "route_policy_scale": f"{self.route_policy_scale:g}",
            "transition_policy_scale": f"{self.transition_policy_scale:g}",
            "skip_forced_policy": str(self.skip_forced_policy).lower(),
            "hidden_capture": str(self.capture_hidden).lower(),
            "hidden_state": (
                self.hidden_state_source.hidden_state_identity
                if self.hidden_state_source is not None and self.capture_hidden
                else "off"
            ),
        }

    def _context(
        self,
        state: RunState,
        node_id: str,
        *,
        decision_type: str,
        evaluate_policy: bool = True,
    ) -> _PolicyContext:
        key = (
            state.run_id,
            state.step_count,
            node_id,
            decision_type,
            state.updated_at,
            state.last_progress_key,
            state.data.get("route"),
            state.data.get("verdict"),
            state.data.get("repair_count"),
            state.data.get("plan_revision_count"),
            state.metrics.llm_calls,
            state.metrics.tool_calls,
            state.metrics.total_tokens,
            evaluate_policy,
        )
        with self._cache_lock:
            if key == self._cached_context_key and self._cached_context is not None:
                return self._cached_context

            explicit = controller_feature_vector(
                self.graph,
                self.tables,
                state,
                node_id,
            )
            observation: HiddenStateObservation | None = None
            if (
                evaluate_policy
                and self.capture_hidden
                and self.hidden_state_source is not None
            ):
                observation = self.hidden_state_source.capture_policy_hidden(
                    state=state,
                    node_id=node_id,
                    decision_type=decision_type,
                )

            output: PolicyOutput | None = None
            if evaluate_policy and self.policy is not None:
                if self.affinity_executor is not None:
                    output = self.affinity_executor.run_on_affinity(
                        self._predict_policy,
                        self.policy,
                        explicit,
                        observation,
                    )
                else:
                    output = self._predict_policy(
                        self.policy,
                        explicit,
                        observation,
                    )

            context = _PolicyContext(
                explicit_features=explicit,
                output=output,
                observation=observation,
            )
            self._cached_context_key = key
            self._cached_context = context
            return context

    @staticmethod
    def _predict_policy(
        policy: PolicyPredictor,
        features: Sequence[float],
        observation: HiddenStateObservation | None,
    ) -> PolicyOutput:
        predict = policy.predict
        try:
            parameters = inspect.signature(predict).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_hidden = (
            "hidden_features" in parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        )
        if accepts_hidden:
            metadata = (
                {
                    "extractor_schema_hash": observation.reference.extractor_schema_hash,
                    "model_fingerprint": observation.reference.model_fingerprint,
                    "artifact_sha256": observation.reference.sha256,
                }
                if observation is not None
                else None
            )
            return predict(
                features,
                hidden_features=(observation.features if observation is not None else None),
                hidden_metadata=metadata,
            )
        if bool(getattr(policy, "requires_hidden", False)):
            raise ValueError("policy requires hidden features but its predict API cannot accept them")
        return predict(features)

    def _masked_choice(
        self,
        logits: Sequence[float],
        mask: Sequence[bool],
    ):
        if self.affinity_executor is not None:
            return self.affinity_executor.run_on_affinity(
                self.backend.masked_softmax_argmax,
                logits,
                mask,
            )
        return self.backend.masked_softmax_argmax(logits, mask)

    def _decision_metrics(
        self,
        context: _PolicyContext,
        *,
        valid_choice_count: int,
        policy_context_evaluated: bool,
        policy_could_change_choice: bool,
        static_choice: str,
        learned_choice: str,
    ) -> dict[str, object]:
        policy_evaluated = context.output is not None
        metrics: dict[str, object] = {
            "graph_schema_hash": self.tables.schema_hash,
            "feature_vector": list(context.explicit_features),
            "valid_choice_count": int(valid_choice_count),
            "policy_context_evaluated": bool(policy_context_evaluated),
            "policy_evaluated": policy_evaluated,
            "policy_could_change_choice": bool(policy_could_change_choice),
            "static_choice": static_choice,
            "learned_choice": learned_choice,
            "choice_changed": static_choice != learned_choice,
        }
        if context.observation is not None:
            metrics["hidden_state"] = context.observation.reference.as_dict()
            metrics["hidden_state_cache_hit"] = context.observation.cache_hit
        metrics.update(_policy_metrics(context.output))
        return metrics

    def select_route(self, task: str, state: RunState) -> RouteDecision:
        fallback = rule_route(task)
        base_logits = [0.0] * len(ROUTES)
        base_logits[ROUTES.index(fallback)] = 8.0
        static = self._masked_choice(base_logits, [True] * len(ROUTES))
        static_route = ROUTES[static.selected_index]
        context = self._context(state, self.graph.start, decision_type="route")
        logits = list(base_logits)
        if context.output is not None:
            logits = _add_logits(
                logits,
                context.output.route_logits,
                self.route_policy_scale,
            )
        choice = self._masked_choice(logits, [True] * len(ROUTES))
        route = ROUTES[choice.selected_index]
        probabilities = {
            name: float(choice.probabilities[index]) for index, name in enumerate(ROUTES)
        }
        if context.output is None:
            source = "mlx-hardcoded-route"
        elif self.route_policy_scale == 0:
            source = "mlx-policy-shadow"
        else:
            source = "mlx-policy-residual"
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
            policy_metrics=self._decision_metrics(
                context,
                valid_choice_count=len(ROUTES),
                policy_context_evaluated=True,
                policy_could_change_choice=(
                    context.output is not None and self.route_policy_scale > 0
                ),
                static_choice=static_route,
                learned_choice=route,
            ),
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
        valid_choice_count = sum(action_mask)
        prior_action = (
            _action_for_target(candidates[0].target)
            if candidates
            else ("abort" if node_id == "abort" else "finish")
        )
        base_logits = [0.0] * len(STOP_ACTIONS)
        base_logits[STOP_ACTIONS.index(prior_action)] = 8.0
        static = self._masked_choice(base_logits, action_mask)
        static_action = STOP_ACTIONS[static.selected_index]
        evaluate = not (self.skip_forced_policy and valid_choice_count <= 1)
        context = self._context(
            state,
            node_id,
            decision_type="transition",
            evaluate_policy=evaluate,
        )
        logits = list(base_logits)
        if context.output is not None:
            logits = _add_logits(
                logits,
                context.output.stop_logits,
                self.transition_policy_scale,
            )
        choice = self._masked_choice(logits, action_mask)
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
        if context.output is None:
            source = "mlx-hardcoded-stop"
        elif self.transition_policy_scale == 0:
            source = "mlx-policy-shadow"
        else:
            source = "mlx-policy-residual"
        return StopDecision(
            action=action,
            confidence=probabilities[action],
            probabilities=probabilities,
            source=source,
            preferred_target=preferred_target,
            allowed_actions=tuple(
                action_name for action_name in STOP_ACTIONS if action_name in allowed_actions
            ),
            policy_metrics=self._decision_metrics(
                context,
                valid_choice_count=valid_choice_count,
                policy_context_evaluated=evaluate,
                policy_could_change_choice=(
                    context.output is not None
                    and self.transition_policy_scale > 0
                    and valid_choice_count > 1
                ),
                static_choice=static_action,
                learned_choice=action,
            ),
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

        base_logits = [priority / 10.0 for priority in self.tables.edge_priorities]
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
                base_logits[index] += 6.0

        valid_choice_count = sum(mask)
        static = self._masked_choice(base_logits, mask)
        static_edge_key = self.tables.edge_keys[static.selected_index]
        evaluate = not (self.skip_forced_policy and valid_choice_count <= 1)
        context = self._context(
            state,
            node_id,
            decision_type="transition",
            evaluate_policy=evaluate,
        )
        logits = list(base_logits)
        if context.output is not None:
            logits = _add_logits(
                logits,
                context.output.edge_logits,
                self.transition_policy_scale,
            )
        choice = self._masked_choice(logits, mask)
        edge_key = self.tables.edge_keys[choice.selected_index]
        chosen = candidate_by_key[edge_key]
        probabilities = {
            candidate.key: float(choice.probabilities[self.tables.edge_index[candidate.key]])
            for candidate in candidates
        }
        if context.output is None:
            source = "mlx-hard-masked-edge"
        elif self.transition_policy_scale == 0:
            source = "mlx-policy-shadow"
        else:
            source = "mlx-policy-residual"
        return EdgeDecision(
            edge=chosen,
            confidence=probabilities[chosen.key],
            probabilities=probabilities,
            allowed_edge_keys=tuple(candidate.key for candidate in candidates),
            source=source,
            policy_metrics=self._decision_metrics(
                context,
                valid_choice_count=valid_choice_count,
                policy_context_evaluated=evaluate,
                policy_could_change_choice=(
                    context.output is not None
                    and self.transition_policy_scale > 0
                    and valid_choice_count > 1
                ),
                static_choice=static_edge_key,
                learned_choice=edge_key,
            ),
        )
