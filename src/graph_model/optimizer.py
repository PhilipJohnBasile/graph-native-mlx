from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Awaitable, Protocol

from .graph import clone_graph
from .models import GraphSpec
from .mlx_native.graph_tables import graph_schema_hash


class GraphMutation(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def slot(self) -> str: ...

    def apply(self, graph: GraphSpec) -> GraphSpec: ...


class GraphEvaluator(Protocol):
    def __call__(self, graph: GraphSpec) -> float: ...


class AsyncGraphEvaluator(Protocol):
    def __call__(self, graph: GraphSpec) -> Awaitable[float]: ...


@dataclass(frozen=True)
class SetNodeConfig:
    node_id: str
    name: str
    value: object

    @property
    def key(self) -> str:
        return f"node:{self.node_id}:{self.name}={self.value!r}"

    @property
    def slot(self) -> str:
        return f"node:{self.node_id}:{self.name}"

    def apply(self, graph: GraphSpec) -> GraphSpec:
        candidate = clone_graph(graph)
        node = candidate.nodes[self.node_id].model_copy(deep=True)
        node.config[self.name] = self.value
        candidate.nodes[self.node_id] = node
        return GraphSpec.model_validate(candidate.model_dump(by_alias=True))


@dataclass(frozen=True)
class SetEdgeLimit:
    edge_key: str
    max_traversals: int

    @property
    def key(self) -> str:
        return f"edge:{self.edge_key}:max={self.max_traversals}"

    @property
    def slot(self) -> str:
        return f"edge:{self.edge_key}:max"

    def apply(self, graph: GraphSpec) -> GraphSpec:
        if self.max_traversals < 1:
            raise ValueError("max_traversals must be >= 1")
        candidate = clone_graph(graph)
        found = False
        new_edges = []
        for edge in candidate.edges:
            if edge.key == self.edge_key:
                new_edges.append(edge.model_copy(update={"max_traversals": self.max_traversals}))
                found = True
            else:
                new_edges.append(edge)
        if not found:
            raise KeyError(f"edge {self.edge_key!r} not found")
        candidate.edges = new_edges
        return GraphSpec.model_validate(candidate.model_dump(by_alias=True))


@dataclass(frozen=True)
class SetEdgePriority:
    edge_key: str
    priority: int

    @property
    def key(self) -> str:
        return f"edge:{self.edge_key}:priority={self.priority}"

    @property
    def slot(self) -> str:
        return f"edge:{self.edge_key}:priority"

    def apply(self, graph: GraphSpec) -> GraphSpec:
        candidate = clone_graph(graph)
        found = False
        new_edges = []
        for edge in candidate.edges:
            if edge.key == self.edge_key:
                new_edges.append(edge.model_copy(update={"priority": self.priority}))
                found = True
            else:
                new_edges.append(edge)
        if not found:
            raise KeyError(f"edge {self.edge_key!r} not found")
        candidate.edges = new_edges
        return GraphSpec.model_validate(candidate.model_dump(by_alias=True))


@dataclass
class SearchNode:
    graph: GraphSpec
    parent: "SearchNode | None" = None
    mutation_key: str | None = None
    mutation_slot: str | None = None
    applied_slots: frozenset[str] = frozenset()
    untried: list[GraphMutation] = field(default_factory=list)
    children: list["SearchNode"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else float("-inf")

    def uct(self, exploration: float) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = max(1, self.parent.visits if self.parent else self.visits)
        return self.mean_reward + exploration * math.sqrt(math.log(parent_visits) / self.visits)


@dataclass(frozen=True)
class CandidateEvaluation:
    schema_hash: str
    reward: float
    mutation_path: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    graph: GraphSpec
    reward: float
    mutation_path: tuple[str, ...]
    evaluations: int
    evaluated_candidates: tuple[CandidateEvaluation, ...] = ()


def _mutation_slot(mutation: GraphMutation) -> str:
    slot = getattr(mutation, "slot", None)
    return str(slot) if slot else str(mutation.key)


def _path(node: SearchNode) -> tuple[str, ...]:
    path: list[str] = []
    cursor = node
    while cursor.parent is not None:
        if cursor.mutation_key:
            path.append(cursor.mutation_key)
        cursor = cursor.parent
    return tuple(reversed(path))


def _new_child(
    *,
    node: SearchNode,
    mutation: GraphMutation,
    mutations: list[GraphMutation],
) -> SearchNode:
    slot = _mutation_slot(mutation)
    applied_slots = frozenset((*node.applied_slots, slot))
    child_graph = mutation.apply(node.graph)
    inherited = [item for item in mutations if _mutation_slot(item) not in applied_slots]
    child = SearchNode(
        graph=child_graph,
        parent=node,
        mutation_key=mutation.key,
        mutation_slot=slot,
        applied_slots=applied_slots,
        untried=inherited,
    )
    node.children.append(child)
    return child


def _backpropagate(node: SearchNode, reward: float) -> None:
    cursor: SearchNode | None = node
    while cursor is not None:
        cursor.visits += 1
        cursor.total_reward += reward
        cursor = cursor.parent


def mcts_optimize(
    *,
    base_graph: GraphSpec,
    mutations: list[GraphMutation],
    evaluator: GraphEvaluator,
    iterations: int = 20,
    exploration: float = 1.4,
    seed: int = 42,
) -> SearchResult:
    """Constrained AFlow-style graph search with mutation-slot exclusivity.

    Candidate graphs remain typed and validated. Only one mutation can occupy a logical slot in a
    path, preventing contradictory variants such as two different traversal limits for one edge.
    Duplicate graph schemas are evaluated once and replayed from the cache.
    """

    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if exploration < 0:
        raise ValueError("exploration must be non-negative")
    rng = random.Random(seed)
    root = SearchNode(graph=clone_graph(base_graph), untried=list(mutations))
    evaluation_cache: dict[str, float] = {}
    evaluated: list[CandidateEvaluation] = []

    def evaluate(node: SearchNode) -> float:
        schema = graph_schema_hash(node.graph)
        if schema not in evaluation_cache:
            reward = float(evaluator(node.graph))
            if not math.isfinite(reward):
                raise ValueError("graph evaluator returned a non-finite reward")
            evaluation_cache[schema] = reward
            evaluated.append(CandidateEvaluation(schema, reward, _path(node)))
        return evaluation_cache[schema]

    best_graph = root.graph
    best_reward = evaluate(root)
    best_path: tuple[str, ...] = ()
    _backpropagate(root, best_reward)

    for _ in range(iterations - 1):
        node = root
        while not node.untried and node.children:
            node = max(node.children, key=lambda child: child.uct(exploration))

        if node.untried:
            mutation = rng.choice(node.untried)
            node.untried.remove(mutation)
            node = _new_child(node=node, mutation=mutation, mutations=mutations)

        reward = evaluate(node)
        _backpropagate(node, reward)
        if reward > best_reward:
            best_reward = reward
            best_graph = node.graph
            best_path = _path(node)

    return SearchResult(
        graph=best_graph,
        reward=best_reward,
        mutation_path=best_path,
        evaluations=len(evaluation_cache),
        evaluated_candidates=tuple(evaluated),
    )


async def async_mcts_optimize(
    *,
    base_graph: GraphSpec,
    mutations: list[GraphMutation],
    evaluator: AsyncGraphEvaluator,
    iterations: int = 20,
    exploration: float = 1.4,
    seed: int = 42,
) -> SearchResult:
    """Async variant used by real model and repository benchmark evaluators."""

    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if exploration < 0:
        raise ValueError("exploration must be non-negative")
    rng = random.Random(seed)
    root = SearchNode(graph=clone_graph(base_graph), untried=list(mutations))
    evaluation_cache: dict[str, float] = {}
    evaluated: list[CandidateEvaluation] = []

    async def evaluate(node: SearchNode) -> float:
        schema = graph_schema_hash(node.graph)
        if schema not in evaluation_cache:
            pending = evaluator(node.graph)
            reward = float(await pending)
            if not math.isfinite(reward):
                raise ValueError("graph evaluator returned a non-finite reward")
            evaluation_cache[schema] = reward
            evaluated.append(CandidateEvaluation(schema, reward, _path(node)))
        return evaluation_cache[schema]

    best_graph = root.graph
    best_reward = await evaluate(root)
    best_path: tuple[str, ...] = ()
    _backpropagate(root, best_reward)

    for _ in range(iterations - 1):
        node = root
        while not node.untried and node.children:
            node = max(node.children, key=lambda child: child.uct(exploration))

        if node.untried:
            mutation = rng.choice(node.untried)
            node.untried.remove(mutation)
            node = _new_child(node=node, mutation=mutation, mutations=mutations)

        reward = await evaluate(node)
        _backpropagate(node, reward)
        if reward > best_reward:
            best_reward = reward
            best_graph = node.graph
            best_path = _path(node)

    return SearchResult(
        graph=best_graph,
        reward=best_reward,
        mutation_path=best_path,
        evaluations=len(evaluation_cache),
        evaluated_candidates=tuple(evaluated),
    )
