from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from .graph import clone_graph
from .models import GraphSpec


class GraphMutation(Protocol):
    @property
    def key(self) -> str: ...

    def apply(self, graph: GraphSpec) -> GraphSpec: ...


class GraphEvaluator(Protocol):
    def __call__(self, graph: GraphSpec) -> float: ...


@dataclass(frozen=True)
class SetNodeConfig:
    node_id: str
    name: str
    value: object

    @property
    def key(self) -> str:
        return f"node:{self.node_id}:{self.name}={self.value!r}"

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

    def apply(self, graph: GraphSpec) -> GraphSpec:
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


@dataclass
class SearchNode:
    graph: GraphSpec
    parent: "SearchNode | None" = None
    mutation_key: str | None = None
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
class SearchResult:
    graph: GraphSpec
    reward: float
    mutation_path: tuple[str, ...]
    evaluations: int


def mcts_optimize(
    *,
    base_graph: GraphSpec,
    mutations: list[GraphMutation],
    evaluator: GraphEvaluator,
    iterations: int = 20,
    exploration: float = 1.4,
    seed: int = 42,
) -> SearchResult:
    """Small, deterministic AFlow-style search scaffold.

    Candidate graphs remain typed and validated. The caller supplies real benchmark execution as
    the evaluator and an explicit mutation library, keeping the production search space constrained.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    rng = random.Random(seed)
    root = SearchNode(graph=clone_graph(base_graph), untried=list(mutations))
    best_graph = root.graph
    best_reward = evaluator(root.graph)
    best_path: tuple[str, ...] = ()
    evaluations = 1

    for _ in range(iterations - 1):
        node = root
        while not node.untried and node.children:
            node = max(node.children, key=lambda child: child.uct(exploration))

        if node.untried:
            mutation = rng.choice(node.untried)
            node.untried.remove(mutation)
            child_graph = mutation.apply(node.graph)
            inherited = [item for item in mutations if item.key != mutation.key]
            child = SearchNode(
                graph=child_graph,
                parent=node,
                mutation_key=mutation.key,
                untried=inherited,
            )
            node.children.append(child)
            node = child

        reward = evaluator(node.graph)
        evaluations += 1
        cursor: SearchNode | None = node
        while cursor is not None:
            cursor.visits += 1
            cursor.total_reward += reward
            cursor = cursor.parent

        if reward > best_reward:
            best_reward = reward
            best_graph = node.graph
            path: list[str] = []
            cursor = node
            while cursor.parent is not None:
                if cursor.mutation_key:
                    path.append(cursor.mutation_key)
                cursor = cursor.parent
            best_path = tuple(reversed(path))

    return SearchResult(
        graph=best_graph,
        reward=best_reward,
        mutation_path=best_path,
        evaluations=evaluations,
    )
