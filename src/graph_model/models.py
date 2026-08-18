from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NodeKind(str, Enum):
    ROUTER = "router"
    LLM = "llm"
    TOOL = "tool"
    VERIFIER = "verifier"
    FINAL = "final"


class NodeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: NodeKind
    operator: str
    description: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    cacheable: bool = True
    side_effect: bool = False


class EdgeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source: str = Field(alias="from")
    target: str = Field(alias="to")
    when: str = "always"
    priority: int = 0
    max_traversals: int = Field(default=1, ge=1)
    label: str = ""

    @property
    def key(self) -> str:
        return f"{self.source}->{self.target}:{self.label or self.when}"


class GraphSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    description: str = ""
    start: str
    terminals: set[str]
    nodes: dict[str, NodeSpec]
    edges: list[EdgeSpec]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_graph(self) -> "GraphSpec":
        if self.start not in self.nodes:
            raise ValueError(f"start node {self.start!r} is not defined")
        missing_terminals = self.terminals.difference(self.nodes)
        if missing_terminals:
            raise ValueError(f"terminal nodes are missing: {sorted(missing_terminals)}")
        for key, node in self.nodes.items():
            if key != node.id:
                raise ValueError(f"node map key {key!r} must equal node.id {node.id!r}")
        edge_keys: set[str] = set()
        adjacency: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        reverse: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            if edge.source not in self.nodes:
                raise ValueError(f"edge source {edge.source!r} is not defined")
            if edge.target not in self.nodes:
                raise ValueError(f"edge target {edge.target!r} is not defined")
            if edge.source in self.terminals:
                raise ValueError(f"terminal node {edge.source!r} cannot have outgoing edges")
            if edge.key in edge_keys:
                raise ValueError(f"duplicate edge key {edge.key!r}")
            edge_keys.add(edge.key)
            adjacency[edge.source].append(edge.target)
            reverse[edge.target].append(edge.source)
        for node_id in self.nodes:
            if node_id not in self.terminals and not adjacency[node_id]:
                raise ValueError(f"non-terminal node {node_id!r} has no outgoing edges")

        reachable = {self.start}
        frontier = [self.start]
        while frontier:
            current = frontier.pop()
            for target in adjacency[current]:
                if target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        unreachable = set(self.nodes).difference(reachable)
        if unreachable:
            raise ValueError(f"unreachable nodes: {sorted(unreachable)}")

        can_reach_terminal = set(self.terminals)
        frontier = list(self.terminals)
        while frontier:
            current = frontier.pop()
            for source in reverse[current]:
                if source not in can_reach_terminal:
                    can_reach_terminal.add(source)
                    frontier.append(source)
        nonterminating = set(self.nodes).difference(can_reach_terminal)
        if nonterminating:
            raise ValueError(f"nodes cannot reach a terminal: {sorted(nonterminating)}")
        return self

    def outgoing(self, node_id: str) -> list[EdgeSpec]:
        return sorted(
            (edge for edge in self.edges if edge.source == node_id),
            key=lambda edge: edge.priority,
            reverse=True,
        )


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=24, ge=1)
    max_llm_calls: int = Field(default=12, ge=0)
    max_tool_calls: int = Field(default=24, ge=0)
    max_tokens: int = Field(default=64_000, ge=0)
    max_seconds: float = Field(default=900.0, gt=0)
    max_no_progress_steps: int = Field(default=2, ge=1)


class RunMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_steps: int = 0
    elapsed_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class NodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delta: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    verdict: str | None = None
    progress_key: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    notes: list[str] = Field(default_factory=list)


class RunState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    graph_name: str
    graph_version: str
    task: str
    current_node: str
    status: str = "running"
    data: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    attempts: dict[str, int] = Field(default_factory=dict)
    edge_counts: dict[str, int] = Field(default_factory=dict)
    completed_nodes: list[str] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    budget: Budget = Field(default_factory=Budget)
    step_count: int = 0
    no_progress_count: int = 0
    last_progress_key: str | None = None
    started_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    output: Any = None
    error: str | None = None

    @classmethod
    def new(
        cls,
        *,
        graph: GraphSpec,
        task: str,
        budget: Budget | None = None,
        run_id: str | None = None,
        initial_data: dict[str, Any] | None = None,
    ) -> "RunState":
        return cls(
            run_id=run_id or str(uuid.uuid4()),
            graph_name=graph.name,
            graph_version=graph.version,
            task=task,
            current_node=graph.start,
            budget=budget or Budget(),
            data={"repair_count": 0, "plan_revision_count": 0, **(initial_data or {})},
        )
