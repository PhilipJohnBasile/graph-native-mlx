from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from .models import GraphSpec, NodeSpec


def _normalize_graph(payload: dict[str, Any]) -> dict[str, Any]:
    nodes = payload.get("nodes", {})
    normalized_nodes: dict[str, Any] = {}
    for node_id, node_data in nodes.items():
        item = dict(node_data or {})
        item["id"] = node_id
        normalized_nodes[node_id] = item
    payload = dict(payload)
    payload["nodes"] = normalized_nodes
    return payload


def load_graph(path: str | Path) -> GraphSpec:
    graph_path = Path(path)
    with graph_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"graph file {graph_path} must contain a mapping")
    return GraphSpec.model_validate(_normalize_graph(payload))


def load_default_graph() -> GraphSpec:
    graph_resource = files("graph_model.graphs").joinpath("coding_supergraph.yaml")
    with graph_resource.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("bundled graph must contain a mapping")
    return GraphSpec.model_validate(_normalize_graph(payload))


def clone_graph(graph: GraphSpec) -> GraphSpec:
    return GraphSpec.model_validate(graph.model_dump(by_alias=True))


def replace_node(graph: GraphSpec, node: NodeSpec) -> GraphSpec:
    payload = graph.model_dump(by_alias=True)
    payload["nodes"][node.id] = node.model_dump()
    return GraphSpec.model_validate(payload)
