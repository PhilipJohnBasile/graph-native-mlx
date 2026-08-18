from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

from graph_model.models import GraphSpec


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, set):
        return sorted((_normalize(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def normalized_graph_payload(graph: GraphSpec) -> dict[str, Any]:
    payload = graph.model_dump(mode="json", by_alias=True)
    payload["terminals"] = sorted(graph.terminals)
    payload["nodes"] = {
        node_id: _normalize(payload["nodes"][node_id]) for node_id in sorted(graph.nodes)
    }
    payload["edges"] = sorted(
        (_normalize(edge) for edge in payload["edges"]),
        key=lambda edge: (
            str(edge.get("from", "")),
            str(edge.get("to", "")),
            str(edge.get("label", "")),
            str(edge.get("when", "")),
        ),
    )
    return _normalize(payload)


def graph_schema_hash(graph: GraphSpec) -> str:
    payload = json.dumps(
        normalized_graph_payload(graph),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CompiledGraphTables:
    name: str
    version: str
    schema_hash: str
    node_ids: tuple[str, ...]
    edge_keys: tuple[str, ...]
    edge_sources: tuple[int, ...]
    edge_targets: tuple[int, ...]
    edge_priorities: tuple[float, ...]
    edge_max_traversals: tuple[int, ...]
    edge_conditions: tuple[str, ...]
    allowed_edge_mask: tuple[tuple[bool, ...], ...]
    terminal_mask: tuple[bool, ...]

    @property
    def node_index(self) -> dict[str, int]:
        return {node_id: index for index, node_id in enumerate(self.node_ids)}

    @property
    def edge_index(self) -> dict[str, int]:
        return {edge_key: index for index, edge_key in enumerate(self.edge_keys)}

    def validate_graph(self, graph: GraphSpec) -> None:
        actual_hash = graph_schema_hash(graph)
        if self.name != graph.name or self.version != graph.version:
            raise ValueError(
                "compiled graph identity does not match loaded graph: "
                f"compiled={self.name}@{self.version}, loaded={graph.name}@{graph.version}"
            )
        if self.schema_hash != actual_hash:
            raise ValueError(
                "compiled graph schema hash does not match loaded graph: "
                f"compiled={self.schema_hash}, loaded={actual_hash}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "schema_hash": self.schema_hash,
            "node_ids": list(self.node_ids),
            "edge_keys": list(self.edge_keys),
            "edge_sources": list(self.edge_sources),
            "edge_targets": list(self.edge_targets),
            "edge_priorities": list(self.edge_priorities),
            "edge_max_traversals": list(self.edge_max_traversals),
            "edge_conditions": list(self.edge_conditions),
            "allowed_edge_mask": [list(row) for row in self.allowed_edge_mask],
            "terminal_mask": list(self.terminal_mask),
        }


def compile_graph(graph: GraphSpec) -> CompiledGraphTables:
    node_ids = tuple(sorted(graph.nodes))
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    edges = tuple(sorted(graph.edges, key=lambda edge: edge.key))
    edge_keys = tuple(edge.key for edge in edges)
    allowed_rows: list[tuple[bool, ...]] = []
    for node_id in node_ids:
        allowed_rows.append(tuple(edge.source == node_id for edge in edges))

    return CompiledGraphTables(
        name=graph.name,
        version=graph.version,
        schema_hash=graph_schema_hash(graph),
        node_ids=node_ids,
        edge_keys=edge_keys,
        edge_sources=tuple(node_index[edge.source] for edge in edges),
        edge_targets=tuple(node_index[edge.target] for edge in edges),
        edge_priorities=tuple(float(edge.priority) for edge in edges),
        edge_max_traversals=tuple(int(edge.max_traversals) for edge in edges),
        edge_conditions=tuple(edge.when for edge in edges),
        allowed_edge_mask=tuple(allowed_rows),
        terminal_mask=tuple(node_id in graph.terminals for node_id in node_ids),
    )


def write_generated_module(tables: CompiledGraphTables, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = tables.as_dict()
    source = (
        '"""Generated graph tables. Do not edit by hand; run `graph-model compile-graph`."""\n\n'
        "from __future__ import annotations\n\n"
        "from .graph_tables import CompiledGraphTables\n\n"
        f"GRAPH_PAYLOAD = {pformat(payload, width=100, sort_dicts=False)}\n\n"
        "COMPILED_GRAPH = CompiledGraphTables(\n"
        "    name=GRAPH_PAYLOAD['name'],\n"
        "    version=GRAPH_PAYLOAD['version'],\n"
        "    schema_hash=GRAPH_PAYLOAD['schema_hash'],\n"
        "    node_ids=tuple(GRAPH_PAYLOAD['node_ids']),\n"
        "    edge_keys=tuple(GRAPH_PAYLOAD['edge_keys']),\n"
        "    edge_sources=tuple(GRAPH_PAYLOAD['edge_sources']),\n"
        "    edge_targets=tuple(GRAPH_PAYLOAD['edge_targets']),\n"
        "    edge_priorities=tuple(GRAPH_PAYLOAD['edge_priorities']),\n"
        "    edge_max_traversals=tuple(GRAPH_PAYLOAD['edge_max_traversals']),\n"
        "    edge_conditions=tuple(GRAPH_PAYLOAD['edge_conditions']),\n"
        "    allowed_edge_mask=tuple(tuple(row) for row in GRAPH_PAYLOAD['allowed_edge_mask']),\n"
        "    terminal_mask=tuple(GRAPH_PAYLOAD['terminal_mask']),\n"
        ")\n"
    )
    output.write_text(source, encoding="utf-8")
    return output
