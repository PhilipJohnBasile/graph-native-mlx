from pathlib import Path

from graph_model.graph import load_default_graph
from graph_model.mlx_native.generated_coding_graph import COMPILED_GRAPH
from graph_model.mlx_native.graph_tables import (
    compile_graph,
    graph_schema_hash,
    write_generated_module,
)


def test_generated_graph_tables_match_the_validated_yaml() -> None:
    graph = load_default_graph()
    compiled = compile_graph(graph)
    COMPILED_GRAPH.validate_graph(graph)
    assert compiled == COMPILED_GRAPH
    assert compiled.schema_hash == graph_schema_hash(graph)
    assert len(compiled.node_ids) == 12
    assert len(compiled.edge_keys) == 19


def test_compiled_masks_encode_only_declared_sources() -> None:
    graph = load_default_graph()
    tables = compile_graph(graph)
    for node_id, row in zip(tables.node_ids, tables.allowed_edge_mask, strict=True):
        allowed_keys = {
            edge_key for edge_key, allowed in zip(tables.edge_keys, row, strict=True) if allowed
        }
        expected_keys = {edge.key for edge in graph.edges if edge.source == node_id}
        assert allowed_keys == expected_keys
        if node_id in graph.terminals:
            assert not any(row)


def test_graph_compiler_emits_an_importable_module(tmp_path: Path) -> None:
    output = write_generated_module(
        compile_graph(load_default_graph()),
        tmp_path / "generated_graph.py",
    )
    source = output.read_text(encoding="utf-8")
    assert "COMPILED_GRAPH = CompiledGraphTables" in source
    assert graph_schema_hash(load_default_graph()) in source
