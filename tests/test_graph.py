import pytest

from graph_model.graph import load_default_graph
from graph_model.models import GraphSpec


def test_default_graph_is_validated() -> None:
    graph = load_default_graph()
    assert graph.start == "intake"
    assert graph.terminals == {"finish", "abort"}
    assert len(graph.nodes) == 11
    assert any(edge.target == "repair" for edge in graph.edges)
    assert next(edge for edge in graph.edges if edge.source == "plan" and edge.target == "plan_check").max_traversals == 2
    assert next(edge for edge in graph.edges if edge.source == "tests" and edge.target == "review").max_traversals == 3


def test_graph_validation_rejects_unreachable_nodes() -> None:
    graph = load_default_graph()
    payload = graph.model_dump(by_alias=True)
    payload["nodes"]["orphan"] = {
        "id": "orphan",
        "kind": "final",
        "operator": "abort",
        "description": "unreachable",
        "config": {},
        "cacheable": True,
        "side_effect": False,
    }
    payload["terminals"].add("orphan")
    with pytest.raises(ValueError, match="unreachable nodes"):
        GraphSpec.model_validate(payload)
