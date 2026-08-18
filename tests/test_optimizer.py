from graph_model.graph import load_default_graph
from graph_model.optimizer import SetNodeConfig, mcts_optimize


def test_mcts_search_selects_better_validated_variant() -> None:
    graph = load_default_graph()
    mutations = [
        SetNodeConfig("review", "strictness", 0.4),
        SetNodeConfig("review", "strictness", 0.8),
        SetNodeConfig("review", "strictness", 1.0),
    ]

    def evaluator(candidate):
        value = candidate.nodes["review"].config.get("strictness", 0.0)
        return 1.0 - abs(float(value) - 0.8)

    result = mcts_optimize(
        base_graph=graph,
        mutations=mutations,
        evaluator=evaluator,
        iterations=8,
        seed=7,
    )
    assert result.reward == 1.0
    assert result.graph.nodes["review"].config["strictness"] == 0.8
