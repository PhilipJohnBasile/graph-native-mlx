import pytest

from graph_model.controller import DeterministicGraphController
from graph_model.graph import load_default_graph
from graph_model.graph_bundle import verify_graph_bundle
from graph_model.graph_search import ObjectiveWeights, SearchCase, optimize_graph
from graph_model.optimizer import SetNodeConfig
from graph_model.provider import MockProvider


class TemperatureCostProvider(MockProvider):
    async def complete_json(self, *, system, user, temperature=None):
        payload, prompt_tokens, _ = await super().complete_json(
            system=system,
            user=user,
            temperature=temperature,
        )
        completion_tokens = 100 if temperature == 0.1 else 1
        return payload, prompt_tokens, completion_tokens


@pytest.mark.asyncio
async def test_graph_search_promotes_only_a_validated_better_candidate(tmp_path) -> None:
    graph = load_default_graph()
    provider = TemperatureCostProvider()
    report = await optimize_graph(
        base_graph=graph,
        train_cases=(SearchCase(case_id="train", task="quick fix: rename one variable"),),
        validation_cases=(
            SearchCase(case_id="validation", task="quick fix: rename one local symbol"),
        ),
        provider=provider,
        controller_factory=lambda candidate: DeterministicGraphController(),
        output_dir=tmp_path / "optimized",
        mutations=[SetNodeConfig("implement", "temperature", 0.0)],
        weights=ObjectiveWeights(
            quality=1.0,
            tokens=1.0,
            llm_calls=0.0,
            tool_calls=0.0,
            path_length=0.0,
            active_seconds=0.0,
        ),
        iterations=2,
        seed=1,
        min_validation_improvement=0.0,
    )

    assert report["status"] == "promoted"
    assert report["independent_validation"] is True
    assert report["promotion_gate"]["quality_not_worse"] is True
    assert report["promotion_gate"]["actual_improvement"] > 0
    assert report["search"]["winning_mutation_path"] == [
        "node:implement:temperature=0.0"
    ]

    bundle = verify_graph_bundle(tmp_path / "optimized", require_promoted=True)
    assert bundle.graph.nodes["implement"].config["temperature"] == 0.0
    assert bundle.manifest["promotion_status"] == "promoted"


def test_graph_search_rejects_retry_cap_increases_and_non_llm_config() -> None:
    from graph_model.graph_search import validate_mutation_envelope
    from graph_model.optimizer import SetEdgeLimit, SetNodeConfig

    graph = load_default_graph()
    with pytest.raises(ValueError, match="preserve or lower"):
        validate_mutation_envelope(
            graph,
            [SetEdgeLimit("tests->diagnose:test-repair", 3)],
        )
    with pytest.raises(ValueError, match="temperature of existing LLM"):
        validate_mutation_envelope(
            graph,
            [SetNodeConfig("tests", "command_timeout", 9999)],
        )


def test_graph_search_requires_truly_disjoint_validation_cases() -> None:
    from graph_model.graph_search import validate_held_out_cases

    training = [SearchCase(case_id="train", task="quick fix: rename one variable")]
    with pytest.raises(ValueError, match="case IDs"):
        validate_held_out_cases(
            training,
            [SearchCase(case_id="train", task="different task")],
        )
    with pytest.raises(ValueError, match="exact training tasks"):
        validate_held_out_cases(
            training,
            [SearchCase(case_id="validation", task="quick fix: rename one variable")],
        )
