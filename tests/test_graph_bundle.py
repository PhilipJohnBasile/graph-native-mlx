import json

import pytest

from graph_model.graph import load_default_graph
from graph_model.graph_bundle import (
    GraphBundleError,
    optimized_graph_version,
    verify_graph_bundle,
    write_graph_bundle,
)
from graph_model.optimizer import SetNodeConfig


def test_graph_bundle_is_reproducible_and_requires_explicit_promotion(tmp_path) -> None:
    base = load_default_graph()
    candidate = SetNodeConfig("implement", "temperature", 0.0).apply(base)
    candidate = optimized_graph_version(
        candidate,
        mutation_path=("node:implement:temperature=0.0",),
    )
    bundle = write_graph_bundle(
        graph=candidate,
        output_dir=tmp_path / "bundle",
        benchmark_report={
            "status": "promoted",
            "promotion_gate": {
                "promotion_allowed": True,
                "quality_not_worse": True,
                "has_mutation": True,
                "actual_improvement": 0.1,
                "minimum_improvement": 0.0,
            },
            "search": {
                "winning_mutation_path": ["node:implement:temperature=0.0"]
            },
            "validation": {
                "baseline": {"reward": 0.8},
                "candidate": {"reward": 0.9},
            },
        },
        baseline_reward=0.8,
        candidate_reward=0.9,
        mutation_path=("node:implement:temperature=0.0",),
        promotion_status="promoted",
        optimizer_config={"kind": "test"},
    )

    verified = verify_graph_bundle(bundle.root, require_promoted=True)
    assert verified.identity == bundle.identity
    assert verified.graph.version.startswith("0.3.0+opt.")
    assert verified.manifest["benchmark"]["improvement"] == pytest.approx(0.1)

    manifest = json.loads((bundle.root / "manifest.json").read_text())
    assert manifest["graph"]["sha256"]
    assert manifest["compiled"]["sha256"]
    assert manifest["bundle_sha256"]


def test_graph_bundle_rejects_candidate_and_tampering(tmp_path) -> None:
    graph = load_default_graph()
    bundle = write_graph_bundle(
        graph=graph,
        output_dir=tmp_path / "candidate",
        benchmark_report={"status": "candidate"},
        baseline_reward=1.0,
        candidate_reward=1.0,
        mutation_path=(),
        promotion_status="candidate",
        optimizer_config={"kind": "test"},
    )
    assert bundle.graph.version == graph.version

    with pytest.raises(GraphBundleError, match="not promoted"):
        verify_graph_bundle(bundle.root, require_promoted=True)

    graph_path = bundle.root / "graph.yaml"
    graph_path.write_text(graph_path.read_text() + "\n# tampered\n", encoding="utf-8")
    with pytest.raises(GraphBundleError, match="graph file digest mismatch"):
        verify_graph_bundle(bundle.root)
