import json
from dataclasses import dataclass, field

import pytest

from graph_model.controller import DeterministicGraphController, RouteDecision
from graph_model.graph import load_default_graph
from graph_model.qualification import qualify_mlx_host


@dataclass
class FakeReference:
    def as_dict(self):
        return {
            "path": "/tmp/projected-hidden.json",
            "sha256": "a" * 64,
            "format": "graph-native-hidden-state-v1",
            "feature_size": 3,
            "model_fingerprint": "b" * 64,
            "extractor_schema_hash": "c" * 64,
        }


@dataclass
class FakeObservation:
    features: tuple[float, ...] = (0.2, -0.3, 0.4)
    reference: FakeReference = field(default_factory=FakeReference)
    cache_hit: bool = False


class FakeProvider:
    loaded = False
    closed = False
    hidden_state_identity = "fake-hidden-state"

    @property
    def identity(self):
        return {"kind": "fake-mlx", "model": "fake-qwen", "revision": "abc123"}

    def load(self):
        self.loaded = True

    async def complete_json(self, *, system, user, temperature=None):
        del system, user, temperature
        return {"ok": True, "runtime": "mlx", "purpose": "qualification"}, 8, 5

    def capture_policy_hidden(self, *, state, node_id, decision_type):
        del state, node_id, decision_type
        return FakeObservation()

    def close(self):
        self.closed = True


class FakeController:
    identity = {"kind": "fake-mlx-controller"}

    def __init__(self):
        self.delegate = DeterministicGraphController()

    def select_route(self, task, state):
        del task, state
        return RouteDecision(
            route="fast",
            confidence=1.0,
            probabilities={"fast": 1.0, "deep": 0.0, "repair": 0.0},
            source="test",
            rule_route="fast",
        )

    def select_stop(self, **kwargs):
        return self.delegate.select_stop(**kwargs)

    def select_edge(self, **kwargs):
        return self.delegate.select_edge(**kwargs)


@pytest.mark.asyncio
async def test_mac_qualification_writes_complete_portable_evidence(tmp_path) -> None:
    provider = FakeProvider()

    report = await qualify_mlx_host(
        graph=load_default_graph(),
        output_dir=tmp_path,
        provider_factory=lambda: provider,
        controller_factory=lambda graph, **kwargs: FakeController(),
        diagnostics_factory=lambda **kwargs: {
            "platform": {"system": "test", "machine": "arm64"},
            "mlx": {"installed": True, "metal_available": True},
            "mlx_lm": {"installed": True},
            "configuration": {"model": "fake-qwen"},
        },
        require_apple_silicon=False,
    )

    assert report["passed"] is True
    assert [stage["name"] for stage in report["stages"]] == [
        "platform-and-configuration",
        "model-load",
        "structured-generation",
        "qwen-hidden-capture",
        "mlx-hard-masked-controller",
        "provider-close",
    ]
    assert report["hidden_reference"]["feature_size"] == 3
    assert report["security"]["raw_hidden_tensors_persisted"] is False
    assert provider.closed is True

    json_path = tmp_path / "mlx-m5-qualification.json"
    markdown_path = tmp_path / "mlx-m5-qualification.md"
    assert json_path.is_file()
    assert markdown_path.is_file()
    persisted = json.loads(json_path.read_text())
    assert persisted["passed"] is True
    assert persisted["artifacts"]["json"] == str(json_path)
    assert "PASS" in markdown_path.read_text()
