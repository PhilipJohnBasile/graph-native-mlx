from pathlib import Path

import pytest

from graph_model.graph import load_default_graph
from graph_model.mlx_native.graph_tables import compile_graph
from graph_model.mlx_native.policy import GraphPolicyConfig


def test_policy_config_is_bound_to_graph_schema(tmp_path: Path) -> None:
    tables = compile_graph(load_default_graph())
    config = GraphPolicyConfig.for_graph(tables, hidden_size=96)
    path = config.save(tmp_path / "graph_policy.json")
    loaded = GraphPolicyConfig.load(path)
    loaded.validate(tables)
    assert loaded.hidden_size == 96
    assert loaded.edge_count == 19
    assert loaded.graph_schema_hash == tables.schema_hash


def test_policy_config_rejects_a_schema_mismatch() -> None:
    tables = compile_graph(load_default_graph())
    config = GraphPolicyConfig.for_graph(tables)
    incompatible = GraphPolicyConfig(
        **{**config.__dict__, "graph_schema_hash": "0" * 64}
    )
    with pytest.raises(ValueError, match="incompatible"):
        incompatible.validate(tables)


def test_policy_config_rejects_an_invalid_hidden_size() -> None:
    tables = compile_graph(load_default_graph())
    config = GraphPolicyConfig.for_graph(tables, hidden_size=4)
    with pytest.raises(ValueError, match="hidden_size"):
        config.validate(tables)


def test_hidden_policy_config_binds_model_and_extractor_identity(tmp_path: Path) -> None:
    tables = compile_graph(load_default_graph())
    config = GraphPolicyConfig.for_graph(
        tables,
        hidden_size=128,
        backbone_feature_size=256,
        hidden_state_schema_hash="a" * 64,
        model_fingerprint="b" * 64,
    )
    path = config.save(tmp_path / "hidden-policy.json")
    loaded = GraphPolicyConfig.load(path)
    loaded.validate(tables)
    assert loaded.requires_backbone_features is True
    assert loaded.backbone_feature_size == 256
    assert loaded.fusion_version == "gated-residual-v1"
    assert loaded.hidden_state_schema_hash == "a" * 64
    assert loaded.model_fingerprint == "b" * 64


def test_hidden_policy_config_rejects_invalid_identity_hashes() -> None:
    tables = compile_graph(load_default_graph())
    config = GraphPolicyConfig.for_graph(
        tables,
        backbone_feature_size=256,
        hidden_state_schema_hash="not-a-digest",
        model_fingerprint="b" * 64,
    )
    with pytest.raises(ValueError, match="hidden_state_schema_hash"):
        config.validate(tables)
