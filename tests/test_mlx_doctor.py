from graph_model.mlx_native.doctor import mlx_diagnostics


def test_mlx_doctor_returns_a_structured_report_without_mlx(monkeypatch) -> None:
    monkeypatch.delenv("GRAPH_MODEL_MLX_MODEL", raising=False)
    report = mlx_diagnostics()
    assert set(report) >= {"platform", "mlx", "mlx_lm", "configuration", "ready"}
    assert report["configuration"]["model"] is None
    assert report["configuration"]["model_path"]["value"] is None
    assert report["configuration"]["execution"] == "dedicated-single-worker"
    assert report["configuration"]["hidden_state"]["persists_raw_prompts"] is False
    assert report["configuration"]["hidden_state"]["persists_raw_hidden_tensors"] is False
    assert report["ready"] is False

def test_mlx_doctor_validates_hidden_policy_config(tmp_path) -> None:
    from graph_model.graph import load_default_graph
    from graph_model.mlx_native.doctor import _policy_report
    from graph_model.mlx_native.graph_tables import compile_graph
    from graph_model.mlx_native.policy import GraphPolicyConfig

    graph = load_default_graph()
    tables = compile_graph(graph)
    hidden_schema = "a" * 64
    model_fingerprint = "b" * 64

    config = GraphPolicyConfig.for_graph(
        tables,
        hidden_size=128,
        backbone_feature_size=256,
        hidden_state_schema_hash=hidden_schema,
        model_fingerprint=model_fingerprint,
    )

    config_path = config.save(tmp_path / "graph_policy.json")
    weights_path = tmp_path / "graph_policy.safetensors"
    weights_path.write_bytes(b"diagnostic-only-placeholder")

    report = _policy_report(
        str(weights_path),
        str(config_path),
        graph=graph,
        hidden_schema_hash=hidden_schema,
    )

    assert report["configured"] is True
    assert report["validation"]["ok"] is True
    assert report["validation"]["requires_hidden"] is True
    assert report["validation"]["hidden_feature_size"] == 256
    assert report["validation"]["hidden_state_schema_hash"] == hidden_schema
    assert report["validation"]["model_fingerprint"] == model_fingerprint


def test_mlx_doctor_rejects_hidden_policy_extractor_mismatch(tmp_path) -> None:
    from graph_model.graph import load_default_graph
    from graph_model.mlx_native.doctor import _policy_report
    from graph_model.mlx_native.graph_tables import compile_graph
    from graph_model.mlx_native.policy import GraphPolicyConfig

    graph = load_default_graph()
    tables = compile_graph(graph)

    config = GraphPolicyConfig.for_graph(
        tables,
        hidden_size=128,
        backbone_feature_size=256,
        hidden_state_schema_hash="a" * 64,
        model_fingerprint="b" * 64,
    )

    config_path = config.save(tmp_path / "graph_policy.json")
    weights_path = tmp_path / "graph_policy.safetensors"
    weights_path.write_bytes(b"diagnostic-only-placeholder")

    report = _policy_report(
        str(weights_path),
        str(config_path),
        graph=graph,
        hidden_schema_hash="c" * 64,
    )

    assert report["validation"]["ok"] is False
    assert "does not match" in report["validation"]["error"]

