from graph_model.mlx_native.doctor import mlx_diagnostics


def test_mlx_doctor_returns_a_structured_report_without_mlx(monkeypatch) -> None:
    monkeypatch.delenv("GRAPH_MODEL_MLX_MODEL", raising=False)
    report = mlx_diagnostics()
    assert set(report) >= {"platform", "mlx", "mlx_lm", "configuration", "ready"}
    assert report["configuration"]["model"] is None
    assert report["ready"] is False
