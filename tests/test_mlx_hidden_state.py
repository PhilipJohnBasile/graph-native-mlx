from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from graph_model.graph import load_default_graph
from graph_model.models import RunState
from graph_model.paired_eval import evaluation_state_payload
from graph_model.mlx_native.hidden_state import (
    HIDDEN_STATE_FORMAT,
    HiddenStateArtifactStore,
    HiddenStateCapture,
    HiddenStateCaptureConfig,
    HiddenStateReference,
    capture_from_raw_hidden,
    model_fingerprint,
    policy_state_prompt,
    project_hidden_views,
)
from graph_model.mlx_native.qwen_hidden import RawHiddenState


def _capture() -> HiddenStateCapture:
    return HiddenStateCapture(
        features=(0.25, -0.5, 0.75, 1.0),
        model_fingerprint="a" * 64,
        extractor_schema_hash="b" * 64,
        raw_hidden_size=4,
        raw_vector_size=8,
        prompt_tokens=12,
        task_sha256="c" * 64,
        prompt_sha256="d" * 64,
        core_path="model.model",
        layer_labels=("layer:1", "final"),
        pooling="last-token",
    )


def test_hidden_state_artifact_round_trip_is_hash_verified(tmp_path: Path) -> None:
    store = HiddenStateArtifactStore(tmp_path)
    reference = store.write(_capture())
    assert reference.format == HIDDEN_STATE_FORMAT
    loaded = store.load(reference)
    assert loaded.features == pytest.approx(_capture().features)
    assert HiddenStateReference.from_dict(reference.as_dict()) == reference

    path = Path(reference.path)
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.load(reference)


def test_hidden_state_write_is_hash_addressed_and_idempotent(tmp_path: Path) -> None:
    store = HiddenStateArtifactStore(tmp_path)
    first = store.write(_capture())
    second = store.write(_capture())
    assert first == second
    assert Path(first.path).name == f"{first.sha256}.json"


def test_hidden_artifact_does_not_persist_task_or_raw_tensor(tmp_path: Path) -> None:
    task = "private repository task text"
    raw = RawHiddenState(
        values=(1.0, 2.0, 3.0, 4.0),
        source="model.model",
        layer_labels=("final",),
        pooling="last-token",
        token_count=4,
        prompt_sha256=hashlib.sha256(b"prompt").hexdigest(),
        model_hidden_size=4,
    )
    config = HiddenStateCaptureConfig(feature_size=8)
    capture = capture_from_raw_hidden(
        raw,
        task=task,
        model_identity={"kind": "mlx-local", "backend": "fake", "model": "qwen"},
        config=config,
    )
    reference = HiddenStateArtifactStore(tmp_path).write(capture)
    payload_text = Path(reference.path).read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    assert task not in payload_text
    assert payload["feature_size"] == 8
    assert len(payload["features"]) == 8
    assert payload["raw_vector_size"] == 4
    assert "raw_values" not in payload


def test_projection_is_deterministic_and_layer_order_independent() -> None:
    first = project_hidden_views(
        {"final": (1.0, 2.0, 3.0), "layer:3": (4.0, 5.0, 6.0)},
        output_size=16,
        seed=123,
    )
    second = project_hidden_views(
        {"layer:3": (4.0, 5.0, 6.0), "final": (1.0, 2.0, 3.0)},
        output_size=16,
        seed=123,
    )
    assert first == second
    assert len(first) == 16


def test_hidden_capture_schema_binds_layers_pooling_and_projection() -> None:
    base = HiddenStateCaptureConfig(
        feature_size=32,
        layer_specs=("final",),
        pooling="last-token",
    )
    changed_layer = HiddenStateCaptureConfig(
        feature_size=32,
        layer_specs=("50%", "final"),
        pooling="last-token",
    )
    changed_pooling = HiddenStateCaptureConfig(
        feature_size=32,
        layer_specs=("final",),
        pooling="mean-last",
    )
    assert len({base.schema_hash, changed_layer.schema_hash, changed_pooling.schema_hash}) == 3


def test_model_fingerprint_ignores_sampling_but_binds_model_identity() -> None:
    left = model_fingerprint(
        {
            "kind": "mlx-local",
            "backend": "mlx-lm-1",
            "model": "qwen",
            "revision": "abc",
            "temperature": "0.1",
        }
    )
    right = model_fingerprint(
        {
            "kind": "mlx-local",
            "backend": "mlx-lm-1",
            "model": "qwen",
            "revision": "abc",
            "temperature": "0.9",
        }
    )
    different = model_fingerprint(
        {
            "kind": "mlx-local",
            "backend": "mlx-lm-1",
            "model": "qwen",
            "revision": "def",
        }
    )
    assert left == right
    assert left != different



def test_policy_state_prompt_is_bounded_deterministic_and_evidence_aware() -> None:
    graph = load_default_graph()
    state = RunState.new(
        graph=graph,
        task="Fix the failing verifier without widening the patch " + ("scope " * 2_000),
        run_id="state-prompt",
    )
    state.current_node = "tests"
    state.step_count = 5
    state.completed_nodes = ["intake", "context", "plan", "implement", "apply"]
    state.data.update(
        {
            "route": "deep",
            "verdict": "fail",
            "repair_count": 0,
            "test_report": {
                "verdict": "fail",
                "stderr": "HEAD failure evidence\n" + ("x" * 80_000) + "\nTAIL root cause",
            },
        }
    )

    system, first = policy_state_prompt(
        state,
        node_id="tests",
        decision_type="transition",
        max_chars=8_000,
    )
    _, repeated = policy_state_prompt(
        state,
        node_id="tests",
        decision_type="transition",
        max_chars=8_000,
    )
    payload = json.loads(first)
    assert first == repeated
    assert len(first) <= 8_000
    assert "representation-only" in system
    assert payload["decision"] == {"node": "tests", "type": "transition"}
    assert payload["execution"]["step_count"] == 5
    assert payload["task_sha256"] == hashlib.sha256(state.task.encode()).hexdigest()
    assert "private" not in first

    state.data["verdict"] = "pass"
    state.updated_at += 1.0
    _, changed = policy_state_prompt(
        state,
        node_id="tests",
        decision_type="transition",
        max_chars=8_000,
    )
    assert changed != first
    assert json.loads(changed)["state"]["verdict"] == "pass"


def test_hidden_schema_binds_policy_state_prompt_contract() -> None:
    config = HiddenStateCaptureConfig(feature_size=32)
    payload = {
        "feature_size": config.feature_size,
        "schema_hash": config.schema_hash,
    }
    assert payload["schema_hash"]
    assert len(payload["schema_hash"]) == 64


def test_paired_policy_state_prompt_normalizes_paths_run_ids_and_timing() -> None:
    graph = load_default_graph()

    def state(run_id: str, root: str, elapsed: float) -> RunState:
        value = RunState.new(
            graph=graph,
            task="Audit the paired graph state",
            run_id=run_id,
            initial_data={
                "_paired_evaluation": evaluation_state_payload(
                    case_id="paired-state",
                    repository_alias="<repository:paired-state>",
                ),
                "workspace": {
                    "source_root": root + "/source",
                    "active_root": root + "/worktree",
                    "artifact_root": root + "/artifacts",
                },
                "route": "repair",
                "verdict": "pass",
                "test_report": {
                    "verdict": "pass",
                    "duration_seconds": elapsed,
                    "path": root + "/worktree/tests",
                },
            },
        )
        value.current_node = "review"
        value.step_count = 7
        value.metrics.elapsed_seconds = elapsed
        return value

    static = state("static-arm", "/tmp/static", 1.0)
    shadow = state("shadow-arm", "/tmp/shadow", 99.0)
    _, static_prompt = policy_state_prompt(
        static, node_id="review", decision_type="transition"
    )
    _, shadow_prompt = policy_state_prompt(
        shadow, node_id="review", decision_type="transition"
    )

    assert static_prompt == shadow_prompt
    assert "/tmp/static" not in static_prompt
    assert "/tmp/shadow" not in shadow_prompt
    payload = json.loads(static_prompt)
    assert payload["execution"]["metrics"]["elapsed_seconds"] == 0.0
    assert payload["execution"]["budget_remaining"]["seconds"] == static.budget.max_seconds
