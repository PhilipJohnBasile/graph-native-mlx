from __future__ import annotations

from graph_model.graph import load_default_graph
from graph_model.models import RunState
from graph_model.paired_eval import (
    canonical_json,
    canonicalize_prompt_value,
    deterministic_generation_seed,
    evaluation_state_payload,
    logical_idempotency_key,
    prompt_audit_record,
)


def _state(run_id: str, source: str, active: str, *, elapsed: float = 0.0) -> RunState:
    graph = load_default_graph()
    state = RunState.new(
        graph=graph,
        task="Implement the paired evaluation contract",
        run_id=run_id,
        initial_data={
            "_paired_evaluation": evaluation_state_payload(
                case_id="same-case",
                repository_alias="<repository:same-case>",
                base_seed=42057,
            ),
            "workspace": {
                "source_root": source,
                "active_root": active,
                "artifact_root": active + "/artifacts",
            },
            "trace_manifest": {"manifest_repo": source},
        },
    )
    state.metrics.elapsed_seconds = elapsed
    return state


def test_paired_prompt_canonicalization_hides_runtime_identity() -> None:
    state = _state(
        "static-arm-run",
        "/tmp/static/source",
        "/tmp/static/worktree",
    )
    payload = {
        "run": state.run_id,
        "workspace": state.data["workspace"],
        "message": "read /tmp/static/worktree/pkg.py from /tmp/static/source",
    }
    canonical = canonicalize_prompt_value(payload, state)
    serialized = canonical_json(payload, state)

    assert canonical["run"] == "<run:same-case>"
    assert canonical["workspace"]["source_root"] == "<repository:same-case>"
    assert canonical["workspace"]["active_root"] == "<worktree:same-case>"
    assert "/tmp/static" not in serialized
    assert "static-arm-run" not in serialized


def test_paired_seed_is_stable_across_arm_specific_paths_and_run_ids() -> None:
    static = _state("static-run", "/tmp/a/repo", "/tmp/a/worktree", elapsed=1.5)
    shadow = _state("shadow-run", "/tmp/b/repo", "/tmp/b/worktree", elapsed=99.0)
    raw_static = {
        "context": static.data["workspace"],
        "run_id": static.run_id,
    }
    raw_shadow = {
        "context": shadow.data["workspace"],
        "run_id": shadow.run_id,
    }
    system = "paired prompt"
    static_user = canonical_json(raw_static, static)
    shadow_user = canonical_json(raw_shadow, shadow)
    assert static_user == shadow_user

    static_seed = deterministic_generation_seed(
        static,
        node_id="implement",
        call_kind="implement",
        revision=0,
        system=system,
        user=static_user,
    )
    shadow_seed = deterministic_generation_seed(
        shadow,
        node_id="implement",
        call_kind="implement",
        revision=0,
        system=system,
        user=shadow_user,
    )
    assert static_seed == shadow_seed


def test_prompt_audit_persists_only_hashes_and_seed() -> None:
    state = _state("arm-run", "/tmp/a/repo", "/tmp/a/worktree")
    record = prompt_audit_record(
        state,
        node_id="review",
        call_kind="review-initial",
        revision=0,
        system="SECRET SYSTEM",
        user="SECRET USER",
        seed=123,
    )
    assert record is not None
    assert record["generation_seed"] == 123
    assert record["raw_system_persisted"] is False
    assert record["raw_user_persisted"] is False
    assert "SECRET" not in str(record)


def test_paired_idempotency_key_uses_logical_case_identity() -> None:
    state = _state("arm-run", "/tmp/a/repo", "/tmp/a/worktree")
    assert logical_idempotency_key(
        state,
        node_id="implement",
        revision=2,
        fallback="raw-run:path:hash",
    ) == "same-case:implement:revision-2"
