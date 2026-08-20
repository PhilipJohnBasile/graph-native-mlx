from __future__ import annotations

import pytest

from graph_model.mlx_native.candidate_v2_data import (
    CandidateV2Decision,
    action_imitation_weight,
    discriminative_records,
    grouped_by_task,
)


def _decision(**overrides) -> CandidateV2Decision:
    payload = {
        "run_id": "run-1",
        "repository_id": "repo-a",
        "task_family": "deep-parser",
        "node_id": "context",
        "decision_type": "route",
        "static_action": "fast",
        "selected_action": "deep",
        "valid_actions": ("fast", "deep", "repair"),
        "terminal_correct": True,
        "false_success": False,
        "bounded_failure_correct": False,
        "reward": 1.0,
        "advantage": 0.25,
        "normalized_tokens": 0.2,
        "normalized_active_time": 0.3,
        "normalized_llm_calls": 0.2,
        "normalized_tool_calls": 0.1,
        "normalized_repairs": 0.0,
        "normalized_steps": 0.2,
    }
    payload.update(overrides)
    return CandidateV2Decision(**payload)


def test_forced_decision_cannot_train_action_imitation() -> None:
    record = _decision(
        static_action="fast",
        selected_action="fast",
        valid_actions=("fast",),
        advantage=1.0,
    )

    assert record.is_forced is True
    assert record.valid_choice_count == 1
    assert record.action_weight == 0.0


def test_failed_or_false_success_decision_cannot_train_action_imitation() -> None:
    assert _decision(terminal_correct=False).action_weight == 0.0
    assert _decision(false_success=True).action_weight == 0.0


def test_only_positive_measured_advantage_trains_action_head() -> None:
    assert _decision(advantage=-0.5).action_weight == 0.0
    assert _decision(advantage=0.0).action_weight == 0.0
    assert _decision(advantage=0.4).action_weight == pytest.approx(0.4)


def test_action_imitation_weight_is_independent_of_terminal_reward_scale() -> None:
    record = _decision(reward=0.6, advantage=0.3)

    assert record.reward == pytest.approx(0.6)
    assert record.action_weight == pytest.approx(0.3)


def test_group_key_uses_repository_and_task_family() -> None:
    first = _decision(run_id="run-1")
    second = _decision(run_id="run-2")
    other_family = _decision(run_id="run-3", task_family="repair-cache")

    grouped = grouped_by_task((first, second, other_family))

    assert set(grouped) == {
        ("repo-a", "deep-parser"),
        ("repo-a", "repair-cache"),
    }
    assert {record.run_id for record in grouped[("repo-a", "deep-parser")]} == {
        "run-1",
        "run-2",
    }


def test_discriminative_records_excludes_forced_and_nonadvantageous_rows() -> None:
    useful = _decision(run_id="useful", advantage=0.2)
    forced = _decision(
        run_id="forced",
        static_action="fast",
        selected_action="fast",
        valid_actions=("fast",),
        advantage=1.0,
    )
    neutral = _decision(run_id="neutral", advantage=0.0)

    assert discriminative_records((useful, forced, neutral)) == (useful,)


def test_json_payload_contains_derived_audit_fields_and_round_trips() -> None:
    record = _decision()

    payload = record.to_json_dict()

    assert payload["valid_choice_count"] == 3
    assert payload["is_forced"] is False
    assert payload["choice_changed"] is True
    assert payload["action_imitation_weight"] == pytest.approx(0.25)
    assert payload["normalized_cost"] == pytest.approx(1.0 / 6.0)

    reconstructed = CandidateV2Decision.from_mapping(payload)
    assert reconstructed == record


def test_selected_and_static_actions_must_be_valid() -> None:
    with pytest.raises(ValueError, match="static_action"):
        _decision(static_action="abort")
    with pytest.raises(ValueError, match="selected_action"):
        _decision(selected_action="abort")


def test_normalized_cost_inputs_are_bounded() -> None:
    with pytest.raises(ValueError, match="normalized_tokens"):
        _decision(normalized_tokens=1.1)


def test_pure_weight_helper_matches_record_rule() -> None:
    assert action_imitation_weight(
        valid_choice_count=3,
        advantage=0.7,
        terminal_correct=True,
        false_success=False,
    ) == pytest.approx(0.7)
    assert action_imitation_weight(
        valid_choice_count=1,
        advantage=0.7,
        terminal_correct=True,
        false_success=False,
    ) == 0.0
