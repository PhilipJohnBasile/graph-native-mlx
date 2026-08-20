from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


CANDIDATE_V2_FORMAT_VERSION = 2
_VALID_DECISION_TYPES = {"route", "edge", "stop"}


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result != result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{name} must be finite")
    return result


def _normalized(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def action_imitation_weight(
    *,
    valid_choice_count: int,
    advantage: float,
    terminal_correct: bool,
    false_success: bool,
) -> float:
    """Return the candidate-v2 action-learning weight.

    Forced decisions are telemetry only. A decision can teach the action head
    only when at least two valid choices existed, the terminal outcome was
    mechanically correct, no false success occurred, and the selected action
    had positive measured advantage over the static baseline.
    """

    if valid_choice_count < 2 or not terminal_correct or false_success:
        return 0.0
    return max(0.0, _finite_float(advantage, name="advantage"))


@dataclass(frozen=True)
class CandidateV2Decision:
    run_id: str
    repository_id: str
    task_family: str
    node_id: str
    decision_type: str
    static_action: str
    selected_action: str
    valid_actions: tuple[str, ...]
    terminal_correct: bool
    false_success: bool
    bounded_failure_correct: bool
    reward: float
    advantage: float
    normalized_tokens: float
    normalized_active_time: float
    normalized_llm_calls: float
    normalized_tool_calls: float
    normalized_repairs: float
    normalized_steps: float
    hidden_state_schema_hash: str = ""
    model_fingerprint: str = ""
    hidden_artifact_sha256: str = ""
    format_version: int = CANDIDATE_V2_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != CANDIDATE_V2_FORMAT_VERSION:
            raise ValueError(
                f"candidate-v2 format_version must be {CANDIDATE_V2_FORMAT_VERSION}"
            )
        for name, value in (
            ("run_id", self.run_id),
            ("repository_id", self.repository_id),
            ("task_family", self.task_family),
            ("node_id", self.node_id),
            ("static_action", self.static_action),
            ("selected_action", self.selected_action),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.decision_type not in _VALID_DECISION_TYPES:
            raise ValueError(
                "decision_type must be one of 'route', 'edge', or 'stop'"
            )
        if not self.valid_actions:
            raise ValueError("valid_actions must contain at least one action")
        if len(set(self.valid_actions)) != len(self.valid_actions):
            raise ValueError("valid_actions must not contain duplicates")
        if self.static_action not in self.valid_actions:
            raise ValueError("static_action must be present in valid_actions")
        if self.selected_action not in self.valid_actions:
            raise ValueError("selected_action must be present in valid_actions")
        object.__setattr__(self, "reward", _normalized(self.reward, name="reward"))
        object.__setattr__(self, "advantage", _finite_float(self.advantage, name="advantage"))
        for name in (
            "normalized_tokens",
            "normalized_active_time",
            "normalized_llm_calls",
            "normalized_tool_calls",
            "normalized_repairs",
            "normalized_steps",
        ):
            object.__setattr__(self, name, _normalized(getattr(self, name), name=name))

    @property
    def valid_choice_count(self) -> int:
        return len(self.valid_actions)

    @property
    def is_forced(self) -> bool:
        return self.valid_choice_count == 1

    @property
    def choice_changed(self) -> bool:
        return self.selected_action != self.static_action

    @property
    def action_weight(self) -> float:
        return action_imitation_weight(
            valid_choice_count=self.valid_choice_count,
            advantage=self.advantage,
            terminal_correct=self.terminal_correct,
            false_success=self.false_success,
        )

    @property
    def group_key(self) -> tuple[str, str]:
        """Leak-resistant train/validation grouping key."""

        return (self.repository_id, self.task_family)

    @property
    def normalized_cost(self) -> float:
        """Equal-weight diagnostic cost used by counterfactual corpus tooling.

        The training pipeline may later choose a different calibrated objective;
        keeping the raw normalized components in the record prevents that choice
        from being baked into the data format.
        """

        values = (
            self.normalized_tokens,
            self.normalized_active_time,
            self.normalized_llm_calls,
            self.normalized_tool_calls,
            self.normalized_repairs,
            self.normalized_steps,
        )
        return sum(values) / len(values)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["valid_actions"] = list(self.valid_actions)
        payload["valid_choice_count"] = self.valid_choice_count
        payload["is_forced"] = self.is_forced
        payload["choice_changed"] = self.choice_changed
        payload["action_imitation_weight"] = self.action_weight
        payload["normalized_cost"] = self.normalized_cost
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CandidateV2Decision":
        return cls(
            run_id=str(payload["run_id"]),
            repository_id=str(payload["repository_id"]),
            task_family=str(payload["task_family"]),
            node_id=str(payload["node_id"]),
            decision_type=str(payload["decision_type"]),
            static_action=str(payload["static_action"]),
            selected_action=str(payload["selected_action"]),
            valid_actions=tuple(str(value) for value in payload["valid_actions"]),
            terminal_correct=bool(payload["terminal_correct"]),
            false_success=bool(payload["false_success"]),
            bounded_failure_correct=bool(payload["bounded_failure_correct"]),
            reward=float(payload["reward"]),
            advantage=float(payload["advantage"]),
            normalized_tokens=float(payload["normalized_tokens"]),
            normalized_active_time=float(payload["normalized_active_time"]),
            normalized_llm_calls=float(payload["normalized_llm_calls"]),
            normalized_tool_calls=float(payload["normalized_tool_calls"]),
            normalized_repairs=float(payload["normalized_repairs"]),
            normalized_steps=float(payload["normalized_steps"]),
            hidden_state_schema_hash=str(payload.get("hidden_state_schema_hash", "")),
            model_fingerprint=str(payload.get("model_fingerprint", "")),
            hidden_artifact_sha256=str(payload.get("hidden_artifact_sha256", "")),
            format_version=int(payload.get("format_version", CANDIDATE_V2_FORMAT_VERSION)),
        )


def discriminative_records(
    records: Iterable[CandidateV2Decision],
) -> tuple[CandidateV2Decision, ...]:
    """Return only decisions eligible to influence an action head."""

    return tuple(record for record in records if record.action_weight > 0.0)


def grouped_by_task(
    records: Sequence[CandidateV2Decision],
) -> dict[tuple[str, str], tuple[CandidateV2Decision, ...]]:
    grouped: dict[tuple[str, str], list[CandidateV2Decision]] = {}
    for record in records:
        grouped.setdefault(record.group_key, []).append(record)
    return {key: tuple(values) for key, values in grouped.items()}
