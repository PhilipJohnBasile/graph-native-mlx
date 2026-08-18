import pytest

from graph_model.conditions import ConditionError, evaluate_condition


def test_condition_evaluator_supports_nested_state_and_membership() -> None:
    context = {"data": {"route": "deep", "repair_count": 1}}
    assert evaluate_condition("data.route in {'deep', 'repair'}", context)
    assert evaluate_condition("data.repair_count < 2", context)


def test_condition_evaluator_short_circuits_boolean_expressions() -> None:
    assert evaluate_condition("False and data.missing < 2", {"data": {}}) is False
    assert evaluate_condition("True or data.missing < 2", {"data": {}}) is True


def test_condition_evaluator_rejects_calls_and_dunder_access() -> None:
    with pytest.raises(ConditionError):
        evaluate_condition("data.get('x')", {"data": {"x": 1}})
    assert evaluate_condition("data.__class__ is None", {"data": {}})
