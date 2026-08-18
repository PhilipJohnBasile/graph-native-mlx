from __future__ import annotations

from graph_model.mlx_native.trainer import split_policy_records
from graph_model.mlx_native.training_data import PolicyTrainingRecord


def _record(run_id: str, node_id: str) -> PolicyTrainingRecord:
    return PolicyTrainingRecord(
        run_id=run_id,
        node_id=node_id,
        decision_type="route",
        features=(0.0,),
        route_label=0,
        edge_label=-1,
        stop_label=-1,
        allowed_edge_mask=(False,),
        allowed_stop_mask=(False, False, False, False),
        reward=1.0,
        cost_target=(0.1, 0.2, 0.3),
    )


def test_policy_split_keeps_each_run_in_exactly_one_partition() -> None:
    records = [
        _record("run-a", "intake"),
        _record("run-a", "context"),
        _record("run-b", "intake"),
        _record("run-b", "context"),
        _record("run-c", "intake"),
        _record("run-c", "context"),
    ]
    train, validation = split_policy_records(
        records,
        validation_fraction=0.34,
        seed=42,
    )
    train_runs = {record.run_id for record in train}
    validation_runs = {record.run_id for record in validation}
    assert train_runs
    assert validation_runs
    assert train_runs.isdisjoint(validation_runs)
    assert train_runs | validation_runs == {"run-a", "run-b", "run-c"}


def test_policy_split_is_deterministic() -> None:
    records = [_record(f"run-{index}", "intake") for index in range(10)]
    first = split_policy_records(records, validation_fraction=0.2, seed=7)
    second = split_policy_records(records, validation_fraction=0.2, seed=7)
    assert first == second


def test_failed_runs_do_not_receive_action_imitation_weight() -> None:
    from graph_model.mlx_native.trainer import action_imitation_weights

    successful = _record("success", "intake")
    failed = PolicyTrainingRecord(
        run_id="failed",
        node_id="intake",
        decision_type="route",
        features=(0.0,),
        route_label=2,
        edge_label=-1,
        stop_label=-1,
        allowed_edge_mask=(False,),
        allowed_stop_mask=(False, False, False, False),
        reward=0.0,
        cost_target=(0.9, 0.8, 0.7),
    )

    assert action_imitation_weights([successful, failed]) == (1.0, 0.0)
