from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from graph_model.models import GraphSpec

from .decision import MLXUnavailableError
from .graph_tables import CompiledGraphTables, compile_graph
from .policy import GraphPolicyConfig, GraphPolicyHeads
from .training_data import (
    PolicyTrainingRecord,
    dataset_identity,
    read_policy_training_data,
)


@dataclass(frozen=True)
class TrainingSummary:
    records: int
    train_records: int
    validation_records: int
    epochs_requested: int
    epochs_completed: int
    initial_train_loss: float
    final_train_loss: float
    best_validation_loss: float | None
    best_epoch: int | None
    uses_hidden_states: bool
    hidden_feature_size: int
    hidden_state_schema_hash: str
    model_fingerprint: str
    weights_path: str
    config_path: str


def policy_config_for_records(
    *,
    tables: CompiledGraphTables,
    records: Sequence[PolicyTrainingRecord],
    hidden_size: int,
) -> GraphPolicyConfig:
    identity = dataset_identity(list(records))
    return GraphPolicyConfig.for_graph(
        tables,
        hidden_size=hidden_size,
        backbone_feature_size=identity.hidden_feature_size,
        hidden_state_schema_hash=identity.hidden_state_schema_hash,
        model_fingerprint=identity.model_fingerprint,
    )


def action_imitation_weights(
    records: Sequence[PolicyTrainingRecord],
) -> tuple[float, ...]:
    """Return per-decision imitation weights derived from terminal reward.

    Successful executions currently carry reward 1.0 and may teach the
    selected route/edge/stop action. Failed executions carry reward 0.0 and
    therefore train only value and cost targets, not action imitation.
    Fractional rewards remain supported for future graded outcomes.
    """

    return tuple(float(record.reward) for record in records)


def split_policy_records(
    records: Sequence[PolicyTrainingRecord],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[PolicyTrainingRecord], list[PolicyTrainingRecord]]:
    """Split by run ID so decisions from one execution never cross train/validation."""

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    items = list(records)
    if not items or validation_fraction == 0.0:
        return items, []
    run_ids = sorted({record.run_id for record in items})
    if len(run_ids) < 2:
        return items, []

    scores = {
        run_id: int(
            hashlib.sha256(f"{seed}:{run_id}".encode("utf-8")).hexdigest()[:16],
            16,
        )
        / float(0xFFFFFFFFFFFFFFFF)
        for run_id in run_ids
    }
    validation_runs = {
        run_id for run_id, score in scores.items() if score < validation_fraction
    }
    if not validation_runs:
        validation_runs = {min(run_ids, key=lambda run_id: scores[run_id])}
    if validation_runs == set(run_ids):
        validation_runs.remove(max(run_ids, key=lambda run_id: scores[run_id]))

    train = [record for record in items if record.run_id not in validation_runs]
    validation = [record for record in items if record.run_id in validation_runs]
    if not train:
        raise ValueError("validation split left no training records")
    return train, validation


def train_mlx_policy_file(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    graph: GraphSpec,
    hidden_size: int = 128,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    validation_fraction: float = 0.15,
    patience: int = 15,
    require_hidden: bool = False,
    seed: int = 42,
) -> TrainingSummary:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if hidden_size < 8:
        raise ValueError("hidden_size must be >= 8")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except ImportError as exc:  # pragma: no cover - Apple Silicon only
        raise MLXUnavailableError("MLX policy training requires the mlx package") from exc

    tables = compile_graph(graph)
    records = read_policy_training_data(
        input_path,
        tables=tables,
        require_hidden=require_hidden,
    )
    identity = dataset_identity(records)
    config = policy_config_for_records(
        tables=tables,
        records=records,
        hidden_size=hidden_size,
    )
    train_records, validation_records = split_policy_records(
        records,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    mx.random.seed(seed)
    model = GraphPolicyHeads(config)
    optimizer = optim.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    def arrays_for(rows: Sequence[PolicyTrainingRecord]):
        explicit = mx.array([list(record.features) for record in rows], dtype=mx.float32)
        if identity.uses_hidden_states:
            backbone = mx.array(
                [list(record.hidden_features) for record in rows],
                dtype=mx.float32,
            )
        else:
            backbone = mx.zeros((len(rows), 0), dtype=mx.float32)
        return (
            explicit,
            backbone,
            mx.array([record.route_label for record in rows], dtype=mx.int32),
            mx.array([record.edge_label for record in rows], dtype=mx.int32),
            mx.array([record.stop_label for record in rows], dtype=mx.int32),
            mx.array(
                [list(record.allowed_edge_mask) for record in rows],
                dtype=mx.bool_,
            ),
            mx.array(
                [list(record.allowed_stop_mask) for record in rows],
                dtype=mx.bool_,
            ),
            mx.array(action_imitation_weights(rows), dtype=mx.float32),
            mx.array([record.reward for record in rows], dtype=mx.float32),
            mx.array([list(record.cost_target) for record in rows], dtype=mx.float32),
        )

    train_arrays = arrays_for(train_records)
    validation_arrays = arrays_for(validation_records) if validation_records else None

    def masked_cross_entropy(logits, labels, imitation_weight):
        valid = labels >= 0
        safe_labels = mx.where(valid, labels, mx.zeros_like(labels))
        losses = nn.losses.cross_entropy(logits, safe_labels, reduction="none")
        effective_weight = valid.astype(mx.float32) * imitation_weight
        return (losses * effective_weight).sum() / mx.maximum(
            effective_weight.sum(),
            1.0,
        )

    def loss_fn(
        model,
        explicit,
        backbone,
        route_y,
        edge_y,
        stop_y,
        edge_allowed,
        stop_allowed,
        imitation_weight_y,
        reward_y,
        cost_y,
    ):
        route_logits, edge_logits, stop_logits, value, cost = model(
            explicit,
            backbone if identity.uses_hidden_states else None,
        )
        masked_edge_logits = mx.where(
            edge_allowed,
            edge_logits,
            mx.full_like(edge_logits, -1e9),
        )
        masked_stop_logits = mx.where(
            stop_allowed,
            stop_logits,
            mx.full_like(stop_logits, -1e9),
        )
        route_loss = masked_cross_entropy(
            route_logits,
            route_y,
            imitation_weight_y,
        )
        edge_loss = masked_cross_entropy(
            masked_edge_logits,
            edge_y,
            imitation_weight_y,
        )
        stop_loss = masked_cross_entropy(
            masked_stop_logits,
            stop_y,
            imitation_weight_y,
        )
        value_prediction = mx.sigmoid(value[:, 0])
        value_loss = ((value_prediction - reward_y) ** 2).mean()
        cost_prediction = nn.softplus(cost)
        cost_loss = ((cost_prediction - cost_y) ** 2).mean()
        return route_loss + edge_loss + stop_loss + 0.5 * value_loss + 0.25 * cost_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_weights = output / ".graph_policy.best.safetensors"
    final_weights = output / "graph_policy.safetensors"
    config_path = output / "graph_policy.json"

    initial_train_loss = 0.0
    final_train_loss = 0.0
    best_validation_loss = math.inf
    stale_epochs = 0
    epochs_completed = 0
    best_epoch: int | None = None

    for epoch in range(epochs):
        loss, gradients = loss_and_grad(model, *train_arrays)
        optimizer.update(model, gradients)
        mx.eval(loss, model.parameters(), optimizer.state)
        train_loss = float(loss.item())
        if not math.isfinite(train_loss):
            raise ValueError("policy training loss became non-finite")
        if epoch == 0:
            initial_train_loss = train_loss
        final_train_loss = train_loss
        epochs_completed = epoch + 1

        if validation_arrays is None:
            continue
        validation_loss_array = loss_fn(model, *validation_arrays)
        mx.eval(validation_loss_array)
        validation_loss = float(validation_loss_array.item())
        if not math.isfinite(validation_loss):
            raise ValueError("policy validation loss became non-finite")
        if validation_loss + 1e-7 < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            stale_epochs = 0
            model.save_weights(str(best_weights))
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if validation_arrays is not None:
        if not best_weights.exists():
            model.save_weights(str(best_weights))
            validation_loss_array = loss_fn(model, *validation_arrays)
            mx.eval(validation_loss_array)
            best_validation_loss = float(validation_loss_array.item())
            best_epoch = epochs_completed
        os.replace(best_weights, final_weights)
        model.load_weights(str(final_weights), strict=True)
    else:
        model.save_weights(str(final_weights))
        best_validation_loss = math.inf

    # Report the losses for the weights that are actually deployed. With early
    # stopping this is the restored best-validation checkpoint, not the final
    # optimization step that happened to run.
    mx.eval(model.parameters())
    deployed_train_loss = loss_fn(model, *train_arrays)
    mx.eval(deployed_train_loss)
    final_train_loss = float(deployed_train_loss.item())
    if not math.isfinite(final_train_loss):
        raise ValueError("deployed policy training loss is non-finite")
    if validation_arrays is not None:
        deployed_validation_loss = loss_fn(model, *validation_arrays)
        mx.eval(deployed_validation_loss)
        best_validation_loss = float(deployed_validation_loss.item())
        if not math.isfinite(best_validation_loss):
            raise ValueError("deployed policy validation loss is non-finite")

    config.save(config_path)
    summary = TrainingSummary(
        records=len(records),
        train_records=len(train_records),
        validation_records=len(validation_records),
        epochs_requested=epochs,
        epochs_completed=epochs_completed,
        initial_train_loss=initial_train_loss,
        final_train_loss=final_train_loss,
        best_validation_loss=(
            None if not validation_records else float(best_validation_loss)
        ),
        best_epoch=best_epoch,
        uses_hidden_states=identity.uses_hidden_states,
        hidden_feature_size=identity.hidden_feature_size,
        hidden_state_schema_hash=identity.hidden_state_schema_hash,
        model_fingerprint=identity.model_fingerprint,
        weights_path=str(final_weights.resolve()),
        config_path=str(config_path.resolve()),
    )
    (output / "training_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return summary
