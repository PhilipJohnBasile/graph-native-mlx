from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from graph_model.models import GraphSpec

from .decision import MLXUnavailableError
from .graph_tables import compile_graph
from .policy import GraphPolicyConfig, GraphPolicyHeads
from .training_data import read_policy_training_data


@dataclass(frozen=True)
class TrainingSummary:
    records: int
    epochs: int
    initial_loss: float
    final_loss: float
    weights_path: str
    config_path: str


def train_mlx_policy_file(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    graph: GraphSpec,
    hidden_size: int = 128,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> TrainingSummary:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if hidden_size < 8:
        raise ValueError("hidden_size must be >= 8")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except ImportError as exc:  # pragma: no cover - executed on Apple Silicon
        raise MLXUnavailableError("MLX policy training requires the mlx package") from exc

    tables = compile_graph(graph)
    records = read_policy_training_data(input_path, tables=tables)
    config = GraphPolicyConfig.for_graph(tables, hidden_size=hidden_size)
    mx.random.seed(seed)
    model = GraphPolicyHeads(config)
    optimizer = optim.AdamW(learning_rate=learning_rate)

    features = mx.array([list(record.features) for record in records], dtype=mx.float32)
    route_labels = mx.array([record.route_label for record in records], dtype=mx.int32)
    edge_labels = mx.array([record.edge_label for record in records], dtype=mx.int32)
    stop_labels = mx.array([record.stop_label for record in records], dtype=mx.int32)
    edge_masks = mx.array(
        [list(record.allowed_edge_mask) for record in records], dtype=mx.bool_
    )
    stop_masks = mx.array(
        [list(record.allowed_stop_mask) for record in records], dtype=mx.bool_
    )
    rewards = mx.array([record.reward for record in records], dtype=mx.float32)
    costs = mx.array([list(record.cost_target) for record in records], dtype=mx.float32)

    def masked_cross_entropy(logits, labels):
        valid = labels >= 0
        safe_labels = mx.where(valid, labels, mx.zeros_like(labels))
        losses = nn.losses.cross_entropy(logits, safe_labels, reduction="none")
        valid_float = valid.astype(mx.float32)
        return (losses * valid_float).sum() / mx.maximum(valid_float.sum(), 1.0)

    def loss_fn(model, x, route_y, edge_y, stop_y, edge_allowed, stop_allowed, reward_y, cost_y):
        route_logits, edge_logits, stop_logits, value, cost = model(x)
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
        route_loss = masked_cross_entropy(route_logits, route_y)
        edge_loss = masked_cross_entropy(masked_edge_logits, edge_y)
        stop_loss = masked_cross_entropy(masked_stop_logits, stop_y)
        value_prediction = mx.sigmoid(value[:, 0])
        value_loss = ((value_prediction - reward_y) ** 2).mean()
        cost_prediction = nn.softplus(cost)
        cost_loss = ((cost_prediction - cost_y) ** 2).mean()
        return route_loss + edge_loss + stop_loss + 0.5 * value_loss + 0.25 * cost_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    initial_loss = 0.0
    final_loss = 0.0
    for epoch in range(epochs):
        loss, gradients = loss_and_grad(
            model,
            features,
            route_labels,
            edge_labels,
            stop_labels,
            edge_masks,
            stop_masks,
            rewards,
            costs,
        )
        optimizer.update(model, gradients)
        mx.eval(loss, model.parameters(), optimizer.state)
        value = float(loss.item())
        if epoch == 0:
            initial_loss = value
        final_loss = value

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "graph_policy.safetensors"
    config_path = output / "graph_policy.json"
    model.save_weights(str(weights_path))
    config.save(config_path)
    summary = TrainingSummary(
        records=len(records),
        epochs=epochs,
        initial_loss=initial_loss,
        final_loss=final_loss,
        weights_path=str(weights_path.resolve()),
        config_path=str(config_path.resolve()),
    )
    (output / "training_summary.json").write_text(
        json.dumps(summary.__dict__, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary
