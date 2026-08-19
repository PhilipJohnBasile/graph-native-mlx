from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

ROUTES: tuple[str, ...] = ("fast", "deep", "repair")
_TOKEN_RE = re.compile(r"[a-z0-9_./:+-]+")


@dataclass(frozen=True)
class RouterPrediction:
    route: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class TrainingSummary:
    samples: int
    epochs: int
    dimension: int
    training_accuracy: float
    output: str


def _bucket(name: str, dimension: int) -> int:
    if dimension < 8:
        raise ValueError("router dimension must be >= 8")
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return 1 + int.from_bytes(digest, "big") % (dimension - 1)


def extract_features(task: str, *, dimension: int) -> dict[int, float]:
    """Create stable sparse features without a tokenizer dependency."""
    text = task.lower().strip()
    tokens = _TOKEN_RE.findall(text)
    features: dict[int, float] = {0: 1.0}

    def add(name: str, value: float = 1.0) -> None:
        index = _bucket(name, dimension)
        features[index] = features.get(index, 0.0) + value

    normalizer = 1.0 / math.sqrt(max(1, len(tokens)))
    for token in tokens:
        add(f"tok:{token}", normalizer)
    for left, right in zip(tokens, tokens[1:]):
        add(f"bi:{left}:{right}", normalizer * 0.7)

    length = len(text)
    add(f"length:{min(length // 80, 8)}")
    add(f"tokens:{min(len(tokens) // 12, 8)}")

    marker_groups = {
        "failure": ("fail", "error", "broken", "regression", "stack trace", "ci"),
        "small": ("typo", "rename", "one line", "small patch", "format"),
        "large": ("architecture", "migration", "production", "multi-file", "benchmark"),
        "code": ("```", ".py", ".ts", ".rs", ".go", "github", "pull request"),
        "external": ("http://", "https://", "api", "database", "deploy"),
    }
    for group, markers in marker_groups.items():
        if any(marker in text for marker in markers):
            add(f"marker:{group}")

    if "?" in task:
        add("shape:question")
    if "\n" in task:
        add("shape:multiline")
    if any(char.isdigit() for char in task):
        add("shape:numeric")
    return features


class HashedLinearRouter:
    """Small constrained softmax policy for choosing among validated graph routes."""

    VERSION = 1

    def __init__(
        self,
        *,
        dimension: int = 1024,
        routes: tuple[str, ...] = ROUTES,
        weights: dict[str, list[float]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if dimension < 8:
            raise ValueError("dimension must be >= 8")
        if not routes:
            raise ValueError("at least one route is required")
        self.dimension = dimension
        self.routes = tuple(routes)
        self.weights = weights or {route: [0.0] * dimension for route in self.routes}
        self.metadata = metadata or {}
        for route in self.routes:
            row = self.weights.get(route)
            if row is None or len(row) != dimension:
                raise ValueError(f"invalid weight row for route {route!r}")

    def predict(self, task: str) -> RouterPrediction:
        features = extract_features(task, dimension=self.dimension)
        logits = {
            route: sum(self.weights[route][index] * value for index, value in features.items())
            for route in self.routes
        }
        maximum = max(logits.values())
        exponentials = {route: math.exp(score - maximum) for route, score in logits.items()}
        denominator = sum(exponentials.values()) or 1.0
        probabilities = {
            route: exponentials[route] / denominator for route in self.routes
        }
        selected = max(self.routes, key=lambda route: probabilities[route])
        return RouterPrediction(
            route=selected,
            confidence=probabilities[selected],
            probabilities=probabilities,
        )

    def fit(
        self,
        records: Iterable[tuple[str, str]],
        *,
        epochs: int = 40,
        learning_rate: float = 0.12,
        l2: float = 1e-5,
        seed: int = 42,
    ) -> int:
        examples = [(task, route) for task, route in records if route in self.routes]
        if not examples:
            raise ValueError("no valid router training examples")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        rng = random.Random(seed)
        for epoch in range(epochs):
            rng.shuffle(examples)
            rate = learning_rate / math.sqrt(1.0 + epoch * 0.08)
            for task, expected in examples:
                features = extract_features(task, dimension=self.dimension)
                prediction = self.predict(task)
                for route in self.routes:
                    error = prediction.probabilities[route] - float(route == expected)
                    row = self.weights[route]
                    for index, value in features.items():
                        gradient = error * value + l2 * row[index]
                        row[index] -= rate * gradient
        self.metadata.update(
            {
                "samples": len(examples),
                "epochs": epochs,
                "learning_rate": learning_rate,
                "l2": l2,
                "seed": seed,
            }
        )
        return len(examples)

    def accuracy(self, records: Iterable[tuple[str, str]]) -> float:
        examples = [(task, route) for task, route in records if route in self.routes]
        if not examples:
            return 0.0
        correct = sum(self.predict(task).route == route for task, route in examples)
        return correct / len(examples)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "dimension": self.dimension,
            "routes": list(self.routes),
            "weights": self.weights,
            "metadata": self.metadata,
        }
        output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HashedLinearRouter":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("version", -1)) != cls.VERSION:
            raise ValueError("unsupported router model version")
        return cls(
            dimension=int(payload["dimension"]),
            routes=tuple(payload["routes"]),
            weights={key: [float(value) for value in row] for key, row in payload["weights"].items()},
            metadata=dict(payload.get("metadata") or {}),
        )


def _read_records(path: str | Path, *, success_only: bool) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if success_only and not bool(record.get("success")):
                continue
            task = record.get("task")
            route = record.get("route")
            if isinstance(task, str) and route in ROUTES:
                examples.append((task, str(route)))
    return examples


def train_router_file(
    *,
    input_path: str | Path,
    output_path: str | Path,
    dimension: int = 1024,
    epochs: int = 40,
    learning_rate: float = 0.12,
    success_only: bool = False,
    seed: int = 42,
) -> TrainingSummary:
    examples = _read_records(input_path, success_only=success_only)
    model = HashedLinearRouter(dimension=dimension)
    sample_count = model.fit(
        examples,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
    )
    training_accuracy = model.accuracy(examples)
    model.metadata["training_accuracy"] = training_accuracy
    model.save(output_path)
    return TrainingSummary(
        samples=sample_count,
        epochs=epochs,
        dimension=dimension,
        training_accuracy=training_accuracy,
        output=str(Path(output_path).resolve()),
    )


@lru_cache(maxsize=8)
def _load_cached(path: str, modified_ns: int) -> HashedLinearRouter:
    del modified_ns
    return HashedLinearRouter.load(path)


def load_router_cached(path: str | Path) -> HashedLinearRouter:
    resolved = Path(path).expanduser().resolve()
    return _load_cached(str(resolved), resolved.stat().st_mtime_ns)
