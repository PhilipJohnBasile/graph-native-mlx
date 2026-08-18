import json
from pathlib import Path

from graph_model.router_policy import HashedLinearRouter, train_router_file


def test_router_policy_learns_constrained_paths(tmp_path: Path) -> None:
    examples = [
        ("quick typo rename in one file", "fast"),
        ("format a small patch", "fast"),
        ("one line variable rename", "fast"),
        ("design production architecture and migration", "deep"),
        ("implement a multi-file security feature", "deep"),
        ("benchmark a new database architecture", "deep"),
        ("fix failing CI error", "repair"),
        ("debug a broken regression", "repair"),
        ("repair the test failure", "repair"),
    ]
    model = HashedLinearRouter(dimension=256)
    model.fit(examples, epochs=90, learning_rate=0.16, seed=3)
    assert model.accuracy(examples) >= 0.95
    assert model.predict("quick typo rename").route == "fast"
    assert model.predict("production migration architecture").route == "deep"
    assert model.predict("CI regression is failing").route == "repair"


def test_train_router_jsonl_round_trip(tmp_path: Path) -> None:
    input_path = tmp_path / "traces.jsonl"
    output_path = tmp_path / "router.json"
    records = [
        {"task": "quick rename", "route": "fast", "success": True},
        {"task": "architecture migration", "route": "deep", "success": True},
        {"task": "failing CI regression", "route": "repair", "success": True},
    ]
    input_path.write_text("".join(json.dumps(item) + "\n" for item in records))
    summary = train_router_file(
        input_path=input_path,
        output_path=output_path,
        dimension=128,
        epochs=80,
    )
    loaded = HashedLinearRouter.load(output_path)
    assert summary.samples == 3
    assert output_path.exists()
    assert loaded.predict("failing CI regression").route == "repair"
