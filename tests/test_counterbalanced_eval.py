from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate-counterbalanced-policy-eval-v1.py"
COMPARATOR = ROOT / "scripts/compare-counterbalanced-policy-eval-v1.py"
RUNNER = ROOT / "scripts/run-counterbalanced-policy-eval-v1-mac.sh"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_counterbalanced_fixture_inventory_is_exact_and_unique() -> None:
    module = load_script(GENERATOR, "counterbalanced_generator")
    cases = module.fixtures()
    assert len(cases) == 24
    assert len({item["id"] for item in cases}) == 24
    assert Counter(item["category"] for item in cases) == {
        "fast": 6,
        "deep": 6,
        "repair": 6,
        "no-change": 3,
        "impossible": 3,
    }


def test_counterbalanced_fixture_contracts_are_fail_closed() -> None:
    module = load_script(GENERATOR, "counterbalanced_contracts")
    cases = module.fixtures()
    for item in cases:
        if item["expected_status"] == "completed":
            assert item["oracle"] is not None
            assert item["oracle"]["authoritative"] is True
            kinds = {check["kind"] for check in item["oracle"]["checks"]}
            assert {"tests_pass", "tests_unchanged", "files_end_newline"} <= kinds
        else:
            assert item["category"] == "impossible"
            assert item["oracle"] is None


def test_counterbalanced_order_is_twelve_and_twelve() -> None:
    module = load_script(GENERATOR, "counterbalanced_order")
    cases = module.fixtures()
    orders = ["static-first" if index % 2 == 1 else "full-first" for index, _ in enumerate(cases, 1)]
    assert Counter(orders) == {"static-first": 12, "full-first": 12}


def test_generator_writes_two_disjoint_manifests(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    static_manifest = tmp_path / "static.jsonl"
    full_manifest = tmp_path / "full.jsonl"
    cases_path = tmp_path / "cases.json"
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--root",
            str(corpus),
            "--static-first-manifest",
            str(static_manifest),
            "--full-first-manifest",
            str(full_manifest),
            "--cases",
            str(cases_path),
            "--python",
            sys.executable,
            "--skip-baseline",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    static_rows = [json.loads(line) for line in static_manifest.read_text().splitlines()]
    full_rows = [json.loads(line) for line in full_manifest.read_text().splitlines()]
    assert len(static_rows) == len(full_rows) == 12
    assert {row["run_id"] for row in static_rows}.isdisjoint(
        {row["run_id"] for row in full_rows}
    )
    bundle = json.loads(cases_path.read_text())
    assert bundle["order_counts"] == {"full-first": 12, "static-first": 12}
    assert all(case["baseline_validated"] is False for case in bundle["cases"])
    for case in bundle["cases"]:
        status = subprocess.check_output(
            [
                "git",
                "-C",
                case["repo"],
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            text=True,
        )
        assert status == ""


def test_percent_reduction_sign_convention() -> None:
    module = load_script(COMPARATOR, "counterbalanced_comparator")
    assert module.percent_reduction(100.0, 80.0) == 20.0
    assert module.percent_reduction(100.0, 110.0) == -10.0
    assert module.percent_reduction(0.0, 0.0) is None


def test_runner_executes_counterbalanced_block_order() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    blocks = [
        'run_block 1 static-first static "$STATIC_FIRST_MANIFEST"',
        'run_block 2 static-first full "$STATIC_FIRST_MANIFEST"',
        'run_block 3 full-first full "$FULL_FIRST_MANIFEST"',
        'run_block 4 full-first static "$FULL_FIRST_MANIFEST"',
    ]
    positions = [text.index(block) for block in blocks]
    assert positions == sorted(positions)
    assert "LATEST_COUNTERBALANCED_POLICY_EVAL" in text
    assert "GRAPH_MODEL_MLX_TEMPERATURE=0" in text
    assert "GRAPH_MODEL_MLX_SKIP_FORCED_POLICY=true" in text
