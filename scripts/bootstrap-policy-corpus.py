#!/usr/bin/env python3
"""Create a controlled local corpus for graph-policy trace collection.

This corpus qualifies the data pipeline and supplies route/status diversity. It is not a
substitute for varied real-repository traces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Fixture:
    name: str
    task: str
    files: Mapping[str, str]
    tags: tuple[str, ...]
    expected_route: str
    expected_terminal: str
    baseline: str


def _fixtures() -> tuple[Fixture, ...]:
    return (
        Fixture(
            "fast-normalize-key",
            "Quick fix: make normalize_key return a trimmed lowercase key. This should be a one line source-only patch; preserve the tests and verify.",
            {
                "keys.py": "def normalize_key(value: str) -> str:\n    return value.strip()\n",
                "tests/test_keys.py": "from keys import normalize_key\n\ndef test_normalize_key():\n    assert normalize_key('  Customer-ID  ') == 'customer-id'\n",
            },
            ("bootstrap", "fast", "one-line", "success"), "fast", "completed", "fail",
        ),
        Fixture(
            "fast-inclusive-boundary",
            "Quick fix: correct the one line age eligibility boundary so age 18 is accepted. Keep the public function and tests unchanged, then verify.",
            {
                "eligibility.py": "def is_adult(age: int) -> bool:\n    return age > 18\n",
                "tests/test_eligibility.py": "from eligibility import is_adult\n\ndef test_age_boundary():\n    assert is_adult(18) is True\n    assert is_adult(17) is False\n    assert is_adult(19) is True\n",
            },
            ("bootstrap", "fast", "boundary", "success"), "fast", "completed", "fail",
        ),
        Fixture(
            "fast-display-name",
            "Small patch: format display_name as 'first last' rather than a catalog-style name. Do not alter the tests; verify the result.",
            {
                "names.py": "def display_name(first: str, last: str) -> str:\n    return f\"{last}, {first}\"\n",
                "tests/test_names.py": "from names import display_name\n\ndef test_display_name():\n    assert display_name('Ada', 'Lovelace') == 'Ada Lovelace'\n",
            },
            ("bootstrap", "fast", "format", "success"), "fast", "completed", "fail",
        ),
        Fixture(
            "fast-slugify",
            "Quick fix: make slugify lowercase its input and collapse surrounding whitespace while retaining hyphen-separated words. Preserve tests and verify.",
            {
                "slug.py": "def slugify(value: str) -> str:\n    return value.strip().replace(' ', '-')\n",
                "tests/test_slug.py": "from slug import slugify\n\ndef test_slugify():\n    assert slugify('  Graph Native MLX  ') == 'graph-native-mlx'\n",
            },
            ("bootstrap", "fast", "text", "success"), "fast", "completed", "fail",
        ),
        Fixture(
            "repair-clamp-regression",
            "Repair the failing clamp regression. Keep the API and test contract unchanged, make the smallest correct patch, and verify exact boundary behavior.",
            {
                "clamping.py": "def clamp(value: int, low: int, high: int) -> int:\n    return min(low, max(high, value))\n",
                "tests/test_clamping.py": "from clamping import clamp\n\ndef test_clamp():\n    assert clamp(-2, 0, 10) == 0\n    assert clamp(4, 0, 10) == 4\n    assert clamp(12, 0, 10) == 10\n",
            },
            ("bootstrap", "repair", "boundary", "success"), "repair", "completed", "fail",
        ),
        Fixture(
            "repair-pair-parser",
            "Fix the broken pair parser. The failing tests define an equals-delimited contract; do not weaken them, apply a minimal repair, and verify.",
            {
                "parser.py": "def parse_pair(text: str) -> tuple[str, str]:\n    key, value = text.split(':', 1)\n    return key.strip(), value.strip()\n",
                "tests/test_parser.py": "from parser import parse_pair\n\ndef test_parse_pair():\n    assert parse_pair('region = east') == ('region', 'east')\n",
            },
            ("bootstrap", "repair", "parser", "success"), "repair", "completed", "fail",
        ),
        Fixture(
            "repair-retry-off-by-one",
            "Fix the retry bug that schedules one extra attempt. Preserve the public function, leave tests intact, and verify the repaired attempt sequence.",
            {
                "retry.py": "def attempt_indexes(max_attempts: int) -> list[int]:\n    return list(range(max_attempts + 1))\n",
                "tests/test_retry.py": "from retry import attempt_indexes\n\ndef test_attempt_indexes():\n    assert attempt_indexes(3) == [0, 1, 2]\n    assert attempt_indexes(0) == []\n",
            },
            ("bootstrap", "repair", "off-by-one", "success"), "repair", "completed", "fail",
        ),
        Fixture(
            "repair-mean-precision",
            "Repair the failing mean calculation regression so fractional results are retained. Use the smallest source change and verify without modifying tests.",
            {
                "stats.py": "def mean(values: list[float]) -> float:\n    return sum(values) // len(values)\n",
                "tests/test_stats.py": "from stats import mean\n\ndef test_mean_keeps_fraction():\n    assert mean([1, 2]) == 1.5\n    assert mean([2, 3, 4]) == 3.0\n",
            },
            ("bootstrap", "repair", "numeric", "success"), "repair", "completed", "fail",
        ),
        Fixture(
            "deep-email-refactor",
            "Refactor the multi-file email normalization feature so both account creation and newsletter signup use one consistent trimmed lowercase representation. Preserve APIs, keep the test contract unchanged, and verify the complete repository.",
            {
                "accounts.py": "def normalize_account_email(value: str) -> str:\n    return value.strip()\n",
                "newsletter.py": "def normalize_signup_email(value: str) -> str:\n    return value.lower()\n",
                "tests/test_email_normalization.py": "from accounts import normalize_account_email\nfrom newsletter import normalize_signup_email\n\ndef test_email_normalization_is_consistent():\n    value = '  USER@Example.COM  '\n    assert normalize_account_email(value) == 'user@example.com'\n    assert normalize_signup_email(value) == 'user@example.com'\n",
            },
            ("bootstrap", "deep", "multi-file", "refactor", "success"), "deep", "completed", "fail",
        ),
        Fixture(
            "deep-config-migration",
            "Implement the production configuration migration across files: missing port values must use 8080 while numeric strings remain supported. Preserve the existing public API and verify all acceptance tests.",
            {
                "legacy.py": "def parse_port(value: object) -> int:\n    return int(value)\n",
                "config.py": "from legacy import parse_port\n\ndef load_port(data: dict[str, object]) -> int:\n    return parse_port(data.get('port'))\n",
                "tests/test_config.py": "from config import load_port\n\ndef test_load_port():\n    assert load_port({}) == 8080\n    assert load_port({'port': '9000'}) == 9000\n",
            },
            ("bootstrap", "deep", "multi-file", "migration", "success"), "deep", "completed", "fail",
        ),
        Fixture(
            "deep-security-policy",
            "Implement the production security policy for document deletion. Administrators and the document owner may delete; other users may not. Preserve the function signature, do not alter tests, and verify all authorization cases.",
            {
                "policy.py": "def can_delete(user: dict[str, object], owner_id: int) -> bool:\n    return user.get('role') == 'admin'\n",
                "tests/test_policy.py": "from policy import can_delete\n\ndef test_delete_policy():\n    assert can_delete({'id': 1, 'role': 'admin'}, 9) is True\n    assert can_delete({'id': 9, 'role': 'member'}, 9) is True\n    assert can_delete({'id': 2, 'role': 'member'}, 9) is False\n",
            },
            ("bootstrap", "deep", "security", "production", "success"), "deep", "completed", "fail",
        ),
        Fixture(
            "deep-money-rounding",
            "Design and implement the production money-rounding feature across the invoice modules. Use half-up cents for line totals and invoice totals, preserve APIs, and verify the repository acceptance tests.",
            {
                "money/rounding.py": "def round_cents(amount: float) -> float:\n    return int(amount * 100) / 100\n",
                "money/invoice.py": "from .rounding import round_cents\n\ndef invoice_total(amounts: list[float]) -> float:\n    return round_cents(sum(amounts))\n",
                "money/__init__.py": "",
                "tests/test_money.py": "from money.invoice import invoice_total\nfrom money.rounding import round_cents\n\ndef test_half_up_rounding():\n    assert round_cents(10.005) == 10.01\n    assert invoice_total([5.0025, 5.0025]) == 10.01\n",
            },
            ("bootstrap", "deep", "multi-file", "production", "success"), "deep", "completed", "fail",
        ),
        Fixture(
            "fast-no-change",
            "Quick fix review: verify that square returns the mathematical square for integers. Make no changes when the implementation already satisfies the tests.",
            {
                "math_utils.py": "def square(value: int) -> int:\n    return value * value\n",
                "tests/test_math_utils.py": "from math_utils import square\n\ndef test_square():\n    assert square(4) == 16\n    assert square(-3) == 9\n",
            },
            ("bootstrap", "fast", "no-change", "success"), "fast", "completed", "pass",
        ),
        Fixture(
            "deep-no-change",
            "Review the production multi-file greeting design and verify that composition and punctuation satisfy the existing acceptance tests. Make no changes if the architecture is already correct.",
            {
                "greeting/name.py": "def display_name(first: str, last: str) -> str:\n    return f'{first} {last}'\n",
                "greeting/message.py": "from .name import display_name\n\ndef greeting(first: str, last: str) -> str:\n    return f'Hello, {display_name(first, last)}!'\n",
                "greeting/__init__.py": "",
                "tests/test_greeting.py": "from greeting.message import greeting\n\ndef test_greeting():\n    assert greeting('Grace', 'Hopper') == 'Hello, Grace Hopper!'\n",
            },
            ("bootstrap", "deep", "multi-file", "no-change", "success"), "deep", "completed", "pass",
        ),
        Fixture(
            "repair-impossible-contract",
            "Repair the failing toggle regression without modifying the immutable tests. Use bounded evidence-driven attempts and abort rather than claiming success if the contract cannot be satisfied.",
            {
                "toggle.py": "def enabled(flag: bool) -> bool:\n    return bool(flag)\n",
                "tests/test_toggle.py": "from toggle import enabled\n\ndef test_contradictory_contract():\n    assert enabled(True) is True\n    assert enabled(True) is False\n",
            },
            ("bootstrap", "repair", "abort", "failure"), "repair", "failed", "fail",
        ),
        Fixture(
            "deep-impossible-contract",
            "Design a production feature while treating the acceptance tests as immutable. Use the validated multi-step workflow and terminate without success if the two required outcomes cannot both hold.",
            {
                "feature.py": "def feature_value() -> int:\n    return 1\n",
                "tests/test_feature.py": "from feature import feature_value\n\ndef test_inconsistent_feature_contract():\n    assert feature_value() == 1\n    assert feature_value() == 2\n",
            },
            ("bootstrap", "deep", "abort", "failure"), "deep", "failed", "fail",
        ),
    )


def _run(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout


def _fixture_digest(fixture: Fixture) -> str:
    payload = {
        "name": fixture.name, "task": fixture.task, "files": dict(fixture.files),
        "tags": fixture.tags, "expected_route": fixture.expected_route,
        "expected_terminal": fixture.expected_terminal, "baseline": fixture.baseline,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _write_repo(repo: Path, fixture: Fixture, *, reset: bool) -> None:
    marker = repo / ".git" / "graph-bootstrap-fixture.json"
    digest = _fixture_digest(fixture)
    if repo.exists():
        if reset:
            shutil.rmtree(repo)
        elif marker.exists():
            metadata = json.loads(marker.read_text())
            if metadata.get("digest") != digest:
                raise RuntimeError(f"fixture {fixture.name!r} changed; rerun with --reset")
            if _run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repo).strip():
                raise RuntimeError(f"existing fixture repository is dirty: {repo}")
            return
        else:
            raise RuntimeError(f"path already exists and is not a fixture repo: {repo}")
    repo.mkdir(parents=True)
    for relative, content in fixture.files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n*.py[cod]\n.DS_Store\n")
    _run(["git", "init", "-q"], cwd=repo)
    _run(["git", "config", "user.name", "Graph Native MLX Bootstrap"], cwd=repo)
    _run(["git", "config", "user.email", "graph-native-bootstrap@example.invalid"], cwd=repo)
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-qm", f"bootstrap fixture: {fixture.name}"], cwd=repo)
    marker.write_text(json.dumps({
        "format_version": 1, "name": fixture.name, "digest": digest,
        "expected_route": fixture.expected_route, "expected_terminal": fixture.expected_terminal,
        "baseline": fixture.baseline,
    }, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path.home() / "graph-native-mlx-corpus" / "bootstrap-v1"))
    parser.add_argument("--manifest")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    repos_root = root / "repos"
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else root / "manifest.jsonl"
    repos_root.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    records, metadata = [], []
    test_command = f"{Path(args.python).expanduser().resolve()} -m pytest -q"
    for index, fixture in enumerate(_fixtures(), 1):
        repo = repos_root / fixture.name
        _write_repo(repo, fixture, reset=args.reset)
        run_id = f"bootstrap-v1-{index:02d}-{fixture.name}"
        records.append({"run_id": run_id, "repo": str(repo), "task": fixture.task, "test_commands": [test_command], "tags": list(fixture.tags)})
        metadata.append({"run_id": run_id, "repo": str(repo), "expected_route": fixture.expected_route, "expected_terminal": fixture.expected_terminal, "baseline": fixture.baseline, "tags": list(fixture.tags)})
    manifest.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    metadata_path = root / "corpus-metadata.json"
    metadata_path.write_text(json.dumps({"format_version": 1, "fixtures": metadata, "manifest": str(manifest)}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "root": str(root), "manifest": str(manifest), "metadata": str(metadata_path), "tasks": len(records),
        "expected_routes": {route: sum(item["expected_route"] == route for item in metadata) for route in ("fast", "deep", "repair")},
        "expected_terminals": {status: sum(item["expected_terminal"] == status for item in metadata) for status in ("completed", "failed")},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
