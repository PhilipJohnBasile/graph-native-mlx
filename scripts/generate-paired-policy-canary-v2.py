#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(*argv: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def fixtures() -> list[dict[str, Any]]:
    common_oracle = [
        {"kind": "tests_pass"},
        {"kind": "tests_unchanged"},
        {"kind": "files_end_newline"},
    ]
    return [
        {
            "id": "fast-header-trim",
            "baseline": "fail",
            "expected_status": "completed",
            "task": (
                "Quick fix: make normalize_header remove only surrounding whitespace "
                "while preserving case and internal characters. Keep the public "
                "function and tests unchanged, make the smallest source-only patch, "
                "and verify."
            ),
            "oracle": {
                "name": "header-contract",
                "authoritative": True,
                "checks": [
                    {
                        "kind": "allowed_changed_files",
                        "paths": ["headers.py"],
                        "required": ["headers.py"],
                    },
                    *common_oracle,
                ],
            },
            "files": {
                ".gitignore": "__pycache__/\n.pytest_cache/\n*.pyc\n",
                "headers.py": (
                    'def normalize_header(name: str) -> str:\n'
                    '    """Return a canonical HTTP header spelling."""\n'
                    "    return name\n"
                ),
                "tests/test_headers.py": (
                    "from headers import normalize_header\n\n\n"
                    "def test_surrounding_whitespace_is_removed():\n"
                    '    assert normalize_header("  Content-Type  ") == "Content-Type"\n\n\n'
                    "def test_case_and_internal_characters_are_preserved():\n"
                    '    assert normalize_header("X-Trace-ID") == "X-Trace-ID"\n'
                ),
            },
        },
        {
            "id": "deep-pagination-cursor",
            "baseline": "fail",
            "expected_status": "completed",
            "task": (
                "Implement a multi-file pagination cursor migration. Missing or blank "
                "cursors map to offset 0, decimal strings remain supported, negative "
                "offsets are rejected, and query_offset plus page_start must share one "
                "decoder. Preserve public APIs and tests, then verify the repository."
            ),
            "oracle": {
                "name": "pagination-shared-decoder",
                "authoritative": True,
                "checks": [
                    {
                        "kind": "allowed_changed_files",
                        "paths": ["paging/cursor.py", "paging/service.py"],
                        "required": ["paging/cursor.py", "paging/service.py"],
                    },
                    {
                        "kind": "python_function_calls",
                        "path": "paging/query.py",
                        "function": "query_offset",
                        "callee": "decode_cursor",
                    },
                    {
                        "kind": "python_function_calls",
                        "path": "paging/service.py",
                        "function": "page_start",
                        "callee": "decode_cursor",
                    },
                    *common_oracle,
                ],
            },
            "files": {
                ".gitignore": "__pycache__/\n.pytest_cache/\n*.pyc\n",
                "paging/__init__.py": "",
                "paging/cursor.py": (
                    "def decode_cursor(value: str | None) -> int:\n"
                    "    if value is None:\n"
                    "        return 1\n"
                    "    return int(value)\n"
                ),
                "paging/query.py": (
                    "from .cursor import decode_cursor\n\n\n"
                    "def query_offset(params: dict[str, str]) -> int:\n"
                    '    return decode_cursor(params.get("cursor"))\n'
                ),
                "paging/service.py": (
                    "def page_start(cursor: str | None) -> int:\n"
                    "    return int(cursor or 1)\n"
                ),
                "tests/test_paging.py": (
                    "import pytest\n\n"
                    "from paging.cursor import decode_cursor\n"
                    "from paging.query import query_offset\n"
                    "from paging.service import page_start\n\n\n"
                    "def test_missing_and_blank_cursor_start_at_zero():\n"
                    "    assert decode_cursor(None) == 0\n"
                    '    assert decode_cursor("") == 0\n'
                    "    assert query_offset({}) == 0\n"
                    "    assert page_start(None) == 0\n\n\n"
                    "def test_decimal_strings_remain_supported_everywhere():\n"
                    '    assert decode_cursor("12") == 12\n'
                    '    assert query_offset({"cursor": "7"}) == 7\n'
                    '    assert page_start("9") == 9\n\n\n'
                    "def test_negative_offsets_are_rejected():\n"
                    "    with pytest.raises(ValueError):\n"
                    '        decode_cursor("-1")\n'
                    "    with pytest.raises(ValueError):\n"
                    '        page_start("-2")\n'
                ),
            },
        },
        {
            "id": "fast-no-change-backoff",
            "baseline": "pass",
            "expected_status": "completed",
            "task": (
                "Audit retry_delay for exponential backoff from attempt zero. "
                "If correct, make no changes and verify."
            ),
            "oracle": {
                "name": "backoff-no-change",
                "authoritative": True,
                "checks": [
                    {"kind": "allowed_changed_files", "paths": []},
                    *common_oracle,
                ],
            },
            "files": {
                ".gitignore": "__pycache__/\n.pytest_cache/\n*.pyc\n",
                "backoff.py": (
                    "def retry_delay(attempt: int, base: float = 0.5) -> float:\n"
                    "    if attempt < 0:\n"
                    '        raise ValueError("attempt must be non-negative")\n'
                    "    return base * (2 ** attempt)\n"
                ),
                "tests/test_backoff.py": (
                    "import pytest\n\n"
                    "from backoff import retry_delay\n\n\n"
                    "def test_backoff_begins_at_attempt_zero():\n"
                    "    assert retry_delay(0) == 0.5\n"
                    "    assert retry_delay(1) == 1.0\n"
                    "    assert retry_delay(3) == 4.0\n\n\n"
                    "def test_custom_base_and_negative_attempts():\n"
                    "    assert retry_delay(2, base=0.25) == 1.0\n"
                    "    with pytest.raises(ValueError):\n"
                    "        retry_delay(-1)\n"
                ),
            },
        },
        {
            "id": "repair-impossible-status",
            "baseline": "fail",
            "expected_status": "failed",
            "task": (
                "Repair the broken status_label contract while preserving its public "
                "API and every test. Do not edit tests. If the requirements contradict "
                "each other, use bounded verification and terminate safely instead of "
                "claiming success."
            ),
            "oracle": None,
            "files": {
                ".gitignore": "__pycache__/\n.pytest_cache/\n*.pyc\n",
                "status.py": (
                    "def status_label(code: int) -> str:\n"
                    '    return "ready" if code == 200 else "unavailable"\n'
                ),
                "tests/test_status.py": (
                    "from status import status_label\n\n\n"
                    "def test_success_code_is_ready():\n"
                    '    assert status_label(200) == "ready"\n\n\n'
                    "def test_same_success_code_is_complete():\n"
                    '    assert status_label(200) == "complete"\n'
                ),
            },
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=42057)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    cases_path = Path(args.cases).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)

    case_records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    for index, fixture in enumerate(fixtures(), 1):
        case_id = str(fixture["id"])
        repo = root / case_id
        repo.mkdir(parents=True)
        for relative, content in dict(fixture["files"]).items():
            write(repo / relative, str(content))
        run("git", "init", "-q", cwd=repo)
        run("git", "config", "user.email", "paired@example.invalid", cwd=repo)
        run("git", "config", "user.name", "Graph Paired Canary", cwd=repo)
        run("git", "add", ".", cwd=repo)
        run("git", "commit", "-qm", "paired held-out fixture", cwd=repo)
        base_commit = run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
        baseline = run(args.python, "-m", "pytest", "-q", cwd=repo, check=False)
        baseline_status = "pass" if baseline.returncode == 0 else "fail"
        if baseline_status != fixture["baseline"]:
            print(baseline.stdout)
            print(baseline.stderr, file=sys.stderr)
            raise SystemExit(
                f"{case_id}: expected baseline {fixture['baseline']}, got {baseline_status}"
            )
        status = run(
            "git", "status", "--porcelain=v1", "--untracked-files=all", cwd=repo
        ).stdout
        if status.strip():
            raise SystemExit(f"{case_id}: repository is dirty after baseline:\n{status}")
        run_id = f"paired-v2-{index:02d}-{case_id}"
        record = {
            "run_id": run_id,
            "case_id": case_id,
            "repo": str(repo),
            "repository_alias": f"<repository:{case_id}>",
            "paired_evaluation": True,
            "evaluation_seed": args.seed + index,
            "task": fixture["task"],
            "test_commands": [f"{args.python} -m pytest -q"],
            "tags": ["paired-canary-v2", case_id, fixture["expected_status"]],
        }
        if fixture["oracle"] is not None:
            record["contract_oracle"] = fixture["oracle"]
        manifest_records.append(record)
        case_records.append(
            {
                "index": index,
                "id": case_id,
                "repo": str(repo),
                "base_commit": base_commit,
                "baseline": fixture["baseline"],
                "expected_status": fixture["expected_status"],
                "run_id": run_id,
                "task": fixture["task"],
            }
        )

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in manifest_records),
        encoding="utf-8",
    )
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "purpose": "causal paired static/shadow/route-only/full policy canary",
                "cases": case_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(case_records)} paired repositories")
    print(f"Manifest: {manifest}")
    print(f"Cases: {cases_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
