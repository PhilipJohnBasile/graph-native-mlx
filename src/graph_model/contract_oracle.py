from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

ORACLE_FORMAT = "graph-native-contract-oracle-v1"


def _safe_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve(strict=False)
    resolved_root = root.resolve(strict=True)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"oracle path escapes the worktree: {relative!r}") from exc
    return candidate


def _function_node(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _call_name(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _python_function_calls(root: Path, check: Mapping[str, Any]) -> dict[str, Any]:
    relative = str(check.get("path", ""))
    function_name = str(check.get("function", ""))
    callee = str(check.get("callee", ""))
    if not all((relative, function_name, callee)):
        return {
            "kind": "python_function_calls",
            "verdict": "inconclusive",
            "reason": "path, function, and callee are required",
        }
    path = _safe_file(root, relative)
    if not path.is_file():
        return {
            "kind": "python_function_calls",
            "verdict": "fail",
            "path": relative,
            "reason": "file is missing",
        }
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
    except (OSError, UnicodeError, SyntaxError) as exc:
        return {
            "kind": "python_function_calls",
            "verdict": "fail",
            "path": relative,
            "reason": f"could not parse Python source: {type(exc).__name__}: {exc}",
        }
    function = _function_node(tree, function_name)
    if function is None:
        return {
            "kind": "python_function_calls",
            "verdict": "fail",
            "path": relative,
            "function": function_name,
            "reason": "function is missing",
        }
    calls = sorted(
        {
            name
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            for name in [_call_name(node)]
            if name
        }
    )
    segment = ast.get_source_segment(source, function) or ""
    passed = callee in calls
    return {
        "kind": "python_function_calls",
        "verdict": "pass" if passed else "fail",
        "path": relative,
        "function": function_name,
        "callee": callee,
        "observed_calls": calls,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "function_excerpt": segment[:4_000],
        "reason": (
            f"{function_name} calls {callee}"
            if passed
            else f"{function_name} does not call {callee}"
        ),
    }


def _tests_unchanged(changed_files: Sequence[str]) -> dict[str, Any]:
    changed_tests = sorted(
        path
        for path in changed_files
        if path.startswith("tests/")
        or Path(path).name.startswith("test_")
        or "/test_" in path
    )
    return {
        "kind": "tests_unchanged",
        "verdict": "pass" if not changed_tests else "fail",
        "changed_test_files": changed_tests,
        "reason": (
            "no test files changed"
            if not changed_tests
            else "test files changed"
        ),
    }



def _allowed_changed_files(
    changed_files: Sequence[str],
    check: Mapping[str, Any],
) -> dict[str, Any]:
    raw_allowed = check.get("paths", [])
    if not isinstance(raw_allowed, list):
        return {
            "kind": "allowed_changed_files",
            "verdict": "inconclusive",
            "reason": "paths must be an array",
        }
    allowed = sorted({str(path) for path in raw_allowed})
    observed = sorted({str(path) for path in changed_files})
    unexpected = sorted(set(observed).difference(allowed))
    missing_required: list[str] = []
    raw_required = check.get("required", [])
    if isinstance(raw_required, list):
        missing_required = sorted(set(str(path) for path in raw_required).difference(observed))
    passed = not unexpected and not missing_required
    return {
        "kind": "allowed_changed_files",
        "verdict": "pass" if passed else "fail",
        "allowed": allowed,
        "observed": observed,
        "unexpected": unexpected,
        "missing_required": missing_required,
        "reason": (
            "changed files stay within the declared contract"
            if passed
            else "changed files violate the declared contract"
        ),
    }

def _files_end_newline(root: Path, changed_files: Sequence[str]) -> dict[str, Any]:
    failures: list[str] = []
    checked: list[str] = []
    for relative in changed_files:
        path = _safe_file(root, relative)
        if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".txt", ".json", ".yaml", ".yml"}:
            continue
        checked.append(relative)
        try:
            content = path.read_bytes()
        except OSError:
            failures.append(relative)
            continue
        if content and not content.endswith(b"\n"):
            failures.append(relative)
    return {
        "kind": "files_end_newline",
        "verdict": "pass" if not failures else "fail",
        "checked_files": checked,
        "missing_newline": failures,
        "reason": (
            "checked text files end with a newline"
            if not failures
            else "one or more changed text files lack a final newline"
        ),
    }


def _tests_pass(test_report: Any) -> dict[str, Any]:
    verdict = test_report.get("verdict") if isinstance(test_report, Mapping) else None
    return {
        "kind": "tests_pass",
        "verdict": "pass" if verdict == "pass" else "fail",
        "observed_verdict": verdict,
        "reason": "deterministic tests passed" if verdict == "pass" else "deterministic tests did not pass",
    }


def evaluate_contract_oracle(
    spec: Any,
    *,
    worktree: str | Path,
    changed_files: Sequence[str],
    test_report: Any,
) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        return {
            "format": ORACLE_FORMAT,
            "verdict": "inconclusive",
            "definitive": False,
            "checks": [],
            "reason": "no declarative contract oracle was supplied",
        }
    checks = spec.get("checks")
    if not isinstance(checks, list) or not checks:
        return {
            "format": ORACLE_FORMAT,
            "verdict": "inconclusive",
            "definitive": False,
            "checks": [],
            "reason": "contract oracle contains no checks",
        }
    root = Path(worktree).expanduser().resolve(strict=True)
    results: list[dict[str, Any]] = []
    for raw in checks:
        if not isinstance(raw, Mapping):
            results.append(
                {
                    "kind": "invalid",
                    "verdict": "inconclusive",
                    "reason": "oracle check must be an object",
                }
            )
            continue
        kind = str(raw.get("kind", ""))
        if kind == "python_function_calls":
            results.append(_python_function_calls(root, raw))
        elif kind == "tests_unchanged":
            results.append(_tests_unchanged(changed_files))
        elif kind == "files_end_newline":
            results.append(_files_end_newline(root, changed_files))
        elif kind == "allowed_changed_files":
            results.append(_allowed_changed_files(changed_files, raw))
        elif kind == "tests_pass":
            results.append(_tests_pass(test_report))
        else:
            results.append(
                {
                    "kind": kind or "unknown",
                    "verdict": "inconclusive",
                    "reason": "unsupported oracle check kind",
                }
            )
    verdicts = {str(item.get("verdict")) for item in results}
    if "fail" in verdicts:
        verdict = "fail"
    elif verdicts == {"pass"}:
        verdict = "pass"
    else:
        verdict = "inconclusive"
    return {
        "format": ORACLE_FORMAT,
        "name": str(spec.get("name") or "declarative-contract"),
        "verdict": verdict,
        "definitive": verdict in {"pass", "fail"},
        "authoritative": bool(spec.get("authoritative", False)),
        "checks": results,
        "reason": {
            "pass": "all declarative contract checks passed",
            "fail": "one or more declarative contract checks failed",
            "inconclusive": "the declarative contract could not be decided",
        }[verdict],
    }
