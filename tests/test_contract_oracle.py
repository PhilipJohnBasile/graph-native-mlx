from __future__ import annotations

from pathlib import Path

from graph_model.contract_oracle import evaluate_contract_oracle


def _paging_tree(root: Path, *, shared_query: bool = True, newline: bool = True) -> None:
    (root / "paging").mkdir()
    query_body = (
        "from .cursor import decode_cursor\n\n"
        "def query_offset(params: dict[str, str]) -> int:\n"
        + (
            "    return decode_cursor(params.get('cursor'))\n"
            if shared_query
            else "    return int(params.get('cursor') or 0)\n"
        )
    )
    service_body = (
        "from .cursor import decode_cursor\n\n"
        "def page_start(cursor: str | None) -> int:\n"
        "    return decode_cursor(cursor)\n"
    )
    if not newline:
        service_body = service_body.rstrip("\n")
    (root / "paging/query.py").write_text(query_body, encoding="utf-8")
    (root / "paging/service.py").write_text(service_body, encoding="utf-8")


def _spec() -> dict:
    return {
        "name": "pagination-shared-decoder",
        "authoritative": True,
        "checks": [
            {
                "kind": "allowed_changed_files",
                "paths": ["paging/service.py"],
                "required": ["paging/service.py"],
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
            {"kind": "tests_pass"},
            {"kind": "tests_unchanged"},
            {"kind": "files_end_newline"},
        ],
    }


def test_contract_oracle_proves_shared_decoder_in_unchanged_and_changed_files(
    tmp_path: Path,
) -> None:
    _paging_tree(tmp_path)
    report = evaluate_contract_oracle(
        _spec(),
        worktree=tmp_path,
        changed_files=["paging/service.py"],
        test_report={"verdict": "pass"},
    )
    assert report["verdict"] == "pass"
    assert report["definitive"] is True
    assert report["authoritative"] is True
    query = next(
        item for item in report["checks"]
        if item["kind"] == "python_function_calls"
        and item.get("function") == "query_offset"
    )
    assert query["verdict"] == "pass"
    assert "decode_cursor" in query["observed_calls"]
    assert "query_offset" in query["function_excerpt"]


def test_contract_oracle_rejects_missing_shared_decoder(tmp_path: Path) -> None:
    _paging_tree(tmp_path, shared_query=False)
    report = evaluate_contract_oracle(
        _spec(),
        worktree=tmp_path,
        changed_files=["paging/service.py"],
        test_report={"verdict": "pass"},
    )
    assert report["verdict"] == "fail"
    calls = [
        item for item in report["checks"]
        if item["kind"] == "python_function_calls"
    ]
    assert calls[0]["verdict"] == "fail"


def test_contract_oracle_rejects_test_changes_and_missing_newline(tmp_path: Path) -> None:
    _paging_tree(tmp_path, newline=False)
    report = evaluate_contract_oracle(
        _spec(),
        worktree=tmp_path,
        changed_files=["paging/service.py", "tests/test_paging.py"],
        test_report={"verdict": "pass"},
    )
    assert report["verdict"] == "fail"
    kinds = {item["kind"]: item for item in report["checks"]}
    assert kinds["tests_unchanged"]["verdict"] == "fail"
    assert kinds["files_end_newline"]["verdict"] == "fail"


def test_contract_oracle_rejects_unexpected_changed_file(tmp_path: Path) -> None:
    _paging_tree(tmp_path)
    report = evaluate_contract_oracle(
        _spec(),
        worktree=tmp_path,
        changed_files=["paging/service.py", "paging/query.py"],
        test_report={"verdict": "pass"},
    )
    assert report["verdict"] == "fail"
    kinds = {item["kind"]: item for item in report["checks"]}
    assert kinds["allowed_changed_files"]["unexpected"] == ["paging/query.py"]
