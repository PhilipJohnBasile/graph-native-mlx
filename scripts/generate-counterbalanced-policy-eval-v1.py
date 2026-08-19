#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

CATEGORY_COUNTS = {
    "fast": 6,
    "deep": 6,
    "repair": 6,
    "no-change": 3,
    "impossible": 3,
}
TOTAL_CASES = sum(CATEGORY_COUNTS.values())


def run(
    *argv: str,
    cwd: Path,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
        env=env,
    )


def source(value: str) -> str:
    return textwrap.dedent(value).lstrip("\n")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def completed_oracle(
    *,
    name: str,
    allowed: list[str],
    required: list[str] | None = None,
    calls: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = [
        {
            "kind": "allowed_changed_files",
            "paths": allowed,
            "required": required or [],
        }
    ]
    for path, function, callee in calls or []:
        checks.append(
            {
                "kind": "python_function_calls",
                "path": path,
                "function": function,
                "callee": callee,
            }
        )
    checks.extend(
        [
            {"kind": "tests_pass"},
            {"kind": "tests_unchanged"},
            {"kind": "files_end_newline"},
        ]
    )
    return {
        "name": name,
        "authoritative": True,
        "checks": checks,
    }


def fixture(
    *,
    case_id: str,
    category: str,
    task: str,
    files: dict[str, str],
    baseline: str = "fail",
    expected_status: str = "completed",
    oracle: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "category": category,
        "task": task,
        "files": {".gitignore": "__pycache__/\n.pytest_cache/\n*.pyc\n", **files},
        "baseline": baseline,
        "expected_status": expected_status,
        "oracle": oracle,
    }


def fixtures() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # Fast, single-file repairs.
    cases.extend(
        [
            fixture(
                case_id="fast-env-flag",
                category="fast",
                task=(
                    "Quick fix: make parse_env_flag accept surrounding whitespace and "
                    "case-insensitive true/false spellings while preserving its default "
                    "behavior for None. Unknown spellings must raise ValueError. Keep the "
                    "public API and tests unchanged, make the smallest source-only patch, "
                    "and verify."
                ),
                oracle=completed_oracle(
                    name="env-flag-contract",
                    allowed=["env_flags.py"],
                    required=["env_flags.py"],
                ),
                files={
                    "env_flags.py": source(
                        '''
                        TRUTHY = {"1", "true", "yes", "on"}
                        FALSY = {"0", "false", "no", "off"}


                        def parse_env_flag(value: str | None, default: bool = False) -> bool:
                            if value is None:
                                return default
                            if value in TRUTHY:
                                return True
                            if value in FALSY:
                                return False
                            raise ValueError(f"invalid flag: {value}")
                        '''
                    ),
                    "tests/test_env_flags.py": source(
                        '''
                        import pytest

                        from env_flags import parse_env_flag


                        def test_none_uses_default():
                            assert parse_env_flag(None) is False
                            assert parse_env_flag(None, default=True) is True


                        def test_spellings_are_trimmed_and_case_insensitive():
                            assert parse_env_flag(" YES ") is True
                            assert parse_env_flag("off") is False
                            assert parse_env_flag("TrUe") is True


                        def test_unknown_spelling_is_rejected():
                            with pytest.raises(ValueError):
                                parse_env_flag("maybe")
                        '''
                    ),
                },
            ),
            fixture(
                case_id="fast-port-range",
                category="fast",
                task=(
                    "Quick fix: make parse_port return an integer only for ports in the "
                    "inclusive range 1 through 65535. Whitespace around decimal input remains "
                    "valid. Preserve the public API and tests, make the smallest source-only "
                    "patch, and verify."
                ),
                oracle=completed_oracle(
                    name="port-range-contract",
                    allowed=["ports.py"],
                    required=["ports.py"],
                ),
                files={
                    "ports.py": source(
                        '''
                        def parse_port(value: str) -> int:
                            return int(value)
                        '''
                    ),
                    "tests/test_ports.py": source(
                        '''
                        import pytest

                        from ports import parse_port


                        def test_valid_port_boundaries_and_whitespace():
                            assert parse_port(" 8080 ") == 8080
                            assert parse_port("1") == 1
                            assert parse_port("65535") == 65535


                        @pytest.mark.parametrize("value", ["0", "65536", "-1"])
                        def test_out_of_range_ports_are_rejected(value):
                            with pytest.raises(ValueError):
                                parse_port(value)
                        '''
                    ),
                },
            ),
            fixture(
                case_id="fast-stable-unique",
                category="fast",
                task=(
                    "Quick fix: make unique_ordered remove duplicate strings while preserving "
                    "the first-seen order. Keep its public API and tests unchanged, make the "
                    "smallest source-only patch, and verify."
                ),
                oracle=completed_oracle(
                    name="stable-unique-contract",
                    allowed=["collections_util.py"],
                    required=["collections_util.py"],
                ),
                files={
                    "collections_util.py": source(
                        '''
                        def unique_ordered(items: list[str]) -> list[str]:
                            return sorted(set(items))
                        '''
                    ),
                    "tests/test_collections_util.py": source(
                        '''
                        from collections_util import unique_ordered


                        def test_duplicates_are_removed_in_first_seen_order():
                            assert unique_ordered(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


                        def test_empty_and_already_unique_inputs():
                            assert unique_ordered([]) == []
                            assert unique_ordered(["x", "y"]) == ["x", "y"]
                        '''
                    ),
                },
            ),
            fixture(
                case_id="fast-timeout-normalize",
                category="fast",
                task=(
                    "Quick fix: make normalize_timeout use 30.0 only for None, preserve an "
                    "explicit zero timeout, convert positive values to float, and reject "
                    "negative values. Preserve the public API and tests, then verify."
                ),
                oracle=completed_oracle(
                    name="timeout-contract",
                    allowed=["timeouts.py"],
                    required=["timeouts.py"],
                ),
                files={
                    "timeouts.py": source(
                        '''
                        def normalize_timeout(value: int | float | None) -> float:
                            if not value:
                                return 30.0
                            return float(value)
                        '''
                    ),
                    "tests/test_timeouts.py": source(
                        '''
                        import pytest

                        from timeouts import normalize_timeout


                        def test_none_uses_default_and_zero_is_preserved():
                            assert normalize_timeout(None) == 30.0
                            assert normalize_timeout(0) == 0.0


                        def test_positive_values_are_floats():
                            assert normalize_timeout(5) == 5.0
                            assert normalize_timeout(2.5) == 2.5


                        def test_negative_timeout_is_rejected():
                            with pytest.raises(ValueError):
                                normalize_timeout(-0.1)
                        '''
                    ),
                },
            ),
            fixture(
                case_id="fast-mask-token",
                category="fast",
                task=(
                    "Quick fix: make mask_token replace every character except the final four "
                    "with asterisks. Tokens of four characters or fewer must be fully masked. "
                    "Keep the public API and tests unchanged, make the smallest source-only "
                    "patch, and verify."
                ),
                oracle=completed_oracle(
                    name="token-mask-contract",
                    allowed=["secrets.py"],
                    required=["secrets.py"],
                ),
                files={
                    "secrets.py": source(
                        '''
                        def mask_token(token: str) -> str:
                            if len(token) <= 4:
                                return token
                            return token[:4] + "*" * (len(token) - 4)
                        '''
                    ),
                    "tests/test_secrets.py": source(
                        '''
                        from secrets import mask_token


                        def test_long_token_preserves_only_final_four():
                            assert mask_token("abcdefghij") == "******ghij"


                        def test_short_tokens_are_fully_masked():
                            assert mask_token("abcd") == "****"
                            assert mask_token("x") == "*"
                            assert mask_token("") == ""
                        '''
                    ),
                },
            ),
            fixture(
                case_id="fast-extension",
                category="fast",
                task=(
                    "Quick fix: make file_extension return the lowercase final extension "
                    "without a leading dot. Names without an extension, including dotfiles, "
                    "must return an empty string. Preserve the API and tests, then verify."
                ),
                oracle=completed_oracle(
                    name="extension-contract",
                    allowed=["files.py"],
                    required=["files.py"],
                ),
                files={
                    "files.py": source(
                        '''
                        def file_extension(filename: str) -> str:
                            return filename.split(".")[-1]
                        '''
                    ),
                    "tests/test_files.py": source(
                        '''
                        from files import file_extension


                        def test_final_extension_is_lowercase():
                            assert file_extension("archive.TAR.GZ") == "gz"
                            assert file_extension("photo.JPEG") == "jpeg"


                        def test_names_without_real_extension_are_empty():
                            assert file_extension("README") == ""
                            assert file_extension(".env") == ""
                        '''
                    ),
                },
            ),
        ]
    )

    # Deep, multi-file migrations with a shared implementation contract.
    cases.extend(
        [
            fixture(
                case_id="deep-request-id",
                category="deep",
                task=(
                    "Implement a shared request-ID normalization migration. normalize_request_id "
                    "must trim whitespace, uppercase the identifier, and reject blank input. "
                    "Both logging_request_id and tracing_request_id must delegate to that one "
                    "normalizer. Preserve public APIs and tests, then verify the repository."
                ),
                oracle=completed_oracle(
                    name="request-id-shared-normalizer",
                    allowed=["request_ids.py", "logging_ctx.py", "tracing.py"],
                    required=["request_ids.py", "logging_ctx.py", "tracing.py"],
                    calls=[
                        ("logging_ctx.py", "logging_request_id", "normalize_request_id"),
                        ("tracing.py", "tracing_request_id", "normalize_request_id"),
                    ],
                ),
                files={
                    "request_ids.py": source(
                        '''
                        def normalize_request_id(value: str) -> str:
                            return value
                        '''
                    ),
                    "logging_ctx.py": source(
                        '''
                        def logging_request_id(value: str) -> str:
                            return value.strip()
                        '''
                    ),
                    "tracing.py": source(
                        '''
                        def tracing_request_id(value: str) -> str:
                            return value
                        '''
                    ),
                    "tests/test_request_ids.py": source(
                        '''
                        import pytest

                        from logging_ctx import logging_request_id
                        from request_ids import normalize_request_id
                        from tracing import tracing_request_id


                        def test_all_entry_points_share_normalization():
                            for function in (normalize_request_id, logging_request_id, tracing_request_id):
                                assert function("  ab-c9  ") == "AB-C9"


                        def test_blank_ids_are_rejected_everywhere():
                            for function in (normalize_request_id, logging_request_id, tracing_request_id):
                                with pytest.raises(ValueError):
                                    function("   ")
                        '''
                    ),
                },
            ),
            fixture(
                case_id="deep-sort-direction",
                category="deep",
                task=(
                    "Implement a shared sort-direction parser. Missing or blank directions map "
                    "to 'asc'; accepted values are case-insensitive 'asc' and 'desc'; every "
                    "other value raises ValueError. query_direction and export_direction must "
                    "delegate to parse_sort_direction. Preserve APIs and tests, then verify."
                ),
                oracle=completed_oracle(
                    name="sort-direction-shared-parser",
                    allowed=["sorting/direction.py", "sorting/query.py", "sorting/export.py"],
                    required=["sorting/direction.py", "sorting/query.py", "sorting/export.py"],
                    calls=[
                        ("sorting/query.py", "query_direction", "parse_sort_direction"),
                        ("sorting/export.py", "export_direction", "parse_sort_direction"),
                    ],
                ),
                files={
                    "sorting/__init__.py": "",
                    "sorting/direction.py": source(
                        '''
                        def parse_sort_direction(value: str | None) -> str:
                            return value or "desc"
                        '''
                    ),
                    "sorting/query.py": source(
                        '''
                        def query_direction(value: str | None) -> str:
                            return (value or "asc").lower()
                        '''
                    ),
                    "sorting/export.py": source(
                        '''
                        def export_direction(value: str | None) -> str:
                            return value or "asc"
                        '''
                    ),
                    "tests/test_sorting.py": source(
                        '''
                        import pytest

                        from sorting.direction import parse_sort_direction
                        from sorting.export import export_direction
                        from sorting.query import query_direction


                        def test_defaults_and_case_normalization_are_shared():
                            for function in (parse_sort_direction, query_direction, export_direction):
                                assert function(None) == "asc"
                                assert function("  ") == "asc"
                                assert function("DESC") == "desc"
                                assert function(" asc ") == "asc"


                        def test_unknown_direction_is_rejected_everywhere():
                            for function in (parse_sort_direction, query_direction, export_direction):
                                with pytest.raises(ValueError):
                                    function("sideways")
                        '''
                    ),
                },
            ),
            fixture(
                case_id="deep-token-expiry",
                category="deep",
                task=(
                    "Centralize token-expiry behavior. is_expired must return false for no "
                    "expiry and true when expires_at is less than or equal to now. "
                    "session_expired and cache_entry_expired must both delegate to is_expired. "
                    "Preserve public APIs and tests, then verify."
                ),
                oracle=completed_oracle(
                    name="expiry-shared-policy",
                    allowed=["expiry.py", "sessions.py", "cache_entries.py"],
                    required=["expiry.py", "sessions.py", "cache_entries.py"],
                    calls=[
                        ("sessions.py", "session_expired", "is_expired"),
                        ("cache_entries.py", "cache_entry_expired", "is_expired"),
                    ],
                ),
                files={
                    "expiry.py": source(
                        '''
                        def is_expired(expires_at: float | None, now: float) -> bool:
                            return expires_at is not None and expires_at < now
                        '''
                    ),
                    "sessions.py": source(
                        '''
                        def session_expired(expires_at: float | None, now: float) -> bool:
                            return bool(expires_at and expires_at < now)
                        '''
                    ),
                    "cache_entries.py": source(
                        '''
                        def cache_entry_expired(expires_at: float | None, now: float) -> bool:
                            return expires_at is not None and now > expires_at
                        '''
                    ),
                    "tests/test_expiry.py": source(
                        '''
                        from cache_entries import cache_entry_expired
                        from expiry import is_expired
                        from sessions import session_expired


                        def test_no_expiry_never_expires():
                            for function in (is_expired, session_expired, cache_entry_expired):
                                assert function(None, 100.0) is False


                        def test_expiry_boundary_is_inclusive_everywhere():
                            for function in (is_expired, session_expired, cache_entry_expired):
                                assert function(100.0, 99.9) is False
                                assert function(100.0, 100.0) is True
                                assert function(100.0, 100.1) is True
                        '''
                    ),
                },
            ),
            fixture(
                case_id="deep-page-limit",
                category="deep",
                task=(
                    "Implement one shared page-limit normalizer. None or blank input maps to 50; "
                    "decimal input is accepted from 1 through 100 inclusive; all other values "
                    "raise ValueError. api_limit and service_limit must delegate to "
                    "normalize_limit. Preserve APIs and tests, then verify."
                ),
                oracle=completed_oracle(
                    name="page-limit-shared-normalizer",
                    allowed=["limits.py", "api_limits.py", "service_limits.py"],
                    required=["limits.py", "api_limits.py", "service_limits.py"],
                    calls=[
                        ("api_limits.py", "api_limit", "normalize_limit"),
                        ("service_limits.py", "service_limit", "normalize_limit"),
                    ],
                ),
                files={
                    "limits.py": source(
                        '''
                        def normalize_limit(value: str | None) -> int:
                            return int(value or 25)
                        '''
                    ),
                    "api_limits.py": source(
                        '''
                        def api_limit(value: str | None) -> int:
                            return int(value or 50)
                        '''
                    ),
                    "service_limits.py": source(
                        '''
                        def service_limit(value: str | None) -> int:
                            return int(value or 100)
                        '''
                    ),
                    "tests/test_limits.py": source(
                        '''
                        import pytest

                        from api_limits import api_limit
                        from limits import normalize_limit
                        from service_limits import service_limit


                        def test_defaults_and_valid_limits_are_shared():
                            for function in (normalize_limit, api_limit, service_limit):
                                assert function(None) == 50
                                assert function("  ") == 50
                                assert function("1") == 1
                                assert function("100") == 100


                        @pytest.mark.parametrize("value", ["0", "101", "-4", "many"])
                        def test_invalid_limits_are_rejected_everywhere(value):
                            for function in (normalize_limit, api_limit, service_limit):
                                with pytest.raises(ValueError):
                                    function(value)
                        '''
                    ),
                },
            ),
            fixture(
                case_id="deep-permission-policy",
                category="deep",
                task=(
                    "Centralize edit authorization. can_edit must allow an owner or an admin and "
                    "deny every other actor. document_can_edit and comment_can_edit must both "
                    "delegate to can_edit. Preserve public APIs and tests, then verify."
                ),
                oracle=completed_oracle(
                    name="edit-permission-shared-policy",
                    allowed=["permissions.py", "documents.py", "comments.py"],
                    required=["permissions.py", "documents.py", "comments.py"],
                    calls=[
                        ("documents.py", "document_can_edit", "can_edit"),
                        ("comments.py", "comment_can_edit", "can_edit"),
                    ],
                ),
                files={
                    "permissions.py": source(
                        '''
                        def can_edit(owner_id: int, actor_id: int, is_admin: bool) -> bool:
                            return owner_id == actor_id
                        '''
                    ),
                    "documents.py": source(
                        '''
                        def document_can_edit(owner_id: int, actor_id: int, is_admin: bool) -> bool:
                            return owner_id == actor_id
                        '''
                    ),
                    "comments.py": source(
                        '''
                        def comment_can_edit(owner_id: int, actor_id: int, is_admin: bool) -> bool:
                            return is_admin
                        '''
                    ),
                    "tests/test_permissions.py": source(
                        '''
                        from comments import comment_can_edit
                        from documents import document_can_edit
                        from permissions import can_edit


                        def test_owner_and_admin_are_allowed_everywhere():
                            for function in (can_edit, document_can_edit, comment_can_edit):
                                assert function(7, 7, False) is True
                                assert function(7, 9, True) is True


                        def test_unrelated_non_admin_is_denied_everywhere():
                            for function in (can_edit, document_can_edit, comment_can_edit):
                                assert function(7, 9, False) is False
                        '''
                    ),
                },
            ),
            fixture(
                case_id="deep-date-parser",
                category="deep",
                task=(
                    "Implement one shared ISO calendar-date parser. parse_date must trim input, "
                    "accept YYYY-MM-DD, and reject blanks, datetimes, and invalid dates. "
                    "report_date and export_date must delegate to parse_date. Preserve public "
                    "APIs and tests, then verify."
                ),
                oracle=completed_oracle(
                    name="date-shared-parser",
                    allowed=["dates.py", "reports.py", "exports.py"],
                    required=["dates.py", "reports.py", "exports.py"],
                    calls=[
                        ("reports.py", "report_date", "parse_date"),
                        ("exports.py", "export_date", "parse_date"),
                    ],
                ),
                files={
                    "dates.py": source(
                        '''
                        from datetime import date


                        def parse_date(value: str) -> date:
                            return date.fromisoformat(value)
                        '''
                    ),
                    "reports.py": source(
                        '''
                        from datetime import date


                        def report_date(value: str) -> date:
                            return date.fromisoformat(value)
                        '''
                    ),
                    "exports.py": source(
                        '''
                        def export_date(value: str) -> str:
                            return value
                        '''
                    ),
                    "tests/test_dates.py": source(
                        '''
                        from datetime import date

                        import pytest

                        from dates import parse_date
                        from exports import export_date
                        from reports import report_date


                        def test_valid_calendar_date_is_shared():
                            for function in (parse_date, report_date, export_date):
                                assert function(" 2026-08-19 ") == date(2026, 8, 19)


                        @pytest.mark.parametrize("value", ["", "   ", "2026-08-19T12:00:00", "2026-02-30"])
                        def test_invalid_calendar_dates_are_rejected(value):
                            for function in (parse_date, report_date, export_date):
                                with pytest.raises(ValueError):
                                    function(value)
                        '''
                    ),
                },
            ),
        ]
    )

    # Repair-oriented tasks with several behavioral edges.
    cases.extend(
        [
            fixture(
                case_id="repair-nested-get",
                category="repair",
                task=(
                    "Repair nested_get so it traverses dot-separated dictionary keys without "
                    "treating valid falsey values as missing. Return the supplied default only "
                    "when a path component is absent or the current value is not a mapping. "
                    "Preserve the API and tests, then verify."
                ),
                oracle=completed_oracle(
                    name="nested-get-contract",
                    allowed=["nested.py"],
                    required=["nested.py"],
                ),
                files={
                    "nested.py": source(
                        '''
                        from typing import Any


                        def nested_get(data: dict[str, Any], path: str, default: Any = None) -> Any:
                            current: Any = data
                            for part in path.split("."):
                                current = current.get(part) or default
                            return current
                        '''
                    ),
                    "tests/test_nested.py": source(
                        '''
                        from nested import nested_get


                        def test_falsey_values_are_preserved():
                            data = {"a": {"zero": 0, "false": False, "empty": ""}}
                            assert nested_get(data, "a.zero", "missing") == 0
                            assert nested_get(data, "a.false", "missing") is False
                            assert nested_get(data, "a.empty", "missing") == ""


                        def test_missing_or_non_mapping_path_uses_default():
                            data = {"a": {"value": 3}, "leaf": 9}
                            assert nested_get(data, "a.none", "missing") == "missing"
                            assert nested_get(data, "leaf.child", "missing") == "missing"
                        '''
                    ),
                },
            ),
            fixture(
                case_id="repair-semver",
                category="repair",
                task=(
                    "Repair version_at_least to compare dot-separated non-negative integer "
                    "components numerically, padding missing trailing components with zero. "
                    "Invalid versions must raise ValueError. Preserve the public API and tests, "
                    "then verify."
                ),
                oracle=completed_oracle(
                    name="semver-contract",
                    allowed=["versions.py"],
                    required=["versions.py"],
                ),
                files={
                    "versions.py": source(
                        '''
                        def version_at_least(current: str, required: str) -> bool:
                            return current >= required
                        '''
                    ),
                    "tests/test_versions.py": source(
                        '''
                        import pytest

                        from versions import version_at_least


                        def test_numeric_components_and_padding():
                            assert version_at_least("2.10", "2.9") is True
                            assert version_at_least("2.9", "2.10") is False
                            assert version_at_least("1.2", "1.2.0") is True
                            assert version_at_least("1.2.1", "1.2") is True


                        @pytest.mark.parametrize("value", ["1.x", "-1.2", "1..2", ""])
                        def test_invalid_versions_are_rejected(value):
                            with pytest.raises(ValueError):
                                version_at_least(value, "1.0")
                        '''
                    ),
                },
            ),
            fixture(
                case_id="repair-chunks",
                category="repair",
                task=(
                    "Repair chunks so it returns every input item in consecutive lists of at "
                    "most size elements, including a final partial chunk. Non-positive sizes "
                    "must raise ValueError. Preserve the API and tests, then verify."
                ),
                oracle=completed_oracle(
                    name="chunks-contract",
                    allowed=["chunks.py"],
                    required=["chunks.py"],
                ),
                files={
                    "chunks.py": source(
                        '''
                        def chunks(items: list[int], size: int) -> list[list[int]]:
                            return [items[index:index + size] for index in range(0, len(items) - 1, size)]
                        '''
                    ),
                    "tests/test_chunks.py": source(
                        '''
                        import pytest

                        from chunks import chunks


                        def test_complete_and_partial_chunks_are_returned():
                            assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
                            assert chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
                            assert chunks([], 3) == []


                        @pytest.mark.parametrize("size", [0, -1])
                        def test_non_positive_size_is_rejected(size):
                            with pytest.raises(ValueError):
                                chunks([1, 2], size)
                        '''
                    ),
                },
            ),
            fixture(
                case_id="repair-cache-ttl",
                category="repair",
                task=(
                    "Repair cache_expired. A None TTL never expires; a negative TTL is invalid; "
                    "otherwise the entry expires when now is greater than or equal to created_at "
                    "plus TTL. Preserve the public API and tests, then verify."
                ),
                oracle=completed_oracle(
                    name="cache-ttl-contract",
                    allowed=["cache.py"],
                    required=["cache.py"],
                ),
                files={
                    "cache.py": source(
                        '''
                        def cache_expired(created_at: float, ttl: float | None, now: float) -> bool:
                            return bool(ttl and created_at + ttl < now)
                        '''
                    ),
                    "tests/test_cache.py": source(
                        '''
                        import pytest

                        from cache import cache_expired


                        def test_none_ttl_never_expires_and_boundary_is_inclusive():
                            assert cache_expired(10.0, None, 10_000.0) is False
                            assert cache_expired(10.0, 5.0, 14.9) is False
                            assert cache_expired(10.0, 5.0, 15.0) is True
                            assert cache_expired(10.0, 0.0, 10.0) is True


                        def test_negative_ttl_is_rejected():
                            with pytest.raises(ValueError):
                                cache_expired(10.0, -1.0, 10.0)
                        '''
                    ),
                },
            ),
            fixture(
                case_id="repair-csv-fields",
                category="repair",
                task=(
                    "Repair parse_csv_fields so it parses one CSV record using standard CSV "
                    "quoting rules, including commas and escaped quotes inside quoted fields. "
                    "Preserve the API and tests, then verify."
                ),
                oracle=completed_oracle(
                    name="csv-fields-contract",
                    allowed=["csv_fields.py"],
                    required=["csv_fields.py"],
                ),
                files={
                    "csv_fields.py": source(
                        '''
                        def parse_csv_fields(record: str) -> list[str]:
                            return record.split(",")
                        '''
                    ),
                    "tests/test_csv_fields.py": source(
                        '''
                        from csv_fields import parse_csv_fields


                        def test_plain_and_quoted_fields():
                            assert parse_csv_fields("a,b,c") == ["a", "b", "c"]
                            assert parse_csv_fields('a,"b,c",d') == ["a", "b,c", "d"]


                        def test_escaped_quote_inside_field():
                            assert parse_csv_fields('"a""b",c') == ['a"b', "c"]
                        '''
                    ),
                },
            ),
            fixture(
                case_id="repair-origin-normalize",
                category="repair",
                task=(
                    "Repair normalize_origin for HTTP and HTTPS origins. Lowercase the scheme "
                    "and host, remove default ports 80 and 443, preserve non-default ports, and "
                    "reject userinfo, paths other than '/', queries, fragments, or unsupported "
                    "schemes. Return the canonical origin without a trailing slash. Preserve the "
                    "API and tests, then verify."
                ),
                oracle=completed_oracle(
                    name="origin-contract",
                    allowed=["origin.py"],
                    required=["origin.py"],
                ),
                files={
                    "origin.py": source(
                        '''
                        def normalize_origin(value: str) -> str:
                            return value.lower().rstrip("/")
                        '''
                    ),
                    "tests/test_origin.py": source(
                        '''
                        import pytest

                        from origin import normalize_origin


                        def test_canonical_http_and_https_origins():
                            assert normalize_origin("HTTPS://Example.COM:443/") == "https://example.com"
                            assert normalize_origin("http://Example.COM:80") == "http://example.com"
                            assert normalize_origin("https://Example.COM:8443/") == "https://example.com:8443"


                        @pytest.mark.parametrize(
                            "value",
                            [
                                "ftp://example.com",
                                "https://user@example.com",
                                "https://example.com/path",
                                "https://example.com/?q=1",
                                "https://example.com/#fragment",
                            ],
                        )
                        def test_non_origin_values_are_rejected(value):
                            with pytest.raises(ValueError):
                                normalize_origin(value)
                        '''
                    ),
                },
            ),
        ]
    )

    # Already-correct audits.
    cases.extend(
        [
            fixture(
                case_id="nochange-safe-divide",
                category="no-change",
                baseline="pass",
                task=(
                    "Audit safe_divide. It should return the supplied default only when the "
                    "denominator is zero and otherwise return the quotient. If correct, make no "
                    "changes and verify."
                ),
                oracle=completed_oracle(
                    name="safe-divide-no-change",
                    allowed=[],
                ),
                files={
                    "math_util.py": source(
                        '''
                        def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
                            if denominator == 0:
                                return default
                            return numerator / denominator
                        '''
                    ),
                    "tests/test_math_util.py": source(
                        '''
                        from math_util import safe_divide


                        def test_safe_divide_contract():
                            assert safe_divide(8, 2) == 4
                            assert safe_divide(8, 0) == 0.0
                            assert safe_divide(8, 0, default=-1) == -1
                            assert safe_divide(0, 5) == 0
                        '''
                    ),
                },
            ),
            fixture(
                case_id="nochange-stable-dedupe",
                category="no-change",
                baseline="pass",
                task=(
                    "Audit dedupe_names. It should preserve the first occurrence of each exact "
                    "name and preserve input order. If correct, make no changes and verify."
                ),
                oracle=completed_oracle(
                    name="stable-dedupe-no-change",
                    allowed=[],
                ),
                files={
                    "names.py": source(
                        '''
                        def dedupe_names(names: list[str]) -> list[str]:
                            seen: set[str] = set()
                            result: list[str] = []
                            for name in names:
                                if name in seen:
                                    continue
                                seen.add(name)
                                result.append(name)
                            return result
                        '''
                    ),
                    "tests/test_names.py": source(
                        '''
                        from names import dedupe_names


                        def test_exact_first_occurrences_are_preserved():
                            assert dedupe_names(["Ada", "Bob", "Ada", "ada"]) == ["Ada", "Bob", "ada"]
                            assert dedupe_names([]) == []
                        '''
                    ),
                },
            ),
            fixture(
                case_id="nochange-slug-check",
                category="no-change",
                baseline="pass",
                task=(
                    "Audit is_slug. A valid slug is non-empty lowercase ASCII letters or digits "
                    "separated by single hyphens, with no leading or trailing hyphen. If correct, "
                    "make no changes and verify."
                ),
                oracle=completed_oracle(
                    name="slug-check-no-change",
                    allowed=[],
                ),
                files={
                    "slugs.py": source(
                        '''
                        import re

                        _SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\\Z")


                        def is_slug(value: str) -> bool:
                            return bool(_SLUG.fullmatch(value))
                        '''
                    ),
                    "tests/test_slugs.py": source(
                        '''
                        import pytest

                        from slugs import is_slug


                        @pytest.mark.parametrize("value", ["a", "abc-123", "0-1-2"])
                        def test_valid_slugs(value):
                            assert is_slug(value) is True


                        @pytest.mark.parametrize("value", ["", "Abc", "-abc", "abc-", "a--b", "a_b"])
                        def test_invalid_slugs(value):
                            assert is_slug(value) is False
                        '''
                    ),
                },
            ),
        ]
    )

    # Internally contradictory contracts that must terminate through bounded abort.
    cases.extend(
        [
            fixture(
                case_id="impossible-mode-label",
                category="impossible",
                expected_status="failed",
                task=(
                    "Repair mode_label while preserving its public API and every test. Do not "
                    "edit tests. If the contract is contradictory, terminate safely through "
                    "bounded verification instead of claiming success."
                ),
                oracle=None,
                files={
                    "modes.py": source(
                        '''
                        def mode_label(value: str) -> str:
                            return value
                        '''
                    ),
                    "tests/test_modes.py": source(
                        '''
                        from modes import mode_label


                        def test_safe_mode_is_safe():
                            assert mode_label("safe") == "safe"


                        def test_same_safe_mode_is_secure():
                            assert mode_label("safe") == "secure"
                        '''
                    ),
                },
            ),
            fixture(
                case_id="impossible-parse-number",
                category="impossible",
                expected_status="failed",
                task=(
                    "Repair parse_number while preserving its public API and every test. Do not "
                    "edit tests. If the requirements conflict for the same input, use bounded "
                    "verification and abort safely rather than reporting success."
                ),
                oracle=None,
                files={
                    "numbers.py": source(
                        '''
                        def parse_number(value: str):
                            return int(value)
                        '''
                    ),
                    "tests/test_numbers.py": source(
                        '''
                        from numbers import parse_number


                        def test_seven_is_integer():
                            assert parse_number("7") == 7
                            assert type(parse_number("7")) is int


                        def test_same_seven_is_string():
                            assert parse_number("7") == "7"
                            assert type(parse_number("7")) is str
                        '''
                    ),
                },
            ),
            fixture(
                case_id="impossible-default-region",
                category="impossible",
                expected_status="failed",
                task=(
                    "Repair default_region while preserving its public API and every test. Do "
                    "not edit tests. If the tests impose contradictory constants, terminate via "
                    "the bounded abort path and do not fabricate success."
                ),
                oracle=None,
                files={
                    "regions.py": source(
                        '''
                        def default_region() -> str:
                            return "us-east-1"
                        '''
                    ),
                    "tests/test_regions.py": source(
                        '''
                        from regions import default_region


                        def test_default_region_is_us_east():
                            assert default_region() == "us-east-1"


                        def test_same_default_region_is_eu_west():
                            assert default_region() == "eu-west-1"
                        '''
                    ),
                },
            ),
        ]
    )

    validate_fixtures(cases)
    return cases


def validate_fixtures(cases: list[dict[str, Any]]) -> None:
    ids = [str(item["id"]) for item in cases]
    if len(ids) != TOTAL_CASES:
        raise ValueError(f"expected {TOTAL_CASES} fixtures, found {len(ids)}")
    if len(set(ids)) != len(ids):
        raise ValueError("fixture IDs must be unique")
    counts = Counter(str(item["category"]) for item in cases)
    if dict(counts) != CATEGORY_COUNTS:
        raise ValueError(f"unexpected category counts: {dict(counts)}")
    for item in cases:
        expected = str(item["expected_status"])
        if expected not in {"completed", "failed"}:
            raise ValueError(f"invalid expected status for {item['id']}: {expected}")
        if expected == "completed" and item.get("oracle") is None:
            raise ValueError(f"completed fixture lacks oracle: {item['id']}")
        if expected == "failed" and item.get("oracle") is not None:
            raise ValueError(f"impossible fixture should not have oracle: {item['id']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--static-first-manifest", required=True)
    parser.add_argument("--full-first-manifest", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=73157)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    static_first_manifest = Path(args.static_first_manifest).expanduser().resolve()
    full_first_manifest = Path(args.full_first_manifest).expanduser().resolve()
    cases_path = Path(args.cases).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)

    case_records: list[dict[str, Any]] = []
    grouped_records: dict[str, list[dict[str, Any]]] = {
        "static-first": [],
        "full-first": [],
    }

    for index, item in enumerate(fixtures(), 1):
        case_id = str(item["id"])
        category = str(item["category"])
        pair_order = "static-first" if index % 2 == 1 else "full-first"
        repo = root / case_id
        repo.mkdir(parents=True)
        for relative, content in dict(item["files"]).items():
            write(repo / relative, str(content))
        run("git", "init", "-q", cwd=repo)
        run("git", "config", "user.email", "counterbalanced@example.invalid", cwd=repo)
        run("git", "config", "user.name", "Graph Counterbalanced Evaluation", cwd=repo)
        run("git", "add", ".", cwd=repo)
        run("git", "commit", "-qm", "counterbalanced held-out fixture", cwd=repo)
        base_commit = run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
        if args.skip_baseline:
            baseline_status = str(item["baseline"])
        else:
            print(
                f"Baseline {index:02d}/{TOTAL_CASES}: {case_id} ",
                end="",
                flush=True,
            )
            baseline_env = dict(os.environ)
            baseline_env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
            baseline = run(
                args.python,
                "-m",
                "pytest",
                "-q",
                cwd=repo,
                check=False,
                env=baseline_env,
            )
            baseline_status = "pass" if baseline.returncode == 0 else "fail"
            print(baseline_status, flush=True)
            if baseline_status != item["baseline"]:
                print(baseline.stdout)
                print(baseline.stderr, file=sys.stderr)
                raise SystemExit(
                    f"{case_id}: expected baseline {item['baseline']}, got {baseline_status}"
                )
        status = run(
            "git", "status", "--porcelain=v1", "--untracked-files=all", cwd=repo
        ).stdout
        if status.strip():
            raise SystemExit(f"{case_id}: repository is dirty after baseline:\n{status}")

        run_id = f"counterbalanced-v1-{index:02d}-{case_id}"
        manifest_record: dict[str, Any] = {
            "run_id": run_id,
            "case_id": case_id,
            "repo": str(repo),
            "repository_alias": f"<repository:{case_id}>",
            "paired_evaluation": True,
            "evaluation_seed": args.seed + index,
            "task": item["task"],
            "test_commands": [f"{args.python} -m pytest -q"],
            "tags": [
                "counterbalanced-policy-eval-v1",
                category,
                pair_order,
                str(item["expected_status"]),
            ],
        }
        if item["oracle"] is not None:
            manifest_record["contract_oracle"] = item["oracle"]
        grouped_records[pair_order].append(manifest_record)
        case_records.append(
            {
                "index": index,
                "id": case_id,
                "category": category,
                "pair_order": pair_order,
                "repo": str(repo),
                "base_commit": base_commit,
                "baseline": item["baseline"],
                "baseline_validated": not args.skip_baseline,
                "expected_status": item["expected_status"],
                "run_id": run_id,
                "task": item["task"],
            }
        )

    for path, order in (
        (static_first_manifest, "static-first"),
        (full_first_manifest, "full-first"),
    ):
        records = grouped_records[order]
        if len(records) != TOTAL_CASES // 2:
            raise SystemExit(f"{order}: expected 12 cases, found {len(records)}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )

    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "purpose": "24-case counterbalanced static-versus-full policy evaluation",
                "seed": args.seed,
                "category_counts": CATEGORY_COUNTS,
                "order_counts": {
                    key: len(value) for key, value in grouped_records.items()
                },
                "cases": case_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(case_records)} held-out repositories")
    print("Category counts:", dict(Counter(item["category"] for item in case_records)))
    print("Order counts:", dict(Counter(item["pair_order"] for item in case_records)))
    print("Static-first manifest:", static_first_manifest)
    print("Full-first manifest:", full_first_manifest)
    print("Cases:", cases_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
