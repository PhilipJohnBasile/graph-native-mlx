from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .baseline_loop import RalphLoopBaseline
from .graph import load_default_graph
from .provider import ModelProvider
from .runtime import GraphRuntime
from .store import SQLiteRunStore


@dataclass(frozen=True)
class SystemAggregate:
    expected_outcome_rate: float
    average_llm_calls: float
    average_tool_calls: float
    average_tokens: float
    average_path_length: float


async def run_graph_vs_loop_benchmark(
    *,
    input_path: str | Path,
    provider: ModelProvider,
    output_path: str | Path | None = None,
    loop_attempts: int = 3,
) -> dict[str, Any]:
    cases = _read_cases(input_path)
    if not cases:
        raise ValueError("benchmark input contains no cases")

    with tempfile.TemporaryDirectory(prefix="graph-model-benchmark-") as temp_dir:
        store = SQLiteRunStore(Path(temp_dir) / "runs.sqlite3")
        runtime = GraphRuntime(graph=load_default_graph(), store=store, provider=provider)
        loop = RalphLoopBaseline(provider=provider, max_attempts=loop_attempts)
        rows: list[dict[str, Any]] = []

        for index, case in enumerate(cases):
            task = case["task"]
            expected = case.get("expected_status", "completed")
            graph_state = await runtime.run(task, run_id=f"benchmark-{index}")
            loop_result = await loop.run(task)
            rows.append(
                {
                    "id": case.get("id", str(index)),
                    "task": task,
                    "expected_status": expected,
                    "graph": {
                        "status": graph_state.status,
                        "matches_expected": graph_state.status == expected,
                        "path": graph_state.completed_nodes,
                        "metrics": {
                            **graph_state.metrics.model_dump(),
                            "total_tokens": graph_state.metrics.total_tokens,
                        },
                    },
                    "loop": {
                        "status": loop_result.status,
                        "matches_expected": loop_result.status == expected,
                        "path": list(loop_result.path),
                        "attempts": loop_result.attempts,
                        "metrics": {
                            **loop_result.metrics.model_dump(),
                            "total_tokens": loop_result.metrics.total_tokens,
                        },
                    },
                }
            )

    graph_aggregate = _aggregate(rows, "graph")
    loop_aggregate = _aggregate(rows, "loop")
    report = {
        "cases": rows,
        "summary": {
            "case_count": len(rows),
            "graph": asdict(graph_aggregate),
            "loop": asdict(loop_aggregate),
            "delta_graph_minus_loop": {
                "expected_outcome_rate": graph_aggregate.expected_outcome_rate
                - loop_aggregate.expected_outcome_rate,
                "average_llm_calls": graph_aggregate.average_llm_calls
                - loop_aggregate.average_llm_calls,
                "average_tool_calls": graph_aggregate.average_tool_calls
                - loop_aggregate.average_tool_calls,
                "average_tokens": graph_aggregate.average_tokens - loop_aggregate.average_tokens,
                "average_path_length": graph_aggregate.average_path_length
                - loop_aggregate.average_path_length,
            },
        },
        "interpretation": {
            "purpose": "Compare graph-controlled reuse and routing against retry-from-the-top execution.",
            "warning": "Mock-mode results test control-flow economics, not model intelligence. Use real tasks, deterministic evaluators, and a local model endpoint for capability claims.",
        },
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _read_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid benchmark JSONL at line {line_number}: {exc}") from exc
            if not isinstance(item, dict) or not isinstance(item.get("task"), str):
                raise ValueError(f"benchmark line {line_number} must contain a string task")
            expected = item.get("expected_status", "completed")
            if expected not in {"completed", "failed"}:
                raise ValueError(
                    f"benchmark line {line_number} has invalid expected_status {expected!r}"
                )
            cases.append(item)
    return cases


def _aggregate(rows: list[dict[str, Any]], key: str) -> SystemAggregate:
    count = len(rows)
    return SystemAggregate(
        expected_outcome_rate=sum(row[key]["matches_expected"] for row in rows) / count,
        average_llm_calls=sum(row[key]["metrics"]["llm_calls"] for row in rows) / count,
        average_tool_calls=sum(row[key]["metrics"]["tool_calls"] for row in rows) / count,
        average_tokens=sum(row[key]["metrics"]["total_tokens"] for row in rows) / count,
        average_path_length=sum(len(row[key]["path"]) for row in rows) / count,
    )
