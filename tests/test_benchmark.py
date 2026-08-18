import json
from pathlib import Path

import pytest

from graph_model.benchmark import run_graph_vs_loop_benchmark
from graph_model.provider import MockProvider


@pytest.mark.asyncio
async def test_graph_vs_loop_benchmark_exposes_retry_cost(tmp_path: Path) -> None:
    input_path = tmp_path / "cases.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "task": "fix failing CI [force-fail-once]",
                "expected_status": "completed",
            }
        )
        + "\n"
    )
    report = await run_graph_vs_loop_benchmark(
        input_path=input_path,
        provider=MockProvider(),
    )
    row = report["cases"][0]
    assert row["graph"]["status"] == "completed"
    assert row["loop"]["status"] == "completed"
    assert row["graph"]["metrics"]["llm_calls"] < row["loop"]["metrics"]["llm_calls"]
    assert row["graph"]["path"].count("plan") == 1
    assert row["loop"]["path"].count("plan") == 2
