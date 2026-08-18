from __future__ import annotations

import json
from pathlib import Path

from .store import SQLiteRunStore


def export_router_training_data(store: SQLiteRunStore, output: str | Path) -> int:
    """Export one route/outcome record per run for router training or preference construction."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for state in store.list_runs():
            path = [
                event["node_id"]
                for event in store.events(state.run_id)
                if event["event_type"] == "node_completed" and event["node_id"]
            ]
            record = {
                "run_id": state.run_id,
                "task": state.task,
                "graph": f"{state.graph_name}@{state.graph_version}",
                "route": state.data.get("route"),
                "difficulty": state.data.get("difficulty"),
                "provider": (state.data.get("_runtime") or {}).get("provider"),
                "path": path,
                "success": state.status == "completed",
                "status": state.status,
                "repair_count": state.data.get("repair_count", 0),
                "metrics": {
                    **state.metrics.model_dump(),
                    "total_tokens": state.metrics.total_tokens,
                },
                "error": state.error,
            }
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            count += 1
    return count
