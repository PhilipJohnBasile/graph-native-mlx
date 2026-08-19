from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .controller import GraphController
from .models import GraphSpec, RunState
from .provider import ModelProvider
from .runtime import GraphRuntime
from .store import SQLiteRunStore
from .workspace import DEFAULT_ALLOWED_COMMANDS, workspace_initial_data


@dataclass(frozen=True)
class RepositoryTraceTask:
    run_id: str
    task: str
    repo: str
    test_commands: tuple[str, ...] = ()
    allowed_commands: tuple[str, ...] = ()
    base_ref: str = "HEAD"
    workspace_mode: str = "worktree"
    allow_sensitive_paths: bool = False
    tags: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        index: int,
        run_prefix: str,
    ) -> "RepositoryTraceTask":
        task = str(payload.get("task", "")).strip()
        repo = str(payload.get("repo", "")).strip()
        if not task:
            raise ValueError(f"trace manifest record {index}: task is required")
        if not repo:
            raise ValueError(f"trace manifest record {index}: repo is required")
        run_id = str(payload.get("run_id", "")).strip()
        if not run_id:
            digest = hashlib.sha256(f"{repo}\0{task}".encode("utf-8")).hexdigest()[:10]
            run_id = f"{run_prefix}-{index:04d}-{digest}"

        def strings(name: str) -> tuple[str, ...]:
            value = payload.get(name, ())
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"trace manifest record {index}: {name} must be a list")
            return tuple(str(item).strip() for item in value if str(item).strip())

        return cls(
            run_id=run_id,
            task=task,
            repo=repo,
            test_commands=strings("test_commands"),
            allowed_commands=strings("allowed_commands"),
            base_ref=str(payload.get("base_ref", "HEAD")).strip() or "HEAD",
            workspace_mode=str(payload.get("workspace_mode", "worktree")).strip(),
            allow_sensitive_paths=bool(payload.get("allow_sensitive_paths", False)),
            tags=strings("tags"),
        )


def read_repository_trace_manifest(
    path: str | Path,
    *,
    run_prefix: str = "mlx-trace",
) -> list[RepositoryTraceTask]:
    records: list[RepositoryTraceTask] = []
    seen_run_ids: set[str] = set()
    record_index = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid trace manifest JSONL at line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"trace manifest line {line_number} must contain a JSON object"
                )
            record_index += 1
            record = RepositoryTraceTask.from_mapping(
                payload,
                index=record_index,
                run_prefix=run_prefix,
            )
            if record.run_id in seen_run_ids:
                raise ValueError(f"duplicate trace run_id {record.run_id!r}")
            seen_run_ids.add(record.run_id)
            records.append(record)
    if not records:
        raise ValueError("trace manifest contains no tasks")
    return records


def _hidden_artifact_count(store: SQLiteRunStore, run_id: str) -> int:
    digests: set[str] = set()
    for event in store.events(run_id):
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        result = payload.get("result")
        delta = result.get("delta") if isinstance(result, dict) else None
        decisions: list[Any] = []
        if isinstance(delta, dict) and isinstance(delta.get("router"), dict):
            decisions.append(delta["router"])
        decisions.extend(
            decision
            for decision in (payload.get("stop_decision"), payload.get("edge_decision"))
            if isinstance(decision, dict)
        )
        for decision in decisions:
            metrics = decision.get("policy_metrics")
            hidden = metrics.get("hidden_state") if isinstance(metrics, dict) else None
            digest = hidden.get("sha256") if isinstance(hidden, dict) else None
            if isinstance(digest, str) and digest:
                digests.add(digest)
    return len(digests)


def _state_summary(
    state: RunState,
    *,
    tags: Sequence[str],
    hidden_artifacts: int,
    existing: bool = False,
) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "status": state.status,
        "existing": existing,
        "tags": list(tags),
        "steps": state.step_count,
        "completed_nodes": list(state.completed_nodes),
        "llm_calls": state.metrics.llm_calls,
        "tool_calls": state.metrics.tool_calls,
        "policy_calls": state.metrics.policy_calls,
        "prompt_tokens": state.metrics.prompt_tokens,
        "completion_tokens": state.metrics.completion_tokens,
        "policy_prompt_tokens": state.metrics.policy_prompt_tokens,
        "total_tokens": state.metrics.total_tokens,
        "elapsed_seconds": state.metrics.elapsed_seconds,
        "hidden_artifacts": hidden_artifacts,
        "error": state.error,
    }


async def collect_repository_traces(
    *,
    tasks: Iterable[RepositoryTraceTask],
    graph: GraphSpec,
    store: SQLiteRunStore,
    provider: ModelProvider,
    controller: GraphController,
    resume_existing: bool = False,
    continue_on_error: bool = True,
    workspace_home: str | Path | None = None,
    artifact_root: str | Path | None = None,
    default_allowed_commands: Sequence[str] = DEFAULT_ALLOWED_COMMANDS,
    command_timeout_seconds: float = 300.0,
    max_command_output_bytes: int = 200_000,
    max_context_files: int = 18,
    max_context_file_bytes: int = 40_000,
    max_context_bytes: int = 180_000,
    max_patch_bytes: int = 500_000,
    max_patch_files: int = 32,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    task_list = tuple(tasks)
    runtime = GraphRuntime(
        graph=graph,
        store=store,
        provider=provider,
        controller=controller,
    )
    results: list[dict[str, Any]] = []
    for task_index, task in enumerate(task_list, 1):
        if progress is not None:
            progress({"event": "task_start", "index": task_index, "total": len(task_list), "run_id": task.run_id})
        try:
            existing = store.load_run(task.run_id)
            if existing is not None:
                if existing.status == "running" and resume_existing:
                    state = await runtime.run(run_id=task.run_id)
                    results.append(
                        _state_summary(
                            state,
                            tags=task.tags,
                            hidden_artifacts=_hidden_artifact_count(store, task.run_id),
                            existing=True,
                        )
                    )
                    continue
                if existing.status in {"completed", "failed"}:
                    results.append(
                        _state_summary(
                            existing,
                            tags=task.tags,
                            hidden_artifacts=_hidden_artifact_count(store, task.run_id),
                            existing=True,
                        )
                    )
                    continue
                raise ValueError(
                    f"run {task.run_id!r} already exists; pass resume_existing to continue it"
                )

            initial_data = workspace_initial_data(
                source_root=task.repo,
                mode=task.workspace_mode,
                base_ref=task.base_ref,
                workspace_home=workspace_home,
                artifact_root=artifact_root,
                test_commands=task.test_commands,
                allowed_commands=(task.allowed_commands or tuple(default_allowed_commands)),
                command_timeout_seconds=command_timeout_seconds,
                max_command_output_bytes=max_command_output_bytes,
                max_context_files=max_context_files,
                max_context_file_bytes=max_context_file_bytes,
                max_context_bytes=max_context_bytes,
                max_patch_bytes=max_patch_bytes,
                max_patch_files=max_patch_files,
                allow_sensitive_paths=task.allow_sensitive_paths,
            )
            initial_data["trace_manifest"] = {
                "tags": list(task.tags),
                "manifest_repo": str(Path(task.repo).expanduser()),
            }
            state = await runtime.run(
                task.task,
                run_id=task.run_id,
                initial_data=initial_data,
            )
            results.append(
                _state_summary(
                    state,
                    tags=task.tags,
                    hidden_artifacts=_hidden_artifact_count(store, task.run_id),
                )
            )
        except Exception as exc:  # noqa: BLE001 - collection must report per-task failures
            error_text = f"{type(exc).__name__}: {exc}"
            failed_state = store.load_run(task.run_id)
            if failed_state is not None and failed_state.status == "running":
                failed_state.status = "failed"
                failed_state.error = f"collector error: {error_text}"
                store.save_terminal_event(failed_state, "run_failed")
            result = {
                "run_id": task.run_id,
                "status": "collector_error",
                "existing": False,
                "tags": list(task.tags),
                "hidden_artifacts": _hidden_artifact_count(store, task.run_id),
                "error": error_text,
            }
            results.append(result)
            if progress is not None:
                progress({"event": "task_error", "index": task_index, "total": len(task_list), **result})
            if not continue_on_error:
                break
        else:
            if progress is not None and results:
                progress({"event": "task_done", "index": task_index, "total": len(task_list), **results[-1]})

    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "format_version": 1,
        "graph": {"name": graph.name, "version": graph.version},
        "provider": provider.identity,
        "controller": controller.identity,
        "requested_tasks": len(task_list),
        "tasks": len(results),
        "status_counts": status_counts,
        "results": results,
    }


def write_trace_collection_summary(
    summary: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return output
