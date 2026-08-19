from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import RunState

PAIRED_EVALUATION_STATE_KEY = "_paired_evaluation"
PAIRED_EVALUATION_FORMAT = "graph-native-paired-evaluation-v1"
PROMPT_AUDIT_FORMAT = "graph-native-prompt-audit-v1"


@dataclass(frozen=True)
class PairedEvaluationConfig:
    enabled: bool = False
    case_id: str = ""
    repository_alias: str = "<repository>"
    worktree_alias: str = "<worktree>"
    artifact_alias: str = "<artifacts>"
    run_alias: str = "<paired-run>"
    base_seed: int = 42_057
    normalize_timing: bool = True
    prompt_audit: bool = True

    @classmethod
    def from_state(cls, state: RunState) -> "PairedEvaluationConfig":
        raw = state.data.get(PAIRED_EVALUATION_STATE_KEY)
        if not isinstance(raw, Mapping) or not bool(raw.get("enabled", False)):
            return cls()
        case_id = _safe_identifier(str(raw.get("case_id") or state.run_id))
        repository_alias = str(raw.get("repository_alias") or f"<repository:{case_id}>")
        worktree_alias = str(raw.get("worktree_alias") or f"<worktree:{case_id}>")
        artifact_alias = str(raw.get("artifact_alias") or f"<artifacts:{case_id}>")
        run_alias = str(raw.get("run_alias") or f"<run:{case_id}>")
        try:
            base_seed = int(raw.get("base_seed", 42_057))
        except (TypeError, ValueError) as exc:
            raise ValueError("paired evaluation base_seed must be an integer") from exc
        if base_seed < 0:
            raise ValueError("paired evaluation base_seed must be non-negative")
        return cls(
            enabled=True,
            case_id=case_id,
            repository_alias=repository_alias,
            worktree_alias=worktree_alias,
            artifact_alias=artifact_alias,
            run_alias=run_alias,
            base_seed=base_seed,
            normalize_timing=bool(raw.get("normalize_timing", True)),
            prompt_audit=bool(raw.get("prompt_audit", True)),
        )

    def as_state_payload(self) -> dict[str, Any]:
        return {
            "format": PAIRED_EVALUATION_FORMAT,
            "enabled": self.enabled,
            "case_id": self.case_id,
            "repository_alias": self.repository_alias,
            "worktree_alias": self.worktree_alias,
            "artifact_alias": self.artifact_alias,
            "run_alias": self.run_alias,
            "base_seed": self.base_seed,
            "normalize_timing": self.normalize_timing,
            "prompt_audit": self.prompt_audit,
        }


def _safe_identifier(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return normalized[:160] or "paired-case"


def paired_evaluation_enabled(state: RunState) -> bool:
    return PairedEvaluationConfig.from_state(state).enabled


def paired_elapsed_seconds(state: RunState) -> float:
    config = PairedEvaluationConfig.from_state(state)
    return 0.0 if config.enabled and config.normalize_timing else state.metrics.elapsed_seconds


def _path_replacements(state: RunState, config: PairedEvaluationConfig) -> list[tuple[str, str]]:
    replacements: dict[str, str] = {}

    def add(value: Any, alias: str) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        text = value.strip()
        replacements[text] = alias
        try:
            expanded = str(Path(text).expanduser())
        except (OSError, RuntimeError, ValueError):
            return
        replacements[expanded] = alias

    replacements[state.run_id] = config.run_alias
    workspace = state.data.get("workspace")
    if isinstance(workspace, Mapping):
        add(workspace.get("source_root"), config.repository_alias)
        add(workspace.get("active_root"), config.worktree_alias)
        add(workspace.get("workspace_home"), config.worktree_alias)
        add(workspace.get("artifact_root"), config.artifact_alias)
    trace_manifest = state.data.get("trace_manifest")
    if isinstance(trace_manifest, Mapping):
        add(trace_manifest.get("manifest_repo"), config.repository_alias)
    # Longest paths must be replaced first so a parent path cannot mask a more
    # specific worktree or artifact path.
    return sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)


def _canonicalize_text(
    value: str,
    *,
    replacements: Sequence[tuple[str, str]],
) -> str:
    result = value
    for original, alias in replacements:
        if original:
            result = result.replace(original, alias)
    return result


def canonicalize_prompt_value(value: Any, state: RunState) -> Any:
    """Return a prompt-safe paired-evaluation representation.

    The live runtime retains real paths, run IDs, and evidence. Only the copy
    rendered into a model prompt is canonicalized. This keeps static, shadow,
    route-only, and full-policy arms byte-comparable without weakening runtime
    identity or Git-worktree safety.
    """

    config = PairedEvaluationConfig.from_state(state)
    if not config.enabled:
        return value
    replacements = _path_replacements(state, config)

    def convert(candidate: Any, *, key: str | None = None) -> Any:
        if isinstance(candidate, Mapping):
            result: dict[str, Any] = {}
            for raw_key, item in candidate.items():
                name = str(raw_key)
                if name in {"started_at", "updated_at", "created_at", "timestamp"}:
                    continue
                if config.normalize_timing and name in {
                    "elapsed_seconds",
                    "duration_seconds",
                    "active_seconds",
                }:
                    result[name] = 0.0
                    continue
                result[name] = convert(item, key=name)
            return result
        if isinstance(candidate, (list, tuple)):
            return [convert(item) for item in candidate]
        if isinstance(candidate, Path):
            candidate = str(candidate)
        if isinstance(candidate, str):
            return _canonicalize_text(candidate, replacements=replacements)
        return candidate

    return convert(value)


def canonical_json(value: Any, state: RunState) -> str:
    return json.dumps(
        canonicalize_prompt_value(value, state),
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )


def logical_idempotency_key(
    state: RunState,
    *,
    node_id: str,
    revision: int,
    fallback: str,
) -> str:
    config = PairedEvaluationConfig.from_state(state)
    if not config.enabled:
        return fallback
    return f"{config.case_id}:{node_id}:revision-{revision}"


def deterministic_generation_seed(
    state: RunState,
    *,
    node_id: str,
    call_kind: str,
    revision: int,
    system: str,
    user: str,
) -> int | None:
    config = PairedEvaluationConfig.from_state(state)
    if not config.enabled:
        return None
    payload = "\0".join(
        (
            str(config.base_seed),
            config.case_id,
            node_id,
            call_kind,
            str(revision),
            hashlib.sha256(system.encode("utf-8")).hexdigest(),
            hashlib.sha256(user.encode("utf-8")).hexdigest(),
        )
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def prompt_audit_record(
    state: RunState,
    *,
    node_id: str,
    call_kind: str,
    revision: int,
    system: str,
    user: str,
    seed: int | None,
) -> dict[str, Any] | None:
    config = PairedEvaluationConfig.from_state(state)
    if not config.enabled or not config.prompt_audit:
        return None
    system_sha = hashlib.sha256(system.encode("utf-8")).hexdigest()
    user_sha = hashlib.sha256(user.encode("utf-8")).hexdigest()
    combined_sha = hashlib.sha256(
        (system + "\0" + user).encode("utf-8")
    ).hexdigest()
    return {
        "format": PROMPT_AUDIT_FORMAT,
        "case_id": config.case_id,
        "node_id": node_id,
        "call_kind": call_kind,
        "revision": revision,
        "system_sha256": system_sha,
        "user_sha256": user_sha,
        "combined_sha256": combined_sha,
        "generation_seed": seed,
        "raw_system_persisted": False,
        "raw_user_persisted": False,
    }


def prompt_audit_artifact_key(
    *,
    node_id: str,
    call_kind: str,
    revision: int,
) -> str:
    safe_kind = _safe_identifier(call_kind)
    return f"prompt-audit-{node_id}-{revision}-{safe_kind}.json"


async def complete_json_with_seed(
    provider: Any,
    *,
    system: str,
    user: str,
    temperature: float | None,
    seed: int | None,
):
    seeded = getattr(provider, "complete_json_seeded", None)
    if seed is not None and callable(seeded):
        return await seeded(
            system=system,
            user=user,
            temperature=temperature,
            seed=seed,
        )
    return await provider.complete_json(
        system=system,
        user=user,
        temperature=temperature,
    )


async def complete_patch_with_seed(
    provider: Any,
    *,
    system: str,
    user: str,
    temperature: float | None,
    seed: int | None,
):
    seeded = getattr(provider, "complete_patch_seeded", None)
    if seed is not None and callable(seeded):
        return await seeded(
            system=system,
            user=user,
            temperature=temperature,
            seed=seed,
        )
    return await provider.complete_patch(
        system=system,
        user=user,
        temperature=temperature,
    )


def evaluation_state_payload(
    *,
    case_id: str,
    repository_alias: str | None = None,
    base_seed: int = 42_057,
    normalize_timing: bool = True,
    prompt_audit: bool = True,
) -> dict[str, Any]:
    safe_case = _safe_identifier(case_id)
    config = PairedEvaluationConfig(
        enabled=True,
        case_id=safe_case,
        repository_alias=repository_alias or f"<repository:{safe_case}>",
        worktree_alias=f"<worktree:{safe_case}>",
        artifact_alias=f"<artifacts:{safe_case}>",
        run_alias=f"<run:{safe_case}>",
        base_seed=int(base_seed),
        normalize_timing=bool(normalize_timing),
        prompt_audit=bool(prompt_audit),
    )
    return config.as_state_payload()


def paired_eval_from_env() -> dict[str, Any] | None:
    raw = os.getenv("GRAPH_MODEL_PAIRED_EVAL", "").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return None
    case_id = os.getenv("GRAPH_MODEL_EVAL_CASE_ID", "paired-case")
    repository_alias = os.getenv("GRAPH_MODEL_EVAL_REPOSITORY_ALIAS") or None
    base_seed = int(os.getenv("GRAPH_MODEL_MLX_GENERATION_SEED", "42057"))
    return evaluation_state_payload(
        case_id=case_id,
        repository_alias=repository_alias,
        base_seed=base_seed,
    )
