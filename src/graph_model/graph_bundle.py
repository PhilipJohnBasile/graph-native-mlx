from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .graph import load_graph
from .models import GraphSpec
from .mlx_native.graph_tables import (
    compile_graph,
    graph_schema_hash,
    normalized_graph_payload,
    write_generated_module,
)

BUNDLE_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
GRAPH_NAME = "graph.yaml"
COMPILED_NAME = "compiled_graph.py"
BENCHMARK_NAME = "benchmark.json"


class GraphBundleError(ValueError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _graph_yaml_bytes(graph: GraphSpec) -> bytes:
    payload = normalized_graph_payload(graph)
    # The loader injects the map key as node.id; omit duplicate IDs from the editable YAML.
    payload["nodes"] = {
        node_id: {key: value for key, value in node.items() if key != "id"}
        for node_id, node in payload["nodes"].items()
    }
    text = yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=False,
        width=100,
    )
    return text.encode("utf-8")


def optimized_graph_version(graph: GraphSpec, *, mutation_path: tuple[str, ...]) -> GraphSpec:
    if not mutation_path:
        return GraphSpec.model_validate(graph.model_dump(by_alias=True))
    payload = graph.model_dump(by_alias=True)
    suffix_source = json.dumps(
        {
            "schema": graph_schema_hash(graph),
            "mutations": list(mutation_path),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    suffix = hashlib.sha256(suffix_source.encode("utf-8")).hexdigest()[:10]
    base_version = str(graph.version).split("+opt.", 1)[0]
    payload["version"] = f"{base_version}+opt.{suffix}"
    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "promotion_kind": "offline-validated-graph-search",
            "mutation_path": list(mutation_path),
        }
    )
    payload["metadata"] = metadata
    return GraphSpec.model_validate(payload)


@dataclass(frozen=True)
class VerifiedGraphBundle:
    root: Path
    graph: GraphSpec
    manifest: dict[str, Any]

    @property
    def identity(self) -> dict[str, str]:
        graph_info = self.manifest["graph"]
        return {
            "name": str(graph_info["name"]),
            "version": str(graph_info["version"]),
            "schema_hash": str(graph_info["schema_hash"]),
            "bundle_sha256": str(self.manifest["bundle_sha256"]),
        }


def _manifest_without_bundle_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    payload.pop("bundle_sha256", None)
    return payload


def _bundle_digest(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _manifest_without_bundle_hash(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def write_graph_bundle(
    *,
    graph: GraphSpec,
    output_dir: str | Path,
    benchmark_report: Mapping[str, Any],
    baseline_reward: float,
    candidate_reward: float,
    mutation_path: tuple[str, ...],
    promotion_status: str,
    optimizer_config: Mapping[str, Any],
) -> VerifiedGraphBundle:
    if promotion_status not in {"promoted", "candidate", "rejected"}:
        raise ValueError("promotion_status must be promoted, candidate, or rejected")
    if promotion_status == "promoted" and not mutation_path:
        raise ValueError("a promoted graph bundle must contain at least one mutation")
    if promotion_status == "promoted" and candidate_reward < baseline_reward:
        raise ValueError("a promoted graph bundle cannot score below its baseline")
    report_status = benchmark_report.get("status")
    if promotion_status == "promoted" and report_status != "promoted":
        raise ValueError("a promoted graph bundle requires promoted benchmark evidence")
    if report_status is not None and report_status != promotion_status:
        raise ValueError("benchmark status does not match promotion_status")
    root = Path(output_dir).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    graph_path = root / GRAPH_NAME
    compiled_path = root / COMPILED_NAME
    benchmark_path = root / BENCHMARK_NAME
    manifest_path = root / MANIFEST_NAME

    _atomic_write(graph_path, _graph_yaml_bytes(graph))
    write_generated_module(compile_graph(graph), compiled_path)
    benchmark_bytes = json.dumps(
        benchmark_report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8") + b"\n"
    _atomic_write(benchmark_path, benchmark_bytes)

    manifest: dict[str, Any] = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "created_at_unix": time.time(),
        "promotion_status": promotion_status,
        "graph": {
            "name": graph.name,
            "version": graph.version,
            "schema_hash": graph_schema_hash(graph),
            "file": GRAPH_NAME,
            "sha256": _sha256_file(graph_path),
        },
        "compiled": {
            "file": COMPILED_NAME,
            "sha256": _sha256_file(compiled_path),
        },
        "benchmark": {
            "file": BENCHMARK_NAME,
            "sha256": _sha256_file(benchmark_path),
            "baseline_reward": float(baseline_reward),
            "candidate_reward": float(candidate_reward),
            "improvement": float(candidate_reward - baseline_reward),
        },
        "optimizer": {
            **dict(optimizer_config),
            "mutation_path": list(mutation_path),
        },
    }
    manifest["bundle_sha256"] = _bundle_digest(manifest)
    _atomic_write(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n",
    )
    return verify_graph_bundle(root)


def _require_regular_file(root: Path, name: str) -> Path:
    if not name or Path(name).name != name:
        raise GraphBundleError(f"invalid bundle file name {name!r}")
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise GraphBundleError(f"bundle file is missing or not a regular file: {name}")
    return path


def verify_graph_bundle(path: str | Path, *, require_promoted: bool = False) -> VerifiedGraphBundle:
    supplied = Path(path).expanduser().resolve(strict=False)
    root = supplied if supplied.is_dir() else supplied.parent
    manifest_path = supplied if supplied.name == MANIFEST_NAME else root / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise GraphBundleError(f"graph bundle manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphBundleError(f"invalid graph bundle manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise GraphBundleError("graph bundle manifest must be a JSON object")
    if int(manifest.get("format_version", -1)) != BUNDLE_FORMAT_VERSION:
        raise GraphBundleError(
            f"unsupported graph bundle format {manifest.get('format_version')!r}"
        )
    status = manifest.get("promotion_status")
    if status not in {"promoted", "candidate", "rejected"}:
        raise GraphBundleError(f"invalid promotion status {status!r}")
    if require_promoted and status != "promoted":
        raise GraphBundleError(f"graph bundle is not promoted; status={status!r}")
    expected_bundle_hash = str(manifest.get("bundle_sha256") or "")
    if not expected_bundle_hash or _bundle_digest(manifest) != expected_bundle_hash:
        raise GraphBundleError("graph bundle manifest digest mismatch")

    graph_info = manifest.get("graph")
    compiled_info = manifest.get("compiled")
    benchmark_info = manifest.get("benchmark")
    optimizer_info = manifest.get("optimizer")
    if not all(isinstance(item, dict) for item in (graph_info, compiled_info, benchmark_info)):
        raise GraphBundleError("bundle manifest is missing graph, compiled, or benchmark metadata")
    if not isinstance(optimizer_info, dict):
        raise GraphBundleError("bundle manifest is missing optimizer metadata")

    expected_names = (
        (graph_info, GRAPH_NAME, "graph"),
        (compiled_info, COMPILED_NAME, "compiled"),
        (benchmark_info, BENCHMARK_NAME, "benchmark"),
    )
    for info, expected_name, label in expected_names:
        if info.get("file") != expected_name:
            raise GraphBundleError(f"{label} bundle file name must be {expected_name!r}")

    graph_path = _require_regular_file(root, str(graph_info.get("file") or ""))
    compiled_path = _require_regular_file(root, str(compiled_info.get("file") or ""))
    benchmark_path = _require_regular_file(root, str(benchmark_info.get("file") or ""))
    for file_path, info, label in (
        (graph_path, graph_info, "graph"),
        (compiled_path, compiled_info, "compiled"),
        (benchmark_path, benchmark_info, "benchmark"),
    ):
        expected = str(info.get("sha256") or "")
        actual = _sha256_file(file_path)
        if not expected or actual != expected:
            raise GraphBundleError(f"{label} file digest mismatch: expected={expected}, actual={actual}")

    graph = load_graph(graph_path)
    if graph.name != graph_info.get("name") or graph.version != graph_info.get("version"):
        raise GraphBundleError("graph identity does not match bundle manifest")
    actual_schema = graph_schema_hash(graph)
    if actual_schema != graph_info.get("schema_hash"):
        raise GraphBundleError("graph schema hash does not match bundle manifest")

    # Reproduce the generated module rather than trusting executable Python from the bundle.
    with tempfile.TemporaryDirectory(prefix="graph-bundle-verify-") as directory:
        expected_compiled = Path(directory) / COMPILED_NAME
        write_generated_module(compile_graph(graph), expected_compiled)
        if _sha256_file(expected_compiled) != _sha256_file(compiled_path):
            raise GraphBundleError("compiled graph module is not reproducible from graph.yaml")

    try:
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphBundleError(f"invalid benchmark report: {exc}") from exc
    if not isinstance(benchmark, dict):
        raise GraphBundleError("benchmark report must be a JSON object")
    if benchmark.get("status") not in {None, status}:
        raise GraphBundleError("benchmark status does not match bundle promotion status")
    validation = benchmark.get("validation")
    if isinstance(validation, dict):
        baseline = validation.get("baseline")
        candidate = validation.get("candidate")
        if isinstance(baseline, dict) and isinstance(candidate, dict):
            try:
                report_baseline = float(baseline["reward"])
                report_candidate = float(candidate["reward"])
            except (KeyError, TypeError, ValueError) as exc:
                raise GraphBundleError("benchmark validation rewards are invalid") from exc
            if not (
                abs(report_baseline - float(benchmark_info["baseline_reward"])) <= 1e-12
                and abs(report_candidate - float(benchmark_info["candidate_reward"])) <= 1e-12
            ):
                raise GraphBundleError("benchmark rewards do not match bundle manifest")
    if status == "promoted":
        gate = benchmark.get("promotion_gate")
        search = benchmark.get("search")
        if not isinstance(gate, dict) or not isinstance(search, dict):
            raise GraphBundleError("promoted bundle is missing promotion evidence")
        winning_path = search.get("winning_mutation_path")
        manifest_mutation_path = optimizer_info.get("mutation_path")
        if (
            not isinstance(winning_path, list)
            or not winning_path
            or winning_path != manifest_mutation_path
        ):
            raise GraphBundleError("promoted bundle mutation evidence is inconsistent")
        if not all(
            bool(gate.get(name))
            for name in ("promotion_allowed", "quality_not_worse", "has_mutation")
        ):
            raise GraphBundleError("promoted bundle did not satisfy its promotion gate")
        try:
            actual = float(gate["actual_improvement"])
            minimum = float(gate["minimum_improvement"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GraphBundleError("promoted bundle improvement evidence is invalid") from exc
        if actual < minimum:
            raise GraphBundleError("promoted bundle is below its minimum improvement gate")
        try:
            manifest_improvement = float(benchmark_info["improvement"])
            manifest_baseline = float(benchmark_info["baseline_reward"])
            manifest_candidate = float(benchmark_info["candidate_reward"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GraphBundleError("bundle reward metadata is invalid") from exc
        if abs(manifest_candidate - manifest_baseline - manifest_improvement) > 1e-12:
            raise GraphBundleError("bundle reward improvement metadata is inconsistent")
        if abs(actual - manifest_improvement) > 1e-12:
            raise GraphBundleError("promotion gate improvement does not match bundle manifest")
    return VerifiedGraphBundle(root=root, graph=graph, manifest=manifest)


def load_graph_source(path: str | Path, *, require_promoted_bundle: bool = False) -> GraphSpec:
    supplied = Path(path).expanduser()
    if supplied.is_dir() or supplied.name == MANIFEST_NAME:
        return verify_graph_bundle(supplied, require_promoted=require_promoted_bundle).graph
    return load_graph(supplied)
