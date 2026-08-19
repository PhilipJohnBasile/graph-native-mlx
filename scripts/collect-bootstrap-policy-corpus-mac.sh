#!/usr/bin/env bash
set -euo pipefail

PROJECT="${GRAPH_NATIVE_MLX_HOME:-$HOME/graph-native-mlx}"
PYTHON="$PROJECT/.venv/bin/python"
GRAPH_MODEL="$PROJECT/.venv/bin/graph-model"
ENV_FILE="$PROJECT/.graph-env"
TRAINING_ROOT="${GRAPH_MODEL_BOOTSTRAP_ROOT:-$PROJECT/.graph-model/training/bootstrap-v1}"
CORPUS_ROOT="${GRAPH_MODEL_BOOTSTRAP_CORPUS_ROOT:-$HOME/graph-native-mlx-corpus/bootstrap-v1}"
DB="$TRAINING_ROOT/runs.sqlite3"
MANIFEST="$CORPUS_ROOT/manifest.jsonl"
SUMMARY="$TRAINING_ROOT/trace-summary.json"
DATASET="$TRAINING_ROOT/policy-records.jsonl"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: this collector requires Apple Silicon macOS." >&2
  exit 1
fi
if [[ ! -x "$PYTHON" || ! -x "$GRAPH_MODEL" || ! -f "$ENV_FILE" ]]; then
  echo "ERROR: Graph-Native MLX is not installed at $PROJECT" >&2
  exit 1
fi

cd "$PROJECT"
source "$ENV_FILE"
mkdir -p "$TRAINING_ROOT"

"$PYTHON" "$PROJECT/scripts/bootstrap-policy-corpus.py" \
  --root "$CORPUS_ROOT" \
  --manifest "$MANIFEST" \
  --python "$PYTHON"

# The fixtures are compact; bound generation without altering .graph-env.
export GRAPH_MODEL_MLX_MAX_TOKENS="${GRAPH_MODEL_BOOTSTRAP_MAX_TOKENS:-4096}"

"$GRAPH_MODEL" collect-traces \
  --provider mlx \
  --controller mlx \
  --manifest "$MANIFEST" \
  --db "$DB" \
  --workspace-home "$TRAINING_ROOT/worktrees" \
  --artifact-root "$TRAINING_ROOT/artifacts" \
  --output "$SUMMARY" \
  --resume-existing \
  --retry-collector-errors \
  --stop-on-error \
  --max-context-files 8 \
  --max-context-file-bytes 20000 \
  --max-context-bytes 80000 \
  --max-patch-files 8 \
  --max-patch-bytes 100000

# Include both successful and failed traces. v0.5.6 gives failed actions zero
# imitation weight while retaining their value/cost targets.
"$GRAPH_MODEL" export-mlx-policy \
  --db "$DB" \
  --require-hidden \
  --output "$DATASET"

"$PYTHON" - "$SUMMARY" "$DATASET" "$CORPUS_ROOT/corpus-metadata.json" <<'PY'
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

summary_path, dataset_path, metadata_path = map(Path, sys.argv[1:])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
expected = {item["run_id"]: item for item in metadata["fixtures"]}
actual_results = {item["run_id"]: item for item in summary["results"]}
route_counts = collections.Counter(item["expected_route"] for item in expected.values())
reward_counts = collections.Counter(float(item["reward"]) for item in records)
decision_counts = collections.Counter(item["decision_type"] for item in records)
hidden_sizes = sorted({len(item.get("hidden_features", [])) for item in records})
model_fingerprints = sorted({item.get("model_fingerprint", "") for item in records})
status_mismatches = []
for run_id, item in expected.items():
    actual = actual_results.get(run_id, {}).get("status", "missing")
    if actual != item["expected_terminal"]:
        status_mismatches.append({"run_id": run_id, "expected": item["expected_terminal"], "actual": actual})
report = {
    "status_counts": summary.get("status_counts", {}),
    "expected_route_counts": dict(sorted(route_counts.items())),
    "policy_records": len(records),
    "decision_counts": dict(sorted(decision_counts.items())),
    "reward_counts": {str(key): value for key, value in sorted(reward_counts.items())},
    "hidden_feature_sizes": hidden_sizes,
    "model_fingerprints": model_fingerprints,
    "status_mismatches": status_mismatches,
    "summary": str(summary_path),
    "dataset": str(dataset_path),
}
report_path = summary_path.with_name("bootstrap-readiness.json")
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
print(f"Readiness report: {report_path}")
PY

echo
echo "Bootstrap collection complete."
echo "Trace database: $DB"
echo "Policy dataset: $DATASET"
echo "No policy weights were activated automatically."
