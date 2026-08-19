#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$HOME/graph-native-mlx}"
PYTHON="$PROJECT/.venv/bin/python"
GRAPH_MODEL="$PROJECT/.venv/bin/graph-model"

POLICY_ROOT="$PROJECT/.graph-model/policies/bootstrap-policy-v1"
WEIGHTS="$POLICY_ROOT/graph_policy.safetensors"
CONFIG="$POLICY_ROOT/graph_policy.json"
EXPECTED_WEIGHTS_SHA="855e378570a95fc79b18eea00280c9b6a1c6b3f4091e85bfadcb3da77a8a15fb"

STAMP="$(date +%Y%m%d-%H%M%S)"
CORPUS_ROOT="${CORPUS_ROOT:-$HOME/graph-native-mlx-heldout/counterbalanced-policy-eval-v1-$STAMP}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT/.graph-model/evaluation/counterbalanced-policy-eval-v1-$STAMP}"
STATIC_FIRST_MANIFEST="$EVAL_ROOT/static-first-manifest.jsonl"
FULL_FIRST_MANIFEST="$EVAL_ROOT/full-first-manifest.jsonl"
CASES="$EVAL_ROOT/cases.json"
REPORT_JSON="$EVAL_ROOT/counterbalanced-eval-report.json"
REPORT_MD="$EVAL_ROOT/counterbalanced-eval-report.md"

LATEST_CANARY_POINTER="$PROJECT/.graph-model/evaluation/LATEST_PAIRED_CANARY"
PAIRED_CANARY_ROOT="${PAIRED_CANARY_ROOT:-}"

printf '%s\n' "============================================================"
printf '%s\n' " GRAPH-NATIVE MLX COUNTERBALANCED POLICY EVALUATION v1"
printf '%s\n' "============================================================"
printf '\nCorpus:\n  %s\n\nEvaluation:\n  %s\n' "$CORPUS_ROOT" "$EVAL_ROOT"
printf '%s\n' "Design: 24 repositories × static/full = 48 executions"
printf '%s\n' "Order: 12 static-first cases and 12 full-first cases"

for path in \
  "$PYTHON" \
  "$GRAPH_MODEL" \
  "$PROJECT/.graph-env" \
  "$PROJECT/scripts/generate-counterbalanced-policy-eval-v1.py" \
  "$PROJECT/scripts/compare-counterbalanced-policy-eval-v1.py" \
  "$WEIGHTS" \
  "$CONFIG"; do
  if [[ ! -e "$path" ]]; then
    printf 'ERROR: required path is missing:\n  %s\n' "$path"
    exit 1
  fi
done

if [[ -z "$PAIRED_CANARY_ROOT" ]]; then
  if [[ ! -f "$LATEST_CANARY_POINTER" ]]; then
    printf 'ERROR: paired-canary pointer is missing:\n  %s\n' "$LATEST_CANARY_POINTER"
    printf '%s\n' "Run the v0.5.7 paired canary before this larger evaluation."
    exit 1
  fi
  PAIRED_CANARY_ROOT="$(cat "$LATEST_CANARY_POINTER")"
fi
PAIRED_CANARY_REPORT="$PAIRED_CANARY_ROOT/paired-canary-report.json"
if [[ ! -f "$PAIRED_CANARY_REPORT" ]]; then
  printf 'ERROR: paired-canary report is missing:\n  %s\n' "$PAIRED_CANARY_REPORT"
  exit 1
fi

ACTIVE="$({
  pgrep -f \
    'graph-model.*(run|resume|collect-traces|qualify-mac|mlx-doctor)|collect-bootstrap-policy-corpus-mac.sh|run-paired-policy-canary-v2-mac.sh|run-counterbalanced-policy-eval-v1-mac.sh' \
    2>/dev/null || true
} )"
if [[ -n "$ACTIVE" ]]; then
  FILTERED=""
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    if [[ "$pid" != "$$" && "$pid" != "$PPID" ]]; then
      FILTERED+="$pid"$'\n'
    fi
  done <<< "$ACTIVE"
  ACTIVE="${FILTERED%$'\n'}"
fi
if [[ -n "$ACTIVE" ]]; then
  printf '\nERROR: another Graph-Native MLX execution is active:\n'
  ps -p "$(echo "$ACTIVE" | paste -sd, -)" \
    -o pid=,etime=,%cpu=,%mem=,command= 2>/dev/null || true
  printf '\nStop it before running the counterbalanced evaluation.\n'
  exit 2
fi

if grep -Eq \
  '^[[:space:]]*export[[:space:]]+GRAPH_MODEL_MLX_POLICY_(WEIGHTS|CONFIG)=' \
  "$PROJECT/.graph-env"; then
  printf '%s\n' "ERROR: .graph-env persistently activates a learned policy."
  printf '%s\n' "This evaluation requires arm-local activation only."
  exit 1
fi

ACTUAL_WEIGHTS_SHA="$(shasum -a 256 "$WEIGHTS" | awk '{print $1}')"
if [[ "$ACTUAL_WEIGHTS_SHA" != "$EXPECTED_WEIGHTS_SHA" ]]; then
  printf '%s\n' "ERROR: policy weights do not match candidate 855e378570a9."
  printf 'Expected: %s\nActual:   %s\n' "$EXPECTED_WEIGHTS_SHA" "$ACTUAL_WEIGHTS_SHA"
  exit 1
fi

VERSION="$($PYTHON -c 'import graph_model; print(graph_model.__version__)')"
"$PYTHON" - "$VERSION" <<'PY'
import sys
parts = tuple(int(part) for part in sys.argv[1].split(".")[:3])
if parts < (0, 5, 7):
    raise SystemExit(f"Graph-Native MLX >= 0.5.7 is required; found {sys.argv[1]}")
PY

"$PYTHON" - "$PAIRED_CANARY_REPORT" "$EXPECTED_WEIGHTS_SHA" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2]
checks = {
    "gate_passed": report.get("gate_passed") is True,
    "static_shadow_exact": (report.get("safety_checks") or {}).get("static_shadow_exact") is True,
    "candidate_matches": (report.get("candidate") or {}).get("sha256") == expected,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"paired-canary prerequisite failed: {failed}")
print("Paired-canary prerequisite: PASS")
print("  report:", sys.argv[1])
PY

mkdir -p "$EVAL_ROOT"
cd "$PROJECT"
source "$PROJECT/.venv/bin/activate"
source "$PROJECT/.graph-env"

unset GRAPH_MODEL_MLX_POLICY_WEIGHTS || true
unset GRAPH_MODEL_MLX_POLICY_CONFIG || true
unset GRAPH_MODEL_MLX_POLICY_SCALE || true
unset GRAPH_MODEL_MLX_ROUTE_POLICY_SCALE || true
unset GRAPH_MODEL_MLX_TRANSITION_POLICY_SCALE || true

export GRAPH_MODEL_MLX_CAPTURE_HIDDEN=true
export GRAPH_MODEL_MLX_MAX_TOKENS=4096
export GRAPH_MODEL_MLX_TEMPERATURE=0
export GRAPH_MODEL_MLX_TOP_P=1
export GRAPH_MODEL_MLX_TOP_K=0
export GRAPH_MODEL_MLX_STRUCTURED_THINKING=false
export GRAPH_MODEL_MLX_SKIP_FORCED_POLICY=true
export GRAPH_MODEL_MLX_GENERATION_SEED=73157

printf '\n%s\n' "============================================================"
printf '%s\n' " GENERATING 24 SHARED HELD-OUT REPOSITORIES"
printf '%s\n' "============================================================"

"$PYTHON" "$PROJECT/scripts/generate-counterbalanced-policy-eval-v1.py" \
  --root "$CORPUS_ROOT" \
  --static-first-manifest "$STATIC_FIRST_MANIFEST" \
  --full-first-manifest "$FULL_FIRST_MANIFEST" \
  --cases "$CASES" \
  --python "$PYTHON" \
  --seed 73157

run_block() {
  local sequence="$1"
  local order="$2"
  local arm="$3"
  local manifest="$4"
  local arm_root="$EVAL_ROOT/$arm"
  local db="$arm_root/runs.sqlite3"
  local summary_root="$arm_root/block-summaries"
  local summary="$summary_root/${sequence}-${order}.json"
  local progress="$arm_root/${sequence}-${order}-progress.jsonl"
  local output="$arm_root/${sequence}-${order}-command-output.json"

  mkdir -p \
    "$arm_root/worktrees" \
    "$arm_root/artifacts" \
    "$arm_root/hidden-states" \
    "$summary_root"

  (
    cd "$PROJECT"
    source "$PROJECT/.venv/bin/activate"
    source "$PROJECT/.graph-env"

    export GRAPH_MODEL_MLX_CAPTURE_HIDDEN=true
    export GRAPH_MODEL_MLX_HIDDEN_ROOT="$arm_root/hidden-states"
    export GRAPH_MODEL_MLX_MAX_TOKENS=4096
    export GRAPH_MODEL_MLX_TEMPERATURE=0
    export GRAPH_MODEL_MLX_TOP_P=1
    export GRAPH_MODEL_MLX_TOP_K=0
    export GRAPH_MODEL_MLX_STRUCTURED_THINKING=false
    export GRAPH_MODEL_MLX_SKIP_FORCED_POLICY=true
    export GRAPH_MODEL_MLX_GENERATION_SEED=73157

    case "$arm" in
      static)
        unset GRAPH_MODEL_MLX_POLICY_WEIGHTS || true
        unset GRAPH_MODEL_MLX_POLICY_CONFIG || true
        unset GRAPH_MODEL_MLX_POLICY_SCALE || true
        unset GRAPH_MODEL_MLX_ROUTE_POLICY_SCALE || true
        unset GRAPH_MODEL_MLX_TRANSITION_POLICY_SCALE || true
        ;;
      full)
        export GRAPH_MODEL_MLX_POLICY_WEIGHTS="$WEIGHTS"
        export GRAPH_MODEL_MLX_POLICY_CONFIG="$CONFIG"
        export GRAPH_MODEL_MLX_ROUTE_POLICY_SCALE=1
        export GRAPH_MODEL_MLX_TRANSITION_POLICY_SCALE=1
        ;;
      *)
        printf 'ERROR: unknown arm: %s\n' "$arm"
        exit 1
        ;;
    esac

    printf '\n%s\n' "============================================================"
    printf ' BLOCK %s: %s / %s\n' "$sequence" "$order" "$arm"
    printf '%s\n' "============================================================"

    set +e
    "$GRAPH_MODEL" collect-traces \
      --provider mlx \
      --controller mlx \
      --manifest "$manifest" \
      --db "$db" \
      --workspace-home "$arm_root/worktrees" \
      --artifact-root "$arm_root/artifacts" \
      --output "$summary" \
      --stop-on-error \
      2> >(tee "$progress" >&2) \
      | tee "$output"
    status=${PIPESTATUS[0]}
    set -e

    printf '%s\n' "$status" > "$arm_root/${sequence}-${order}-exit-code.txt"
    if [[ ! -s "$summary" ]]; then
      printf 'ERROR: block %s did not write a summary.\n' "$sequence"
      exit 1
    fi
    if [[ "$status" -ne 0 ]]; then
      printf 'WARNING: block %s collector exited with %s; comparator will inspect evidence.\n' \
        "$sequence" "$status"
    fi
  )
}

# Counterbalanced block order. The first half sees static first; the second
# half sees full first. Separate arm databases retain identical run IDs.
run_block 1 static-first static "$STATIC_FIRST_MANIFEST"
run_block 2 static-first full "$STATIC_FIRST_MANIFEST"
run_block 3 full-first full "$FULL_FIRST_MANIFEST"
run_block 4 full-first static "$FULL_FIRST_MANIFEST"

printf '\n%s\n' "============================================================"
printf '%s\n' " COMPARING 24 COUNTERBALANCED PAIRS"
printf '%s\n' "============================================================"

"$PYTHON" "$PROJECT/scripts/compare-counterbalanced-policy-eval-v1.py" \
  --cases "$CASES" \
  --root "$EVAL_ROOT" \
  --output-json "$REPORT_JSON" \
  --output-md "$REPORT_MD" \
  --candidate-sha "$EXPECTED_WEIGHTS_SHA" \
  --canary-report "$PAIRED_CANARY_REPORT"

printf '\n%s\n' "============================================================"
printf '%s\n' " FINAL GLOBAL-ACTIVATION CHECK"
printf '%s\n' "============================================================"

if grep -Eq \
  '^[[:space:]]*export[[:space:]]+GRAPH_MODEL_MLX_POLICY_(WEIGHTS|CONFIG)=' \
  "$PROJECT/.graph-env"; then
  printf '%s\n' "ERROR: policy activation leaked into .graph-env."
  exit 1
fi
printf '%s\n' "PASS: .graph-env remains policy-free."

printf '%s\n' "$EVAL_ROOT" \
  > "$PROJECT/.graph-model/evaluation/LATEST_COUNTERBALANCED_POLICY_EVAL"

printf '\nEvaluation root:\n  %s\n' "$EVAL_ROOT"
printf '\nMarkdown report:\n  %s\n' "$REPORT_MD"
printf '\nJSON report:\n  %s\n' "$REPORT_JSON"
printf '\n%s\n' "The comparator never activates the candidate."
