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
CORPUS_ROOT="${CORPUS_ROOT:-$HOME/graph-native-mlx-heldout/paired-canary-v2-$STAMP}"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT/.graph-model/evaluation/paired-canary-v2-$STAMP}"
MANIFEST="$EVAL_ROOT/manifest.jsonl"
CASES="$EVAL_ROOT/cases.json"
REPORT_JSON="$EVAL_ROOT/paired-canary-report.json"
REPORT_MD="$EVAL_ROOT/paired-canary-report.md"

ARMS=(static shadow route-only full)

echo "============================================================"
echo " GRAPH-NATIVE MLX CAUSAL PAIRED CANARY v2"
echo "============================================================"
echo
echo "Corpus:"
echo "  $CORPUS_ROOT"
echo
echo "Evaluation:"
echo "  $EVAL_ROOT"

for path in \
  "$PYTHON" \
  "$GRAPH_MODEL" \
  "$PROJECT/.graph-env" \
  "$PROJECT/scripts/generate-paired-policy-canary-v2.py" \
  "$PROJECT/scripts/compare-paired-policy-canary-v2.py" \
  "$WEIGHTS" \
  "$CONFIG"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required path is missing:"
    echo "  $path"
    exit 1
  fi
done

ACTIVE="$(
  pgrep -f \
    'graph-model.*(run|resume|collect-traces|qualify-mac|mlx-doctor)|collect-bootstrap-policy-corpus-mac.sh|run-paired-policy-canary-v2-mac.sh' \
    2>/dev/null || true
)"

# pgrep can report this script's own parent shell. Keep only other processes.
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
  echo
echo "ERROR: another Graph-Native MLX execution is active:"
  ps -p "$(echo "$ACTIVE" | paste -sd, -)" \
    -o pid=,etime=,%cpu=,%mem=,command= 2>/dev/null || true
  echo
echo "Stop it before running the paired canary."
  exit 2
fi

if grep -Eq \
  '^[[:space:]]*export[[:space:]]+GRAPH_MODEL_MLX_POLICY_(WEIGHTS|CONFIG)=' \
  "$PROJECT/.graph-env"; then
  echo "ERROR: .graph-env persistently activates a learned policy."
  echo "The paired experiment requires arm-local activation only."
  exit 1
fi

ACTUAL_WEIGHTS_SHA="$(shasum -a 256 "$WEIGHTS" | awk '{print $1}')"
if [[ "$ACTUAL_WEIGHTS_SHA" != "$EXPECTED_WEIGHTS_SHA" ]]; then
  echo "ERROR: policy weights do not match candidate 855e378570a9."
  echo "Expected: $EXPECTED_WEIGHTS_SHA"
  echo "Actual:   $ACTUAL_WEIGHTS_SHA"
  exit 1
fi

VERSION="$($PYTHON -c 'import graph_model; print(graph_model.__version__)')"
if [[ "$VERSION" != "0.5.7" ]]; then
  echo "ERROR: expected Graph-Native MLX 0.5.7, found $VERSION."
  exit 1
fi

mkdir -p "$EVAL_ROOT"

cd "$PROJECT"
source "$PROJECT/.venv/bin/activate"
source "$PROJECT/.graph-env"

# Fail closed if the sourced environment introduced persistent policy paths.
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
export GRAPH_MODEL_MLX_GENERATION_SEED=42057

echo
echo "============================================================"
echo " GENERATING SHARED HELD-OUT CASES"
echo "============================================================"

"$PYTHON" "$PROJECT/scripts/generate-paired-policy-canary-v2.py" \
  --root "$CORPUS_ROOT" \
  --manifest "$MANIFEST" \
  --cases "$CASES" \
  --python "$PYTHON" \
  --seed 42057

run_arm() {
  local arm="$1"
  local arm_root="$EVAL_ROOT/$arm"
  local db="$arm_root/runs.sqlite3"
  local summary="$arm_root/summary.json"

  mkdir -p \
    "$arm_root/worktrees" \
    "$arm_root/artifacts" \
    "$arm_root/hidden-states"

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
    export GRAPH_MODEL_MLX_GENERATION_SEED=42057

    case "$arm" in
      static)
        unset GRAPH_MODEL_MLX_POLICY_WEIGHTS || true
        unset GRAPH_MODEL_MLX_POLICY_CONFIG || true
        unset GRAPH_MODEL_MLX_POLICY_SCALE || true
        unset GRAPH_MODEL_MLX_ROUTE_POLICY_SCALE || true
        unset GRAPH_MODEL_MLX_TRANSITION_POLICY_SCALE || true
        ;;
      shadow)
        export GRAPH_MODEL_MLX_POLICY_WEIGHTS="$WEIGHTS"
        export GRAPH_MODEL_MLX_POLICY_CONFIG="$CONFIG"
        export GRAPH_MODEL_MLX_ROUTE_POLICY_SCALE=0
        export GRAPH_MODEL_MLX_TRANSITION_POLICY_SCALE=0
        ;;
      route-only)
        export GRAPH_MODEL_MLX_POLICY_WEIGHTS="$WEIGHTS"
        export GRAPH_MODEL_MLX_POLICY_CONFIG="$CONFIG"
        export GRAPH_MODEL_MLX_ROUTE_POLICY_SCALE=1
        export GRAPH_MODEL_MLX_TRANSITION_POLICY_SCALE=0
        ;;
      full)
        export GRAPH_MODEL_MLX_POLICY_WEIGHTS="$WEIGHTS"
        export GRAPH_MODEL_MLX_POLICY_CONFIG="$CONFIG"
        export GRAPH_MODEL_MLX_ROUTE_POLICY_SCALE=1
        export GRAPH_MODEL_MLX_TRANSITION_POLICY_SCALE=1
        ;;
      *)
        echo "ERROR: unknown arm: $arm"
        exit 1
        ;;
    esac

    echo
echo "============================================================"
    echo " ARM: $arm"
    echo "============================================================"

    set +e
    "$GRAPH_MODEL" collect-traces \
      --provider mlx \
      --controller mlx \
      --manifest "$MANIFEST" \
      --db "$db" \
      --workspace-home "$arm_root/worktrees" \
      --artifact-root "$arm_root/artifacts" \
      --output "$summary" \
      --stop-on-error \
      2> >(tee "$arm_root/progress.jsonl" >&2) \
      | tee "$arm_root/command-output.json"
    status=${PIPESTATUS[0]}
    set -e

    printf '%s\n' "$status" > "$arm_root/exit-code.txt"
    if [[ ! -s "$summary" ]]; then
      echo "ERROR: $arm did not write a collection summary."
      exit 1
    fi
    # Intentional bounded failures are successful collection outcomes. The
    # comparator, not the collector exit code, decides the scientific gate.
    if [[ "$status" -ne 0 ]]; then
      echo "WARNING: $arm collector exited with $status; comparator will inspect evidence."
    fi
  )
}

for arm in "${ARMS[@]}"; do
  run_arm "$arm"
done

echo
echo "============================================================"
echo " COMPARING FOUR PAIRED ARMS"
echo "============================================================"

"$PYTHON" "$PROJECT/scripts/compare-paired-policy-canary-v2.py" \
  --cases "$CASES" \
  --root "$EVAL_ROOT" \
  --output-json "$REPORT_JSON" \
  --output-md "$REPORT_MD" \
  --candidate-sha "$EXPECTED_WEIGHTS_SHA"

echo
echo "============================================================"
echo " FINAL GLOBAL-ACTIVATION CHECK"
echo "============================================================"

if grep -Eq \
  '^[[:space:]]*export[[:space:]]+GRAPH_MODEL_MLX_POLICY_(WEIGHTS|CONFIG)=' \
  "$PROJECT/.graph-env"; then
  echo "ERROR: policy activation leaked into .graph-env."
  exit 1
fi

echo "PASS: .graph-env remains policy-free."

printf '%s\n' "$EVAL_ROOT" \
  > "$PROJECT/.graph-model/evaluation/LATEST_PAIRED_CANARY"

echo
echo "Evaluation root:"
echo "  $EVAL_ROOT"
echo
echo "Markdown report:"
echo "  $REPORT_MD"
echo
echo "JSON report:"
echo "  $REPORT_JSON"
echo
echo "A four-case pass authorizes a larger counterbalanced evaluation."
echo "It does not activate or promote the candidate."
