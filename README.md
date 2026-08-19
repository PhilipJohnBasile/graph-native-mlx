# Graph-Native MLX Model

A graph-controlled coding-agent runtime for Apple Silicon. It combines a resident MLX-LM language backbone with a validated workflow graph, hard transition masks, state-aware Qwen representations, durable checkpoints, isolated Git worktrees, transactional patch application, real verifier execution, and held-out graph optimization.

The central rule is:

> The graph is the control plane. A loop is only a named, bounded, observable back-edge.

This is a compound model rather than a foundation-model pretraining run. The language model proposes plans, patches, diagnoses, repairs, and semantic reviews. The runtime owns state, permissions, effects, verification order, budgets, retries, termination, recovery, and promotion.

## v0.5.5 capabilities

### MLX-native model and policy path

- Direct in-process `mlx_lm` generation with one resident model and tokenizer
- One dedicated affinity worker for model load, generation, hidden extraction, policy inference, and masked selection
- Qwen hidden-state extraction from the current checkpointed graph state
- Configurable layer selectors: `final`, integer indices, negative indices, and percentages
- `last-token`, `mean`, and `mean-last` pooling
- Stable fixed-size CountSketch projection, 256 dimensions by default
- Gated residual fusion of 62 explicit graph features with projected Qwen features
- Optional route, edge, stop, success-value, and cost policy heads
- Model, graph, extractor, configuration, and policy-weight identity checks
- Hash-addressed projected-feature artifacts; raw prompts and raw hidden tensors are not persisted

### Hard graph and durable repository execution

- A compiled 12-node, 19-edge coding supergraph
- Hard-masked route, edge, and stop decisions
- Predicate evaluation and traversal caps outside the language model
- Two bounded local repair passes; no unbounded retry loop
- Detached per-run Git worktrees by default
- `GRAPH_PATCH_V1` raw patch envelopes keep multiline diffs outside JSON, with one bounded deterministic recovery and strict JSON compatibility
- Strict unified-diff validation and sensitive-path blocking
- Transactional patch application and rollback
- Idempotent effect ledgers with interruption recovery
- Real verifier commands with no shell, executable allowlists, timeouts, and output limits
- Fresh external Python bytecode cache per verifier command, preventing stale `.pyc` reuse across rapid same-size repairs
- Tracked-workspace fingerprints before and after tests
- Independent semantic review after deterministic verification
- SQLite checkpoints, append-only traces, active-time budgets, and same-run process exclusion
- Hash-verified patch export and separate explicit promotion to the source checkout
- Optional Restate durable-execution adapter

### v0.5 qualification and graph optimization

- One-command Mac qualification for platform configuration, model loading, structured generation, Qwen hidden capture, MLX policy control, and memory telemetry
- Constrained asynchronous MCTS over validated graph parameters
- Real graph executions as the optimizer objective, including repository worktree cases
- Independent held-out validation before promotion
- Training/validation duplicate detection by case ID and exact task identity
- A hard mutation envelope:
  - tune temperature only on existing LLM nodes;
  - preserve or lower existing edge traversal caps;
  - reprioritize existing edges within a bounded range;
  - never add nodes, edges, effects, commands, or retries
- Hash-verified graph bundles containing editable YAML, reproduced compiled tables, benchmark evidence, and a manifest
- Runtime loading of bundle directories only when the bundle is marked `promoted` and passes full verification

## Authority boundary

```text
             current task + checkpointed graph state + verifier evidence
                                      │
                                      ▼
                              Qwen MLX backbone
                                      │
                   selected hidden views, pooled and projected
                                      │
                                      ▼
62 explicit graph features ─► gated policy fusion ─► residual logits/value/cost
                                      │
                                      ▼
                  validated graph + predicates + traversal caps
                                      │
                                      ▼
                         hard MLX transition mask
                                      │
                                      ▼
                       selected valid graph edge
                                      │
                                      ▼
          durable host runtime: Git, tests, checkpoints, permissions
```

A learned policy may rank currently valid choices. It cannot add a node, invent an edge, bypass the apply/test/review gates, exceed traversal caps, grant itself additional repair attempts, or commit an external effect.

## Default repository graph

```text
intake/router
      │
      ▼
   context ───── creates or resumes detached worktree
   ┌──┴─────────────────────┐
   │ fast                    │ deep / repair
   ▼                         ▼
implement                  plan ─► plan_check
   │                         │          │
   └──────────────┬──────────┘          └─ one bounded revision
                  ▼
            apply_candidate  ◄──────────────────────┐
                  │                                 │
          pass ───┴─── fail                         │
           │            │                           │
           ▼            ▼                           │
         tests       diagnose ─► repair proposal ───┘
           │
     pass ─┴─ fail ─► diagnose / repair
           │
           ▼
         review
     pass ─┴─ fail ─► diagnose / repair
           │
           ▼
         finish

exhausted bounded paths ─────────────────────────► abort
```

The model never applies its own patch. It returns a unified diff. The host validates and applies it under a separate idempotent operation.

## Install on an M5 Max

```bash
unzip graph-native-model-mlx-v0.5.5.zip
cd graph-native-model-mlx-v0.5.5

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,mlx]'

python -m pytest -q
```

The base package and portable tests work on non-MLX systems. The `mlx` extra is selected only on Apple Silicon.

## Upgrade an existing installation on the M5 Max

The release includes a state-preserving upgrade helper. It keeps the existing virtual environment, `.graph-env`, `.graph-model` database, hidden-state artifacts, detached worktrees, and verified patches, while backing up the prior source tree.

After extracting the v0.5.5 source archive:

```bash
cd graph-native-model-mlx-v0.5.5
./scripts/upgrade-mac-in-place.sh
```

The helper reinstalls the editable package, runs the complete portable suite, validates the graph, displays retained runs, writes a concise report for the latest completed execution, and executes `qualify-mac` against the exact configured MLX model.

## Configure the MLX model

```bash
export GRAPH_MODEL_MLX_MODEL='/absolute/path/to/an-MLX-LM-model'
# For a Hub repository, pin the immutable revision:
export GRAPH_MODEL_MLX_REVISION='<full-commit-sha>'

export GRAPH_MODEL_MLX_MAX_TOKENS=8192
export GRAPH_MODEL_MLX_TEMPERATURE=0.1
export GRAPH_MODEL_MLX_TOP_P=1.0
export GRAPH_MODEL_MLX_MIN_P=0.0
export GRAPH_MODEL_MLX_TOP_K=0
```

Enable state-aware hidden capture:

```bash
export GRAPH_MODEL_MLX_CAPTURE_HIDDEN=1
export GRAPH_MODEL_MLX_HIDDEN_ROOT="$PWD/.graph-model/hidden-states"
export GRAPH_MODEL_MLX_HIDDEN_FEATURE_SIZE=256
export GRAPH_MODEL_MLX_HIDDEN_MAX_INPUT_TOKENS=2048
export GRAPH_MODEL_MLX_HIDDEN_CACHE_ENTRIES=1024
export GRAPH_MODEL_MLX_POLICY_LAYERS='final'
export GRAPH_MODEL_MLX_POLICY_POOLING='last-token'
export GRAPH_MODEL_MLX_HIDDEN_PROJECTION_SEED=47261993
```

Using only `final` is the least expensive capture path. Multiple selected layers produce richer policy features but require additional decoder work and memory traffic.

## Run the M5 qualification gate

Basic diagnostics remain available:

```bash
graph-model validate
graph-model mlx-doctor
graph-model mlx-doctor --load-model
```

v0.5.5 includes one evidence-producing qualification command:

```bash
graph-model qualify-mac \
  --output-dir .graph-model/qualification
```

It writes:

```text
.graph-model/qualification/mlx-m5-qualification.json
.graph-model/qualification/mlx-m5-qualification.md
```

The stages cover:

1. Apple Silicon, MLX, MLX-LM, Metal, and configuration inspection
2. Exact configured model/revision load
3. Structured JSON generation
4. Projected Qwen hidden-state capture
5. Hard-masked MLX route, stop, and edge selection
6. Best-effort active, peak, and cache memory telemetry when exposed by the installed MLX version
7. Clean provider shutdown

A passing report is evidence for the exact local model, revision, adapter, quantization, MLX/MLX-LM versions, and machine. It is not a general performance claim for other configurations.

## Run against a real repository

The repository must be a clean Git top-level checkout. Worktree mode is the default and leaves it untouched.

```bash
graph-model run \
  --provider mlx \
  --run-id fix-auth-regression-001 \
  --repo /Users/pjb/git/my-project \
  --task 'Fix the failing authentication regression. Preserve the public API, add focused tests, run verification, and report exact evidence.'
```

Explicit verifier commands can be supplied in order:

```bash
graph-model run \
  --provider mlx \
  --run-id fix-auth-regression-002 \
  --repo /Users/pjb/git/my-project \
  --test-command 'python3 -m pytest -q' \
  --test-command 'python3 -m mypy src' \
  --task 'Fix the authentication regression and preserve typing guarantees.'
```

Commands are parsed with `shlex`, executed without a shell, resolved through a sanitized `PATH`, restricted to an allowlist, bounded by time and output limits, and stopped on the first failure.

When an absolute Python command is the active virtual-environment interpreter, the runtime validates its canonical target but preserves the venv invocation path. This is required on macOS/Homebrew so `pyvenv.cfg` and environment site-packages remain active during verification.

### Execution boundary

Verifier commands execute repository code with the current user’s operating-system permissions. Path, command shape, duration, output, and tracked mutations are constrained, but v0.5.5 is not a hostile-code sandbox. Use only repositories and test commands you trust, or run the project inside a restricted VM/container.

## Inspect, promote, clean up, and resume

List recent executions without querying SQLite manually:

```bash
graph-model runs --limit 20
graph-model runs --status completed
```

Show the latest completed run as a concise evidence report:

```bash
graph-model report --latest --latest-status completed
# Equivalent summary through the trace command:
graph-model trace --latest --latest-status completed --summary
```

The concise report includes the path, token and call metrics, unique projected Qwen artifacts, decision sources, verifier commands and verdicts, semantic review, and verified-patch identity. Full append-only events remain available when needed:

```bash
graph-model trace --run-id fix-auth-regression-001
```

Promote or clean up an identified run explicitly:

```bash
graph-model apply-result --run-id fix-auth-regression-001
graph-model cleanup --run-id fix-auth-regression-001
```

Patch promotion requires a completed run, a clean source checkout still at the pinned base commit, and a verified patch whose SHA-256 matches the trace.

Interrupted runs resume from the next unfinished checkpoint:

```bash
graph-model resume \
  --provider mlx \
  --run-id fix-auth-regression-001
```

Resume requires the same graph version, provider identity, controller identity, policy configuration, and policy-file fingerprints.

## Bootstrap the policy-data pipeline

The v0.5.5 source archive includes a controlled 16-repository bootstrap corpus spanning fast, deep, repair, no-change, successful, and bounded-failure executions. It uses a dedicated trace database and never activates weights automatically:

```bash
scripts/collect-bootstrap-policy-corpus-mac.sh
```

Use this only to qualify the collection/export pipeline. Real policy training should add substantially more varied real-repository traces and held-out tasks.

## Collect state-aware policy traces

Create a JSONL manifest. Each line contains one repository task:

```json
{"run_id":"project-fix-001","repo":"/Users/pjb/git/project","task":"Fix the failing parser regression and verify it.","test_commands":["python3 -m pytest -q"],"tags":["parser","repair"]}
```

Run the manifest with hidden capture enabled:

```bash
graph-model collect-traces \
  --provider mlx \
  --manifest examples/repository_trace_tasks.example.jsonl \
  --db .graph-model/traces.sqlite3 \
  --workspace-home .graph-model/worktrees \
  --artifact-root .graph-model/artifacts \
  --output .graph-model/trace-summary.json
```

The collector records per-run status, graph path, generation and policy-prefill token costs, policy/tool calls, and distinct hidden artifacts. Existing completed runs are summarized rather than re-executed; resumable runs require `--resume-existing`. Failures specifically terminalized by the collector can be retried at their uncommitted current node with `--retry-collector-errors`; intentional graph aborts remain terminal.

## Export and train the fused policy

Export only records with hash-verified Qwen features:

```bash
graph-model export-mlx-policy \
  --db .graph-model/traces.sqlite3 \
  --success-only \
  --require-hidden \
  --output data/mlx-policy-hidden.jsonl
```

Train the gated route/edge/stop/value/cost sidecar:

```bash
graph-model train-mlx-policy \
  --input data/mlx-policy-hidden.jsonl \
  --output-dir models/graph-policy-v2 \
  --hidden-size 128 \
  --epochs 100 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --validation-fraction 0.15 \
  --patience 15 \
  --require-hidden
```

The split is by `run_id`, preventing decisions from one execution from appearing in both partitions. Hidden-feature datasets must be homogeneous in projected dimension, extractor schema, and model fingerprint.

Activate trained weights:

```bash
export GRAPH_MODEL_MLX_POLICY_WEIGHTS="$PWD/models/graph-policy-v2/graph_policy.safetensors"
export GRAPH_MODEL_MLX_POLICY_CONFIG="$PWD/models/graph-policy-v2/graph_policy.json"
export GRAPH_MODEL_MLX_POLICY_SCALE=1.0
```

A hidden-fusion sidecar refuses to load against a different graph schema, Qwen model fingerprint, hidden extractor schema, or feature dimension.

Behavioral cloning is only the bootstrap. Stronger policies should use held-out evaluator scores, failed alternatives, graph-search winners, and cost-aware preference data rather than treating every completed run as equally good.

## Optimize and promote a graph

Search tasks and held-out validation tasks are separate JSONL files. Reusing a case ID or an exact task/repository identity across the two sets is rejected.

```bash
graph-model optimize-graph \
  --provider mlx \
  --controller mlx \
  --input examples/benchmark_tasks.jsonl \
  --validation-input examples/graph_search_validation_tasks.example.jsonl \
  --mutations examples/graph_mutations.example.json \
  --iterations 24 \
  --min-validation-improvement 0.001 \
  --output-dir .graph-model/graphs/coding-optimized-v1
```

The optimizer evaluates the base graph and constrained variants using ordinary durable graph executions. A candidate is promoted only when:

- held-out validation is present, unless in-sample promotion is explicitly enabled;
- expected outcome quality does not decline;
- the configured reward improvement gate is met;
- at least one permitted mutation is present.

Verify the resulting bundle:

```bash
graph-model verify-graph-bundle \
  --bundle .graph-model/graphs/coding-optimized-v1 \
  --require-promoted
```

Use a promoted bundle by passing the directory or its manifest:

```bash
graph-model run \
  --graph .graph-model/graphs/coding-optimized-v1 \
  --provider mlx \
  --run-id optimized-graph-run-001 \
  --repo /Users/pjb/git/my-project \
  --task 'Fix the failing regression and provide exact verification evidence.'
```

Passing a bundle directory or `manifest.json` requires promoted status. A non-promoted candidate can be inspected explicitly by passing its `graph.yaml`, but that bypass is visible and does not represent promotion.

Each bundle contains:

```text
manifest.json          integrity, identity, optimizer, and promotion metadata
graph.yaml             editable validated graph
compiled_graph.py      generated immutable table representation
benchmark.json         training and held-out evaluation evidence
optimization-summary.json
```

Verification regenerates `compiled_graph.py` from `graph.yaml` and compares hashes instead of importing executable Python from the bundle. Bundle hashes provide integrity and reproducibility, not third-party signing or remote attestation.

## Repository safety controls

The default repository boundary rejects:

- absolute paths and `..` traversal
- `.git`, `.graph-model`, `.ssh`, secret-key, credential, and `.env` targets
- binary patches, submodules, symlink changes, renames, and copies
- mismatches between `diff --git`, `---`, and `+++` paths
- patches above the configured byte or file limit
- undeclared changed paths after application
- shell operators and command substitution
- mutating Git commands in verifier configuration
- executables resolved from the repository through a spoofed `PATH`
- test runs that change the tracked workspace fingerprint

The source checkout is never auto-promoted. Patch promotion and graph promotion are separate explicit boundaries.

## What remains outside MLX

MLX owns model inference, hidden extraction, feature fusion, and policy tensor decisions. The host runtime continues to own:

- filesystem and Git mutation
- process execution
- network and GitHub effects
- secrets and permissions
- durable journals and checkpoints
- idempotency and transaction recovery
- human approval
- final transition authority

Moving these responsibilities into a tensor graph would weaken durability and safety.

## Current limitations

- The selected hidden-state forward path and `qualify-mac` workflow must still be executed against the exact target Qwen/MLX-LM checkpoint on Apple Silicon.
- Joint language-model plus graph-policy LoRA training is not included yet.
- MTP/speculative decoding is deliberately not wired into control decisions.
- Multi-layer hidden capture is not supported with MLX-LM pipeline parallelism; use `final` in that configuration.
- Verifier execution is bounded but not OS-sandboxed.
- Patch proposals are text-only unified diffs; binary, rename, symlink, and submodule edits are blocked.
- Graph optimization changes only approved parameters within the existing topology; runtime-generated arbitrary graphs are intentionally unsupported.
- Validated inference-time subgraph grafting is not implemented.

See [REPOSITORY_AGENT.md](REPOSITORY_AGENT.md), [ARCHITECTURE.md](ARCHITECTURE.md), [MLX_NATIVE.md](MLX_NATIVE.md), and [VALIDATION.md](VALIDATION.md).
