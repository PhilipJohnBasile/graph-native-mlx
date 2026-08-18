# Graph-Native MLX Model

A graph-controlled coding-agent runtime for Apple Silicon. It combines a resident MLX-LM backbone with a validated workflow graph, hard transition masks, durable checkpoints, isolated Git worktrees, transactional patch application, and real test execution.

The central rule is:

> The graph is the control plane. A loop is only a named, bounded, observable back-edge.

This is a compound model rather than a new foundation-model pretraining run. The language model proposes plans, patches, diagnoses, repairs, and semantic reviews. The runtime owns state, permissions, side effects, verification order, budgets, retries, termination, and recovery.

## v0.3 capabilities

- Direct in-process `mlx_lm` generation with one resident model/tokenizer
- A compiled 12-node, 19-edge coding supergraph
- MLX-native masked route, edge, and stop decisions
- Optional trainable route/edge/stop/value/cost policy heads
- Real Git repository inspection and bounded context selection
- Detached per-run worktrees by default; the source checkout remains untouched
- A dedicated patch-application node separate from model generation
- Strict unified-diff validation and sensitive-path blocking
- Idempotent patch ledgers with interruption recovery
- Transactional rollback if post-application validation fails
- Real verifier commands with no shell, executable allowlists, timeouts, and output limits
- Workspace fingerprints before and after tests to reject test-induced source mutation
- Independent semantic review after deterministic verification
- Two bounded local repair passes; no unbounded “try again” loop
- SQLite checkpoints, append-only traces, and same-run cross-process exclusion
- Active-time budgets that do not expire merely because a run was paused
- Hash-verified patch promotion to a clean source checkout
- Worktree cleanup without deleting patches or audit artifacts
- Optional Restate durable-execution adapter
- Policy-trace export and MLX sidecar training

## Authority boundary

```text
                         MLX language backbone
                 plan / patch / diagnose / repair / review
                                  │
                                  ▼
                         typed JSON node result
                                  │
                                  ▼
                  validated graph + runtime predicates
                                  │
                    optional learned residual logits
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

A learned policy may rank currently valid choices. It cannot add a node, invent an edge, bypass the apply/test/review gates, exceed traversal caps, or grant itself more repair attempts.

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
unzip graph-native-model-mlx-v0.3.0.zip
cd graph-native-model-mlx-v0.3.0

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,mlx]'

pytest
```

The base package and portable tests work on non-MLX systems. The `mlx` extra installs only on Apple Silicon.

## Configure Qwen through MLX-LM

```bash
export GRAPH_MODEL_MLX_MODEL='AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP'
export GRAPH_MODEL_MLX_REVISION='<hugging-face-commit-sha>'
export GRAPH_MODEL_MLX_MAX_TOKENS=8192
export GRAPH_MODEL_MLX_TEMPERATURE=0.1
export GRAPH_MODEL_MLX_TOP_P=1.0
export GRAPH_MODEL_MLX_MIN_P=0.0
export GRAPH_MODEL_MLX_TOP_K=0
```

Pinning the Hub revision is strongly recommended. The repository ID alone does not cryptographically identify the model bytes.

Check the environment before a real run:

```bash
graph-model validate
graph-model mlx-doctor
graph-model mlx-doctor --load-model
```

`--load-model` is the hardware gate for the exact MLX-LM version, model architecture, revision, quantization, adapter, and available Mac memory.

## Run against a real repository

The repository must be a clean Git top-level checkout. Worktree mode is the default and leaves it untouched.

```bash
graph-model run \
  --provider mlx \
  --run-id fix-auth-regression-001 \
  --repo /Users/pjb/git/my-project \
  --task 'Fix the failing authentication regression. Preserve the public API, add or update focused tests, run verification, and report exact evidence.'
```

The runtime automatically prepends `git diff --check` and detects common test commands. Explicit commands can be supplied in order:

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

### Important execution boundary

Verifier commands execute repository code with the current user’s operating-system permissions. Path, command shape, duration, and output are constrained, but v0.3 is **not a hostile-code sandbox**. Use only repositories and test commands you trust.

## Inspect, promote, and clean up

Inspect the full durable trace:

```bash
graph-model trace --run-id fix-auth-regression-001
```

A successful worktree run exports a cumulative `verified.patch`. Apply it to the original checkout only after review:

```bash
graph-model apply-result --run-id fix-auth-regression-001
```

Promotion requires:

- a completed run
- the original source `HEAD` still matching the pinned base commit
- a clean source checkout
- the patch file matching its recorded SHA-256
- the same path and patch policy used during the run

Remove the detached worktree while retaining the database, trace, patch, and operation ledgers:

```bash
graph-model cleanup --run-id fix-auth-regression-001
```

A running or unexported dirty worktree requires an explicit `--force`.

## Resume after interruption

```bash
graph-model run \
  --provider mlx \
  --run-id checkpoint-demo \
  --repo /Users/pjb/git/my-project \
  --stop-after-steps 4 \
  --task 'Implement the requested feature and verify it.'

graph-model resume \
  --provider mlx \
  --run-id checkpoint-demo
```

Resume requires the same graph version, model/provider identity, controller identity, policy configuration, and policy-file fingerprints. A completed patch operation is replayed from its ledger rather than applied twice.

Only one process may execute a given local `run_id` at a time. The POSIX file lease is released automatically if the process crashes.

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

The source checkout is never auto-promoted. Promotion is a separate CLI action.

## Graph and policy compilation

Regenerate immutable graph constants after intentional YAML edits:

```bash
graph-model compile-graph \
  --graph src/graph_model/graphs/coding_supergraph.yaml \
  --output src/graph_model/mlx_native/generated_coding_graph.py
```

The generated module includes stable node/edge IDs, adjacency masks, traversal limits, predicates, terminal masks, and a deterministic schema hash. A stale generated module fails validation.

The optional policy sidecar emits:

```text
route logits:  fast / deep / repair
edge logits:   one per compiled edge
stop logits:   continue / repair / finish / abort
value:         estimated completion probability
cost:          normalized token / latency / tool-call estimates
```

It currently receives 62 explicit features for the default graph:

```text
16 task features
34 execution and repository-state features
12 one-hot node features
```

Activate trained policy weights:

```bash
export GRAPH_MODEL_MLX_POLICY_WEIGHTS="$PWD/models/graph-policy-v1/graph_policy.safetensors"
export GRAPH_MODEL_MLX_POLICY_CONFIG="$PWD/models/graph-policy-v1/graph_policy.json"
export GRAPH_MODEL_MLX_POLICY_SCALE=1.0
```

## Train the graph policy

Collect real traces, preferably with held-out evaluation and human review:

```bash
graph-model export-mlx-policy \
  --success-only \
  --output data/mlx-policy-success.jsonl

graph-model train-mlx-policy \
  --input data/mlx-policy-success.jsonl \
  --output-dir models/graph-policy-v1 \
  --hidden-size 128 \
  --epochs 100 \
  --learning-rate 0.001
```

Behavioral cloning is only the bootstrap. To outperform the initial graph, train on selected graph-search winners, preference pairs, independent evaluator scores, and cost-normalized outcomes—not every successful trace indiscriminately.

## What remains outside MLX

MLX owns model inference and policy tensor decisions. The host runtime must continue to own:

- filesystem and Git mutation
- process execution
- network and GitHub effects
- secrets and permissions
- durable journals and checkpoints
- idempotency and transaction recovery
- human approval
- final transition authority

Moving these responsibilities into a tensor graph would weaken, not strengthen, durability and safety.

## Current limitations

- The policy sidecar uses explicit features rather than Qwen hidden-state fusion.
- No joint text-plus-policy LoRA objective is included yet.
- MTP/speculative decoding is deliberately not wired into control decisions.
- Verifier execution is bounded but not OS-sandboxed.
- Patch proposals are text-only unified diffs; binary, rename, symlink, and submodule edits are blocked.
- The included portable environment cannot execute Apple Metal, so the selected 27B model must still pass `mlx-doctor --load-model` on the M5 Max.

See [REPOSITORY_AGENT.md](REPOSITORY_AGENT.md), [ARCHITECTURE.md](ARCHITECTURE.md), [MLX_NATIVE.md](MLX_NATIVE.md), and [VALIDATION.md](VALIDATION.md) for implementation details.
