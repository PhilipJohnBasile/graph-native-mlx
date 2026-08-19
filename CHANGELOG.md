# Changelog

## 0.5.6

- Disabled Qwen thinking by default for bounded structured JSON and patch generation through the tokenizer chat-template control, with a compatibility fallback for tokenizers that do not expose `enable_thinking`.
- Added one truncation-only patch-envelope continuation after the existing deterministic recovery. It is permitted only when the recovery reaches the exact generation ceiling and has emitted a recognizable `GRAPH_PATCH_V1` prefix.
- Added safe salvage rules that discard partial metadata, retain only complete raw-diff lines, and never persist or echo the model's unparsed response.
- Added portable regressions for thinking-disabled templates, legacy tokenizer compatibility, truncated metadata continuation, partial-diff continuation, the three-call hard bound, and resume/model-fingerprint compatibility across the formatting toggle.
- Updated Apple-Silicon upgrade and bootstrap scripts to v0.5.6 while preserving the nine completed clean corpus runs and reopening only task 10.

## 0.5.5

- Added `GRAPH_PATCH_V1`, a bounded raw-text patch envelope that keeps multiline unified diffs outside JSON while retaining compact JSON metadata and strict JSON compatibility.
- Added one deterministic patch-envelope recovery attempt at temperature 0.0, with content hashes, marker presence, token counts, and truncation indicators in failures instead of raw model output.
- Routed repository implementation and repair nodes through the patch-specific provider channel, reducing malformed output on deep multi-file changes without adding an open-ended retry loop.
- Added `--retry-collector-errors` so a collector-terminalized provider/operator failure can reopen and resume its uncommitted current node while intentional graph failures remain terminal.
- Made event sequencing monotonic across collector retries, preserving earlier terminal evidence without colliding with resumed node checkpoints.
- Updated the Apple-Silicon bootstrap collector and in-place upgrade helper to v0.5.5 while retaining completed runs, hidden features, worktrees, and patches.

## 0.5.4

- Enforced reward-weighted action imitation during MLX policy training: completed runs may teach route/edge/stop actions, while failed runs train only success-value and cost prediction.
- Added a regression proving failed decisions receive zero action-imitation weight.
- Added a controlled 16-repository Apple-Silicon bootstrap corpus and one-command trace/export collector with route, no-change, success, and bounded-failure diversity.
- Updated the Apple-Silicon in-place upgrade helper to v0.5.4 while preserving environments, pinned model identity, traces, hidden features, worktrees, and patches.

## 0.5.2

- Isolated Python bytecode for every verifier command with a fresh external `PYTHONPYCACHEPREFIX`, preventing same-second, same-size repairs from reusing stale `.pyc` code.
- Disabled user-site imports inside bounded verifier commands for more reproducible repository tests.
- Added a deterministic regression that forces the stale-bytecode condition across a failed candidate and local repair.
- Updated the Apple-Silicon in-place upgrade helper to v0.5.2 while retaining existing environments, traces, hidden features, worktrees, and patches.

## 0.5.1

- Preserved the active virtual-environment Python invocation after executable trust validation, fixing macOS/Homebrew verifier commands that otherwise lost venv site-packages.
- Added `graph-model runs` to list recent run IDs, statuses, paths, routes, and tasks without querying SQLite manually.
- Added `graph-model trace --latest`, `trace --summary`, and `graph-model report` for concise run, hidden-state, verifier, and patch evidence without manual SQLite inspection.
- Added an in-place Apple-Silicon upgrade helper that preserves the virtual environment, `.graph-env`, run database, hidden-state artifacts, worktrees, and patches.
- Added regression coverage for macOS virtual-environment execution and latest-run CLI resolution.

## 0.5.0

- Added `graph-model qualify-mac` for one-command Apple Silicon/Metal, model-load, structured-generation, Qwen hidden-capture, controller, and memory-telemetry evidence.
- Added constrained synchronous and asynchronous MCTS with logical mutation-slot exclusivity and graph-schema evaluation caching.
- Added real-run graph objectives across synthetic or detached-worktree repository cases.
- Added held-out validation checks that reject reused case IDs and exact task/repository identities.
- Added a production mutation envelope that cannot add topology, raise traversal caps, or alter tool/verifier permissions.
- Added edge-priority mutations and node-temperature-aware operators.
- Added promoted graph bundles with YAML, reproducible compiled tables, benchmark evidence, manifest integrity, and runtime promotion enforcement.
- Added portable tests for qualification, optimization, promotion, bundle tampering, mutation constraints, and held-out leakage.

## 0.4.0

- Added Qwen hidden-state policy features, projected artifact storage, gated fusion, hidden-aware training, and repository trace collection.

## 0.3.0

- Added detached Git worktrees, transactional patch application, real tests, evidence-driven repair, and explicit patch promotion.

## 0.2.0

- Added direct MLX-LM inference, generated graph tables, masked MLX decisions, and trainable policy sidecars.

## 0.1.0

- Added the typed durable graph runtime, SQLite checkpoints, bounded repair, and retry-loop comparison baseline.
