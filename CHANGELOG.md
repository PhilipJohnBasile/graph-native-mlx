# Changelog

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
