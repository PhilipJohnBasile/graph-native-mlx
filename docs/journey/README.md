# Engineering journey

This directory preserves the project as an engineering record: the hypothesis, each release, the failure that exposed the next boundary, the fix, and the measured result.

## Timeline

### [v0.1.0](releases/v0.1.0.md) — Typed durable graph runtime

- **Change:** Introduced the typed coding graph, SQLite checkpoints, bounded cycles, durable state, and a Ralph-style loop baseline for comparison.
- **Problem:** Open-ended agent loops repeated work and had no explicit state or trustworthy stop boundary.
- **Result:** A runnable 12-node coding graph with durable checkpoints and a measurable loop baseline.

### [v0.2.0](releases/v0.2.0.md) — MLX-native controller foundation

- **Change:** Added direct MLX-LM inference, generated graph tables, hard-masked MLX decisions, feature extraction, and trainable policy sidecars.
- **Problem:** The graph existed outside the model but could not yet use MLX-native representations for control.
- **Result:** The language backbone and graph controller could share a resident MLX execution path while the graph remained authoritative.

### [v0.3.0](releases/v0.3.0.md) — Real repository execution

- **Change:** Added detached Git worktrees, transactional patch application, bounded commands, real tests, evidence-driven repair, and explicit patch promotion.
- **Problem:** Synthetic graph execution did not prove safe behavior against actual repositories and side effects.
- **Result:** Repository changes became isolated, resumable, testable, and separately promotable.

### [v0.4.0](releases/v0.4.0.md) — Qwen hidden-state policy features

- **Change:** Added selected Qwen hidden-state capture, deterministic 5,120-to-256 projection, hidden-aware policy fusion, artifact integrity, and repository trace collection.
- **Problem:** Static rules had no compact representation of what the model internally understood at each graph checkpoint.
- **Result:** Real Qwen representations could inform graph decisions without persisting raw prompts or raw hidden tensors.

### [v0.5.0](releases/v0.5.0.md) — Qualification and graph optimization

- **Change:** Added one-command Mac qualification, constrained MCTS, held-out promotion gates, real graph objectives, and hash-verified graph bundles.
- **Problem:** The project needed reproducible hardware evidence and a safe way to search graph parameters.
- **Result:** Graph candidates could be optimized and promoted only after integrity and held-out validation checks.

### [v0.5.1](releases/v0.5.1.md) — macOS virtual-environment correctness

- **Change:** Preserved the lexical venv Python invocation, added recent-run discovery, concise reports, and a state-preserving Mac upgrader.
- **Problem:** Resolving venv/bin/python to the Homebrew framework binary bypassed venv site-packages and made pytest disappear.
- **Result:** Verifier commands used the exact configured venv interpreter; reporting no longer required manual SQLite queries.

### [v0.5.2](releases/v0.5.2.md) — Deterministic verifier bytecode

- **Change:** Added a fresh external PYTHONPYCACHEPREFIX and disabled user-site imports for every verifier command.
- **Problem:** Same-second, same-size repairs could reuse stale timestamp-based .pyc bytecode.
- **Result:** Every verifier run evaluated current source and the repaired-candidate regression became deterministic.

### [v0.5.3](releases/v0.5.3.md) — Clean failed-run learning semantics

- **Change:** Prevented failed runs from teaching route/edge/stop imitation and added a controlled 16-repository bootstrap corpus.
- **Problem:** Failed executions were still being imitated as if their actions were desirable.
- **Result:** Failures train value and cost only; the first balanced fast/deep/repair/no-change/abort corpus became collectable.

### [v0.5.4](releases/v0.5.4.md) — Observable, fail-fast collection

- **Change:** Fixed lexical venv paths in manifests, added bounded JSON recovery, terminalized collector exceptions, and printed per-task progress.
- **Problem:** False pytest failures, malformed JSON, and swallowed exceptions left stale running rows and apparently frozen terminals.
- **Result:** Tasks 1-8 completed cleanly and collector failures became explicit and resumable.

### [v0.5.5](releases/v0.5.5.md) — Patch-native structured output

- **Change:** Introduced GRAPH_PATCH_V1: compact JSON metadata plus a raw unified diff outside JSON, with one deterministic recovery.
- **Problem:** Deep multi-file diffs were fragile when embedded as escaped multiline JSON strings.
- **Result:** The previously failing deep email-refactor task completed through tests, review, and verified finish.

### [v0.5.6](releases/v0.5.6.md) — Bounded truncation recovery and clean corpus

- **Change:** Disabled Qwen thinking for structured artifacts by default and added one truncation-only continuation for GRAPH_PATCH_V1.
- **Problem:** A deep migration task spent the full 4,096-token allowance before emitting the actual patch envelope.
- **Result:** The complete 16-task corpus passed: 14 verified completions, 2 intentional bounded aborts, 165 policy records, one model fingerprint, and homogeneous 256-dimensional features.

## Raw evidence

Sanitized terminal transcripts are under [`raw/`](raw/). They intentionally preserve failed commands and intermediate results rather than presenting a rewritten success-only history.
