# Validation Record — v0.5.6

## Portable validation environment

- Linux x86-64
- Python 3.13
- Git-backed temporary repositories
- Apple MLX, MLX-LM, Metal, and the target Qwen checkpoint intentionally unavailable

## Completed checks

- 118 portable tests passed across release-validation groups
- all source and test modules compiled with `compileall`
- YAML graph validated as 12 nodes, 19 edges, and 2 terminals
- generated default graph module reproduced from the YAML source
- default graph schema hash verified as `1536e86a64cbf09a1c0308ab72985a67eba47eb748f3b4f7ba23accc25e7543f`
- mock fast, deep, repair-success, review-repair, and bounded-abort paths passed
- real temporary Git repository repaired through an initial bad patch, failing pytest evidence, one diagnosis, one local repair, and a passing rerun
- cumulative verified patch excluded the failed intermediate edit
- source checkout remained untouched in detached-worktree mode
- patch promotion was hash-checked and idempotent
- worktree cleanup retained the promotable patch artifact
- patch application replay, interruption recovery, transactional rollback, and declared-path enforcement passed
- traversal, sensitive-file, binary, rename, symlink, submodule, and mismatched-header rejection passed
- verifier shell-control, mutating-Git, repository-executable spoofing, timeout, output, and tracked-mutation controls passed
- active virtual-environment Python invocation is preserved after canonical executable validation, including the macOS/Homebrew framework-interpreter case
- verifier Python bytecode is redirected to a fresh external cache per command; a forced same-mtime, same-size stale-`.pyc` repair regression passed
- recent-run listing, latest-run trace resolution, and concise run/hidden-policy/verification reporting passed
- safe run-ID paths, workspace/artifact-root confinement, same-run leases, and active-time budget semantics passed
- direct MLX provider behavior validated through an injected backend
- model/tokenizer residency, explicit close, post-close rejection, loader-signature compatibility, chat-template fallback, and complete-JSON extraction passed
- `GRAPH_PATCH_V1` parsing passed for raw multi-file unified diffs, compact JSON metadata, strict-JSON provider compatibility, marker-order enforcement, and conflicting no-change rejection
- direct MLX patch generation passed through an injected backend without JSON-escaping the diff, including exactly one deterministic envelope-recovery attempt at temperature 0.0
- structured MLX calls pass `enable_thinking=false` to compatible chat templates by default and fall back safely for legacy tokenizers that reject the optional keyword
- a recovery response that reaches the exact generation ceiling may receive exactly one patch-envelope continuation; partial metadata is discarded and partial diffs retain only complete lines
- continuation remains hard-bounded to a third and final generation call and is unavailable when no recognizable `GRAPH_PATCH_V1` prefix exists
- malformed patch failures expose bounded hashes, lengths, marker presence, and truncation signals without persisting or echoing raw model output
- a real two-file repository change passed patch-native proposal, transactional apply, pytest verification, semantic review, verified patch export, and source-checkout isolation
- single-worker affinity for model load, generation, hidden extraction, and policy inference passed
- Qwen final-layer and selected-layer extraction passed through injected model structures
- both `model.model` and `model.language_model.model` backbone layouts passed
- layer selectors, pooling modes, deterministic CountSketch projection, bounded policy prompts, immutable hidden artifacts, metadata verification, cache behavior, and raw-data non-persistence passed
- MLX route/edge/stop masks and invalid-transition rejection passed
- durable policy-call, policy-prefill-token, and transition-latency accounting passed
- policy binding to graph schema, model fingerprint, extractor schema, dimensions, configuration, and file hashes passed
- repository trace collection, hidden-required export, homogeneous-dataset enforcement, run-level splitting, AdamW training, early stopping, and best-weight restoration passed
- collector/provider exceptions are terminalized with explicit evidence instead of leaving stale `running` rows
- `--retry-collector-errors` reopens only collector-terminalized failures at their uncommitted current node; intentional graph failures remain terminal and completed upstream nodes are not repeated
- append-only event sequencing remains monotonic when a terminalized collection run is reopened and resumed
- failed-run route/edge/stop imitation is reward-masked to zero while failed runs remain active for value and cost learning
- controlled bootstrap corpus generation produced 16 clean Git repositories spanning fast, deep, repair, no-change, completed, and bounded-failure cases without embedding expected labels in the worktree
- bootstrap collector shell syntax and dedicated trace/export paths passed portable validation
- Graph's native MLX generation bridge validates `ffn_mode=mlx-resident`, passes
  `MLXC_USE_DEVICE_FFN=1` only for the opt-in path, and rejects receipts that
  omit or misreport the mode; host mode remains the default
- Graph's native MLX generation bridge validates
  `attention_projection_mode=mlx-resident-query-sdpa-v1`, passes
  `MLXC_USE_SDPA_RESIDENT=1` only when SDPA and native RoPE are both selected,
  and rejects missing or mismatched receipt evidence; host projection remains
  the default
- Graph's native MLX generation bridge validates
  `hidden_state_mode=mlx-resident-hidden-v1`, passes
  `MLXC_USE_DEVICE_RESIDUAL=1` only with resident attention and FFN, and rejects
  missing, mismatched, or prerequisite-free hidden-state requests; host hidden
  state remains the default
- Graph's native MLX generation bridge validates
  `quantization_mode=mlx-affine-q4-v1`, passes `MLXC_USE_QUANTIZED=1` only for
  the opt-in path, and rejects missing or mismatched quantization evidence; host
  F32 weights remain the default

## v0.5.6 qualification checks

- the Mac qualification workflow was executed through injected provider, diagnostics, hidden-state, and controller backends
- platform/configuration, model-load, structured-generation, hidden-capture, hard-masked route/stop/edge control, and provider-close stages passed
- JSON and Markdown evidence artifacts were written and parsed
- provider shutdown after qualification passed
- qualification security metadata confirms raw hidden tensors and raw policy prompts are not persisted
- actual Apple Silicon and Metal execution remains a separate hardware gate

## v0.5 graph-optimization checks

- synchronous and asynchronous constrained MCTS compiled and executed
- contradictory mutations for one logical slot cannot be stacked in one path
- duplicate graph schemas are evaluated once
- real graph executions feed the optimization objective
- LLM-node temperature changes affect actual operator calls
- non-LLM configuration mutation is rejected
- edge traversal limits may be preserved or lowered but not increased
- edge priorities are restricted to existing edges and a bounded range
- training/validation overlap by case ID is rejected
- exact duplicated task/repository identity across training and held-out validation is rejected
- a cost-improving candidate with unchanged expected outcomes passed independent validation and was promoted
- an unchanged base graph remained a non-promoted candidate
- promoted graph bundle verification passed
- non-promoted bundle loading with `--require-promoted` failed as intended
- graph-file tampering was detected
- manifest, graph, compiled-table, benchmark, reward, mutation-path, and promotion-gate consistency checks passed
- compiled graph tables were regenerated from `graph.yaml` and compared byte-for-byte
- CLI `optimize-graph` and `verify-graph-bundle` completed against portable mock task sets

## Packaging gates

The release process additionally validates:

- clean source archive extraction
- the complete portable suite from the extracted source archive
- wheel content and package-data inclusion
- isolated wheel installation without dependency resolution
- imported package version
- installed `graph-model --help`
- installed `graph-model validate`
- installed graph optimization and bundle verification commands
- installed `graph-model runs`, `graph-model report`, and `graph-model trace --latest --summary` run-discovery and concise-evidence commands
- release-wide source and artifact hygiene scan
- SHA-256 hashes for the source archive and wheel

## Hardware validation boundary

This environment cannot execute Apple Metal or load the selected Qwen checkpoint. On the target Apple Silicon Mac, run:

```bash
graph-model validate
graph-model mlx-doctor
graph-model qualify-mac --output-dir .graph-model/qualification
```

A passing qualification report is required before claiming compatibility, memory fit, correctness, or performance for the exact model repository, immutable revision, installed MLX/MLX-LM versions, adapter, quantization, and hardware configuration.

MLX memory telemetry is best-effort runtime evidence and should not be treated as a complete process-footprint measurement without independent operating-system observation.

## Security boundary

The verifier runs trusted repository code with local user permissions. The portable suite validates command, path, timeout, output, mutation, worktree, promotion, and graph-search controls; it does not establish hostile-code isolation. Use a restricted VM or container for untrusted repositories.
