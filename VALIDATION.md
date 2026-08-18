# Validation Record — v0.4.0

## Portable validation environment

- Linux x86-64
- Python 3.13
- Git-backed temporary repositories
- Apple MLX, MLX-LM, Metal, and the target Qwen checkpoint intentionally unavailable

## Completed checks

- 85 portable tests passed
- all source and test modules compiled with `compileall`
- source formatting and patch hygiene passed `git diff --check`
- YAML graph validated as 12 nodes, 19 edges, and 2 terminals
- generated graph module reproduced byte-for-byte from the YAML source
- graph schema hash verified as `1536e86a64cbf09a1c0308ab72985a67eba47eb748f3b4f7ba23accc25e7543f`
- mock fast, deep, repair-success, review-repair, and bounded-abort paths passed
- real temporary Git repository repaired through an initial bad patch, failing pytest evidence, one diagnosis, one local repair, and a passing rerun
- cumulative verified patch excluded the failed intermediate edit
- source checkout remained untouched in worktree mode
- patch promotion was hash-checked and idempotent
- worktree cleanup retained the promotable patch artifact
- patch application replay and crash-window recovery passed
- transactional rollback and declared-path enforcement covered
- traversal, sensitive-file, binary, rename, symlink, and mismatched-header rejection covered
- verifier shell-control and mutating-Git rejection covered
- repository `PATH` executable spoofing rejection covered
- test-induced tracked workspace mutation detection covered
- safe run-ID path generation covered
- workspace/artifact-root escape prevention covered
- same-run execution lease covered
- active-time budget semantics covered
- direct MLX provider behavior validated through an injected backend
- model/tokenizer residency, explicit provider close, and post-close rejection covered
- MLX-LM loader-signature compatibility covered
- chat-template fallback and complete-JSON extraction covered
- single-worker affinity for load, generation, hidden extraction, and policy inference covered
- Qwen final-layer and selected-layer hidden extraction covered through injected model structures
- both `model.model` and `model.language_model.model` backbone layouts covered
- layer selectors, pooling modes, deterministic CountSketch projection, and bounded policy prompts covered
- immutable hidden-artifact hashing, metadata verification, missing/tampered artifact rejection, and raw-data non-persistence covered
- hidden LRU cache behavior and zero duplicate-prefill accounting covered
- MLX route/edge/stop masks and invalid-transition rejection covered
- stop/edge representation reuse at one checkpoint and fresh representations after changed state covered
- durable policy-call, policy-prefill-token, and transition-latency accounting covered
- policy config binding to graph schema, model fingerprint, extractor schema, and dimensions covered
- controller startup rejects an incompatible hidden-policy sidecar before inference
- trace collection executes real repository manifests and records hidden-artifact counts
- exported decisions require hash-verified state-specific hidden representations when requested
- mixed feature dimensions, extractor schemas, model fingerprints, and explicit-only/hidden datasets are rejected
- run-level train/validation splitting prevents decisions from one run crossing partitions
- AdamW training, early stopping, best-validation-weight restoration, and deployed-loss recomputation covered
- CLI export executes once and returns structured failure on invalid trace data
- resume identity checks cover provider, graph, controller, policy config, and policy-file fingerprints

## Packaging gates

The release process additionally validates:

- clean source archive extraction
- the complete portable suite from the extracted source archive
- wheel content and package-data inclusion
- isolated wheel installation without dependency resolution
- imported package version
- installed `graph-model --help`
- installed `graph-model validate`
- SHA-256 hashes for the source archive and wheel

## Hardware validation boundary

This environment cannot execute Apple Metal or load the selected Qwen checkpoint. On the target Apple Silicon Mac, run:

```bash
graph-model validate
graph-model mlx-doctor
graph-model mlx-doctor --load-model
```

A successful model load and hidden-state diagnostic are required before claiming compatibility, memory fit, correctness, or performance for the exact model repository, immutable revision, installed MLX/MLX-LM versions, adapter, quantization, and hardware configuration.

## Security boundary

The verifier runs trusted repository code with local user permissions. The portable suite validates command, path, timeout, output, mutation, and worktree controls; it does not establish hostile-code isolation. Use a restricted VM or container for untrusted repositories.
