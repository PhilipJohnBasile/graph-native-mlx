# Validation Record — v0.3.0

## Portable validation environment

- Linux x86-64
- Python 3.13
- Git-backed temporary repositories
- MLX and MLX-LM intentionally absent

## Completed checks

- 58 portable tests passed
- all source and test modules compiled with `compileall`
- YAML graph validated as 12 nodes, 19 edges, and 2 terminals
- generated graph module reproduced from the YAML source
- graph schema hash verified during packaging
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
- MLX-LM loader-signature compatibility covered
- chat-template fallback and complete-JSON extraction covered
- MLX masks, invalid-transition rejection, policy config binding, and trace export covered

## Hardware validation boundary

This environment cannot execute Apple Metal or load the selected 27B checkpoint. On the target M5 Max, run:

```bash
graph-model mlx-doctor
graph-model mlx-doctor --load-model
```

A successful model load is required before claiming compatibility or performance for the exact model repository, pinned revision, installed MLX/MLX-LM versions, adapter, quantization, and Mac memory configuration.

## Security boundary

The verifier runs trusted repository code with local user permissions. The portable suite validates command, path, timeout, output, mutation, and worktree controls; it does not establish hostile-code isolation. Use a restricted VM or container for untrusted repositories.
