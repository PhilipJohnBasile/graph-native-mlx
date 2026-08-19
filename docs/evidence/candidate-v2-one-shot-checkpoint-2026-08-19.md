# Candidate-v2 one-shot checkpoint — 2026-08-19

## Status

Candidate-v2 implementation has **not** landed yet. The remote branch exists to preserve the orchestration and failure history before another long local run.

- Public repository: `PhilipJohnBasile/graph-native-mlx`
- Base branch: `agent/counterbalanced-policy-eval-v1`
- Candidate branch: `agent/candidate-v2-multichoice`
- Tracking issue: `#3` — candidate v2 trained only on genuine multi-choice graph decisions
- Local public checkout: `/Users/pjb/git/graph-native-mlx`
- Live runtime: `/Users/pjb/graph-native-mlx`
- One-shot state: `/Users/pjb/graph-native-mlx/.graph-model/candidate-v2-one-shot`
- One-shot log: `/Users/pjb/graph-native-mlx/.graph-model/candidate-v2-one-shot/one-shot.log`
- Candidate-v1 baseline: `855e378570a9`
- Candidate-v1 activation: **disabled globally**
- Candidate-v2 activation: **disabled / not yet trained**

## Completed one-shot stages

The local orchestrator successfully completed and checkpointed:

1. **Stage 01 — preflight**
   - candidate-v1 private Release and assets verified;
   - graph schema validated;
   - MLX doctor reported `ready: true` on Apple M5 Max;
   - no learned policy was configured persistently.
2. **Stage 02 — branch preparation**
   - local branch `agent/candidate-v2-multichoice` created from the current counterbalanced-evaluation branch.

The remote candidate branch was subsequently created from public base commit:

`da9ee991456226c57478f47de793de760b3506bb`

## Stage-03 failure history

### Attempt 1 — workspace validator

The original one-shot passed:

```text
--max-context-files 300
```

but `WorkspaceConfig` permits at most `100`. The run stopped before a run row or patch was created.

```text
ValidationError: max_context_files must be <= 100
```

### Attempt 2 — oversized model prompt

After reducing the file count to the valid bound, the generated repository prompt tokenized to:

```text
624,809 tokens
```

against a model context limit of:

```text
262,144 tokens
```

That attempt was stopped. No patch was promoted.

### Attempt 3 — aggregate graph budget

The context-safe attempt used:

```text
60 context files
80,000 bytes per file
400,000 total context bytes
```

The model context warning was eliminated, but the graph run hit the default aggregate execution budget:

```text
budget exceeded: tokens 118393/64000
```

This is the cumulative graph-run budget across plan, implementation, tests, repair, review, and policy-prefill accounting. It is distinct from the model's per-generation context/output limits.

No candidate-v2 source patch was applied to the public branch.

## Current budget-safe orchestrator

The next version preserves the successful context bounds and runs the three implementation passes through `GraphRuntime` with an explicit bounded `Budget` because the current `graph-model run` CLI does not expose aggregate budget flags.

Per implementation pass:

```text
max steps:        32
max LLM calls:    16
max tool calls:   40
max total tokens: 512,000
max active time:  3,600 seconds
```

The orchestrator also:

- force-cleans a failed detached worktree before restarting a pass;
- disables hidden-state capture during source implementation when no learned policy is active;
- retains the 60-file / 400,000-byte context cap;
- remains resumable from existing Stage-01 and Stage-02 markers;
- never merges a PR or writes policy activation to `.graph-env`.

Original restored-script SHA-256:

`f506b193a10be30fdb448652c0472ba44500649ebb5546c395b0c7f09213092e`

Git preservation:

- encoded script: `scripts/candidate-v2/one-shot-orchestrator-budget-safe.sh.gz.b64`
- restore helper: `scripts/candidate-v2/restore-one-shot-orchestrator.sh`
- checksum: `scripts/candidate-v2/ONE_SHOT_SHA256.txt`

Restore locally with:

```bash
bash scripts/candidate-v2/restore-one-shot-orchestrator.sh \
  "$HOME/Downloads/graph-native-mlx-candidate-v2-one-shot-budget-safe.sh"
```

## Safety interpretation

These are orchestration-bound failures, not candidate-v2 research outcomes:

- no candidate-v2 weights exist;
- no candidate-v2 source implementation has been applied;
- no safety or benefit conclusion about candidate v2 can be drawn;
- candidate v1 remains the frozen `safe-but-no-demonstrated-benefit` baseline;
- both policies remain absent from persistent `.graph-env` activation.

## Next action

Run the budget-safe orchestrator. It should resume at Stage 03, clean the failed `candidate-v2-core` worktree/database, and repeat the core implementation pass with the explicit aggregate budget.

Do not mark Issue #3 implementation items complete until verified source commits, tests, corpus evidence, trained candidate identity, canary evidence, statistical evidence, and a private archive all exist.
