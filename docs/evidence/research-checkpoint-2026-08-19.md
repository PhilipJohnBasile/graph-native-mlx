# Research checkpoint — 2026-08-19

This file is a durable breadcrumb for the Graph-Native MLX policy-controller work while the long statistical evaluation is still running.

## Stable identities

- Public repository: `PhilipJohnBasile/graph-native-mlx`
- Live runtime: `/Users/pjb/graph-native-mlx`
- Private backup repository: `PhilipJohnBasile/graph-native-mlx-private-backup`
- Evaluation branch: `agent/counterbalanced-policy-eval-v1`
- Candidate policy: `855e378570a9`
- Full candidate weights SHA-256: `855e378570a95fc79b18eea00280c9b6a1c6b3f4091e85bfadcb3da77a8a15fb`
- Graph schema: `1536e86a64cbf09a1c0308ab72985a67eba47eb748f3b4f7ba23accc25e7543f`
- Hidden-state schema: `201c84fb8c88e0c6d8b12d472ed10b2688101a35344dbdf15bf966dd6be78e06`
- Model fingerprint: `16e1d71f453b0db739c663ca66b36f3fe39a53410be4e4d86ccf10a1fcc42ef8`
- Model: `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP`

The candidate remains globally disabled. `.graph-env` is not to contain persistent policy activation.

## Completed milestones

### v0.5.7 causal paired evaluation

PR #1 merged to `main` at `f7d744a43b3c76afcaae5ca6486d8c94be7f6b9c`.

The four-arm paired canary passed:

- static: 4/4 expected outcomes
- shadow: 4/4
- route-only: 4/4
- full: 4/4
- false successes: 0 in every arm
- static/shadow exact equivalence: true
- candidate remained globally disabled

This established that loading the policy in shadow mode did not itself perturb execution.

### 24-case counterbalanced study

The larger held-out study ran 24 repositories in static and full-policy arms, 48 executions total.

Observed outcome-level result:

- static: 24/24 correct
- full: 24/24 correct
- false successes: 0 in both arms
- route changes: 0
- meaningful graph-choice changes: 0
- no material cost improvement

The initial comparator rejected the run only because one same-state prompt-equivalence check failed on `deep-date-parser`.

### Prompt-divergence forensics

For `deep-date-parser`, the first substantive divergence showed:

- same system prompt hash at the compared repair point
- same canonical repair prompt hash for one repair revision
- same generation seed for that repair revision
- different model output afterward

That proved the difference could not be attributed to a learned route or transition choice.

A targeted four-execution replay then ran:

- `full-primary`
- `static-primary`
- `static-replicate`
- `full-replicate`

All four completed and verified the same final patch, but intermediate prompt/model artifacts were not bitwise identical even across repeated runs of the same arm. The attempted greedy-decoding source patch was automatically rolled back when the exact-output gate failed.

Conclusion: exact intermediate-generation equality is too strong to use as a safety gate for this MLX/Qwen stack. Outcome-level verification must be evaluated statistically across repeated trials.

## Current experiment: statistical policy evaluation v1

Evaluation root:

`/Users/pjb/graph-native-mlx/.graph-model/evaluation/statistical-policy-eval-v1-20260819-152952`

Design:

- 12 representative held-out cases
- 5 repetitions per arm
- static and full-policy arms
- 120 repository executions total
- sampling temperature `0.1`
- candidate remains globally disabled

Selected cases:

- fast: `fast-env-flag`, `fast-extension`, `fast-mask-token`
- deep: `deep-date-parser`, `deep-page-limit`, `deep-permission-policy`
- repair: `repair-cache-ttl`, `repair-chunks`, `repair-csv-fields`
- no-change: `nochange-safe-divide`
- impossible: `impossible-default-region`, `impossible-mode-label`

### Partial checkpoint from the running evaluation

At the captured checkpoint, 84/120 executions had completed (70%):

- repetition 1: static + full complete
- repetition 2: static + full complete
- repetition 3: static + full complete
- repetition 4: full complete
- repetition 4: static started

So far the ordinary tasks continue to complete and the two impossible contracts continue to terminate through the intended bounded abort rather than false success. This is an in-progress observation only; no final statistical conclusion should be drawn until all 120 runs and the final report are complete.

The full-policy arm is loading candidate `855e378570a9`; the static arm reports `hardcoded-priors-only`, confirming that the experiment is comparing the intended controller configurations.

## Current methodological rule

Do not use byte-identical model generations as the decisive comparison for stochastic local inference. Use repeated, counterbalanced trials and compare distributions of:

- mechanically verified completion
- false-success rate
- correct bounded failures
- tokens
- LLM calls
- repairs
- steps
- route changes
- meaningful graph-choice changes
- final verified patch behavior

A learned policy should not be promoted merely because it is safe. It must demonstrate a reproducible benefit over the static controller without degrading verified success or bounded-failure correctness.

## Next expected decision

When the 120-run evaluation completes, archive the complete evaluation before any new training or activation decision.

If the final result is `safe-but-no-demonstrated-benefit`, archive candidate `855e378570a9` as safe-but-inert and train a candidate v2 using genuine multi-choice graph decisions rather than forced or effectively deterministic transitions.
