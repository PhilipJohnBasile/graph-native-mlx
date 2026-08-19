# Causal paired policy canary v2

## Purpose

The v2 canary separates policy effects from model-generation and instrumentation effects. It is a safety and causal-wiring gate, not a promotion benchmark.

## Shared cases

One manifest and one set of committed repositories are reused across all arms. Each case has the same:

- run ID, task text, base commit, and verifier commands
- Qwen checkpoint and immutable model revision
- MLX and MLX-LM versions
- graph schema and traversal limits
- token, call, repair, and time budgets
- logical repository alias and evaluation seed

Each arm uses an isolated SQLite database, worktree root, artifact root, and hidden-artifact root. Original repositories must remain clean and pinned to their original commit.

## Canonical prompts

Paired mode retains real runtime identity for Git and checkpoint safety but replaces model-visible identity with stable aliases:

```text
run ID          → <run:case-id>
source checkout → <repository:case-id>
active worktree → <worktree:case-id>
artifact root   → <artifacts:case-id>
```

Timestamps and elapsed-duration fields are omitted or normalized in the prompt copy. Raw prompts are never persisted. Every graph-level language-model operation stores only:

- system, user, and combined SHA-256
- node and call kind
- case ID and revision
- deterministic generation seed
- explicit flags confirming raw prompt text was not stored

## Deterministic generation

The provider resets the MLX random stream before every paired generation. The seed is derived from:

```text
base evaluation seed
case ID
node ID
call kind
revision
canonical system-prompt hash
canonical user-prompt hash
```

The canary sets temperature `0`, `top_p=1`, and `top_k=0`. Recovery calls use bounded deterministic seed offsets. Identical canonical prompts therefore receive identical seeds and are expected to generate identical artifacts.

## Four arms

| Arm | Policy loaded | Route scale | Transition scale | Purpose |
|---|---:|---:|---:|---|
| Static | no | n/a | n/a | hardcoded baseline |
| Shadow | yes | 0 | 0 | detect instrumentation or prompt contamination |
| Route-only | yes | 1 | 0 | isolate route-head intervention |
| Full | yes | 1 | 1 | evaluate complete candidate |

The required causal comparisons are:

```text
Static vs Shadow      instrumentation effect
Shadow vs Route-only  route-head effect
Route-only vs Full    transition-head effect
```

## Meaningful decisions

A policy output is not counted as a meaningful control decision unless more than one graph-valid choice exists. Telemetry records:

- valid choice count
- whether policy/hidden context was evaluated
- whether the policy could change the choice
- static and learned choices
- whether the selected choice changed

When `GRAPH_MODEL_MLX_SKIP_FORCED_POLICY=true`, forced transitions bypass hidden prefill and policy inference. Hard graph masks and the host-side selected-edge validation remain authoritative.

## Semantic-review adjudication

Controlled fixtures may provide a declarative contract oracle. Supported checks include:

- Python function-to-callee relationships using AST inspection
- exact allowed and required changed-file sets
- tests unchanged
- changed text files end with a newline
- deterministic tests passed

An oracle can inspect unchanged files, preventing a diff-only reviewer from falsely claiming that existing behavior is absent.

Two adjudication modes exist:

1. **Authoritative evaluation contract.** The fixture explicitly marks a complete controlled oracle as authoritative. A full oracle pass can overrule an initial semantic rejection. This is evaluation-only and must be opt-in.
2. **Independent appeal.** A non-authoritative oracle pass triggers a second semantic reviewer that must treat passing concrete oracle facts as authoritative while evaluating any remaining objective or safety concern.

No oracle means the original semantic-review behavior is unchanged.

## Gate

The four-case gate passes only when:

- all arms reach every expected outcome
- every arm has zero false successes
- static and shadow are exact matches for route, path, prompt hashes, patch hashes, verification, and terminal outcome
- the impossible contract reaches bounded abort in every arm
- no raw prompt is persisted
- all original repositories remain clean and unchanged
- candidate identity and per-arm scales are correct

A pass authorizes a larger counterbalanced held-out evaluation. It does not activate or promote the candidate.
