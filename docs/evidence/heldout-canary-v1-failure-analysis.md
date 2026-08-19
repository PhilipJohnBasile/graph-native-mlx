# Held-out policy canary v1: failure analysis

## Result

Candidate `855e378570a9` remained disabled after the first four-repository held-out canary.

```text
Static controller: 4/4 expected outcomes
Learned controller: 3/4 expected outcomes
Static false successes: 0
Learned false successes: 0
```

The learned arm failed only `deep-pagination-cursor`. That one execution accounted for essentially the entire aggregate regression: two additional repair passes, six additional language-model calls, ten additional graph steps, and roughly 48,000 additional tokens.

## Forensic finding

The retained SQLite traces, worktrees, patch proposals, reviews, diagnoses, and hidden-state references were extracted after the run.

```text
Static route: repair
Learned route: repair
Route changed: false
Meaningful changed transitions: 0
Initial patch identical: false
Attribution: generation-or-prompt-divergence
```

The candidate did not change the route and did not change any graph transition for which more than one condition-valid choice existed. The first observable divergence was the generated patch.

Canary v1 did not isolate that divergence causally because the two arms used different run IDs and worktree paths, those values entered model-visible context, generation used temperature `0.1`, and no per-call seed was reset. The result therefore cannot establish that the policy caused the different patch.

## Semantic-review failure

The learned patch passed deterministic verification before every semantic review. The first two semantic reviews nevertheless claimed that `query_offset` had not been changed to use `decode_cursor`.

That claim was contradicted by the baseline repository. Before either arm ran, `paging/query.py` already implemented:

```python
from .cursor import decode_cursor


def query_offset(params: dict[str, str]) -> int:
    return decode_cursor(params.get("cursor"))
```

The static patch correctly modified only the common decoder and `page_start`. The learned arm then entered two unnecessary repair cycles because diagnosis trusted the false semantic claim. A third review raised a newline concern and also questioned the required change from offset `1` to offset `0`, even though offset `0` was explicit in the task contract.

The graph itself behaved safely:

```text
tests pass
→ semantic review fail
→ bounded diagnose/repair
→ tests pass
→ semantic review fail
→ bounded diagnose/repair
→ tests pass
→ semantic review fail
→ abort
```

It did not fabricate success, bypass tests, or exceed the two-repair limit.

## Correct interpretation

```text
Policy loading and identity binding: passed
Hard masks and bounded abort: passed
Policy superiority: not established
Policy inferiority: not established causally
Canary-v1 experimental validity: insufficient
Semantic-review reliability: failed on the pagination case
```

The candidate is preserved for a corrected paired experiment but remains globally disabled and ineligible for promotion.

## Corrective work in v0.5.7

- Canonical model-visible run and filesystem identity
- Prompt-hash audit records without raw prompt persistence
- Prompt-derived per-call MLX seed resets
- Temperature-zero paired evaluation
- Separate route and transition policy scales
- Shadow and route-only intervention arms
- Forced-choice telemetry and optional policy-prefill skipping
- Declarative contract oracles for controlled held-out cases
- Authoritative oracle adjudication only when the evaluation contract explicitly opts in
- Independent appeal review for non-authoritative oracle evidence
