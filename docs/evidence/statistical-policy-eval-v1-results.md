# Statistical policy evaluation v1 — final result

## Result

Candidate `855e378570a9` completed the repeated statistical evaluation with no observed safety regression and no demonstrated benefit over the static controller.

- Date: 2026-08-19
- Evaluation root: `/Users/pjb/graph-native-mlx/.graph-model/evaluation/statistical-policy-eval-v1-20260819-152952`
- Candidate weights SHA-256: `855e378570a95fc79b18eea00280c9b6a1c6b3f4091e85bfadcb3da77a8a15fb`
- Candidate activation: **disabled globally**
- Classification: **safe-but-no-demonstrated-benefit**

## Design

- 12 representative held-out repositories
- 5 repetitions per arm
- static and full-policy arms
- 120 executions total
- sampling temperature `0.1`
- static controller: `hardcoded-priors-only`
- full controller: candidate `855e378570a9`
- forced one-choice transitions skipped by the learned policy

The selected set contained:

- 3 fast repairs
- 3 deep tasks
- 3 repair-oriented tasks
- 1 no-change audit
- 2 contradictory/impossible contracts

## Final outcomes

| Metric | Static | Full policy | Difference |
|---|---:|---:|---:|
| Correct outcomes | 60/60 | 60/60 | 0 |
| False successes | 0 | 0 | 0 |
| Route changes | — | 0/60 paired trials | 0 |
| Final patch changes | — | 0/60 paired trials | 0 |
| Mean token delta | — | — | -1.0 token |
| Mean LLM-call delta | — | — | +0.000 |
| Mean repair delta | — | — | +0.000 |
| Mean step delta | — | — | +0.000 |

Safety gate: **PASS**

Benefit gate: **FAIL**

The two impossible contracts intentionally ended at the bounded abort node. Those `failed` statuses are correct outcomes, not false successes or regressions.

## Interpretation

This experiment supports the following conclusions:

1. Candidate `855e378570a9` did not reduce verified success on the measured distribution.
2. It produced no false successes across 60 full-policy trials.
3. It preserved correct bounded-failure behavior.
4. It did not change the selected route or final verified patch in any paired trial.
5. It did not materially reduce tokens, model calls, repairs, or graph steps.
6. The candidate is therefore safe in this measured study but effectively inert.

The result does **not** justify global activation or guarded promotion. Safety without measurable advantage is insufficient for deployment.

## Relationship to prior evidence

- The four-arm causal canary passed 4/4 in static, shadow, route-only, and full arms.
- The 24-case counterbalanced study produced 24/24 correct outcomes in both static and full arms with zero false successes and no meaningful controller changes.
- A targeted four-execution replay showed that exact intermediate generation is not bitwise reproducible on this MLX/Qwen stack even when final verified behavior is identical.
- The present 120-run study therefore used repeated outcome distributions rather than byte-identical intermediate generation as the decisive evidence.

## Candidate disposition

Archive candidate `855e378570a9` as:

```text
safe on measured evaluations
no demonstrated benefit
not activated
not promoted
```

## Next research gate

Train candidate v2 from **genuine multi-choice graph decisions** rather than forced or effectively deterministic transitions.

Candidate v2 should:

- exclude one-choice transitions from action-learning targets;
- balance real `fast`, `deep`, and `repair` route decisions;
- train against advantage over the static controller rather than raw successful imitation;
- use repository/task-family-separated train and validation splits;
- retain deterministic fallback to hardcoded routing below a calibrated confidence threshold;
- be evaluated with the same repeated statistical harness before any activation decision.

No policy path should be written to `.graph-env` as a consequence of this result.
