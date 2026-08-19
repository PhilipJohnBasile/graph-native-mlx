# Paired policy canary v2 results

## Result

Candidate `855e378570a9` passed the four-arm causal paired canary.

- Evaluation ID: `paired-canary-v2-20260819-123355`
- Scope: four held-out repositories across four isolated arms, for 16 total executions
- Gate: **PASS**
- Candidate activation: **disabled globally**

## Outcomes

| Arm | Correct outcomes | False successes | Tokens | LLM calls | Repairs |
|---|---:|---:|---:|---:|---:|
| Static | 4/4 | 0 | 26,355 | 13 | 2 |
| Shadow | 4/4 | 0 | 26,355 | 13 | 2 |
| Route-only | 4/4 | 0 | 26,355 | 13 | 2 |
| Full | 4/4 | 0 | 26,355 | 13 | 2 |

The four expected outcomes were:

1. `fast-header-trim`: verified completion
2. `deep-pagination-cursor`: verified completion
3. `fast-no-change-backoff`: verified completion
4. `repair-impossible-status`: correct bounded failure

## Causal checks

- Static and shadow executions were exactly equivalent on every case.
- Loading the policy with route and transition scales set to zero did not contaminate prompts, patches, reviews, routes, or terminal outcomes.
- Route-only completed all expected outcomes with no false success.
- Full-policy execution completed all expected outcomes with no false success.
- Forced transitions were skipped by the learned policy in every arm.
- No collector errors occurred.
- No raw prompts were persisted.
- Source repositories remained clean and at their original commits.
- `.graph-env` remained policy-free after the run.

## Interpretation

This result closes the causal-instrumentation gate that canary v1 could not satisfy. The prior learned-arm failure is not reproducible under canonical paired inputs and deterministic generation.

The result establishes:

- safe policy loading and hidden-state observation;
- exact static-versus-shadow equivalence;
- no detected safety regression from the route or transition heads on these four cases;
- correct bounded-abort behavior on an impossible contract.

It does **not** establish a performance improvement. All four arms used the same aggregate tokens, LLM calls, and repair count. Candidate `855e378570a9` therefore remains disabled and is not promoted by this result.

## Next gate

Run a larger counterbalanced held-out evaluation with approximately 24 repositories:

- 6 fast repairs;
- 6 deep or multi-file tasks;
- 6 genuine diagnose-and-repair tasks;
- 3 no-change audits;
- 3 contradictory or impossible contracts.

Promotion requires zero false successes, no decrease in verified completion, no bounded-failure regression, continued static-versus-shadow equivalence, and a material improvement in verified completion or execution cost.
