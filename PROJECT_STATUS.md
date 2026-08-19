# Project status

## Current source

`v0.5.7`

## Proven on the physical M5 Max

- MLX/Metal model loading
- structured generation through MLX-LM
- selected Qwen hidden-state extraction
- 5,120-dimensional hidden state projected to 256 graph features for the qualified checkpoint
- hard-masked route, edge, and stop control
- detached worktree repository execution
- transactional patch application
- real verifier commands and independent review
- resumable bounded repair
- a clean 16-task bootstrap corpus with 165 exported policy records
- candidate `855e378570a9` loading with exact graph, model, and extractor identity binding

## Bootstrap result

- 16 repository tasks
- 14 completed
- 2 intentional bounded aborts
- 0 collector errors
- 0 terminal-outcome mismatches
- 133 positive-reward records
- 32 zero-reward records

## Candidate and canary status

The first hidden-fusion candidate remains globally disabled.

Canary v1 produced static `4/4` and learned `3/4`, but post-run forensics found the same route, zero meaningful transition changes, different generated patches, and a false semantic-review chain. Because model-visible runtime identity and generation randomness were not controlled, the result was scientifically inconclusive rather than a causal rejection of the policy.

v0.5.7 contains the corrective evaluation stack:

- canonical paired prompts
- hash-only prompt audits
- prompt-derived deterministic generation seeds
- separate route and transition interventions
- static, shadow, route-only, and full arms
- forced-choice telemetry
- declarative contract oracles and review adjudication

## Not yet proven

- that candidate `855e378570a9` changes a meaningful graph choice on the paired cases
- that a trained graph policy outperforms the static graph
- that graph control outperforms a matched-budget Ralph-loop baseline on a sufficiently large held-out benchmark
- broad generalization beyond the bootstrap task families

## Next gate

Run `scripts/run-paired-policy-canary-v2-mac.sh` on the qualified M5 Max. Static and shadow must be exact matches, every arm must reach all expected outcomes with zero false successes, and the candidate must remain disabled. A four-case pass authorizes a larger counterbalanced held-out evaluation; it does not authorize activation.
