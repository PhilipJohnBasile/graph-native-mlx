# Project status

## Current stable source

`v0.5.6`

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

## Bootstrap result

- 16 repository tasks
- 14 completed
- 2 intentional bounded aborts
- 0 collector errors
- 0 terminal-outcome mismatches
- 133 positive-reward records
- 32 zero-reward records

## Not yet proven

- that a trained graph policy outperforms the static graph
- that graph control outperforms a matched-budget Ralph-loop baseline on a sufficiently large held-out benchmark
- broad generalization beyond the bootstrap task families

## Next gate

Train a candidate policy, freeze it, and evaluate it on repositories and task identities excluded from training. Do not activate or promote it unless verified success is non-inferior and cost/repair behavior improves under matched budgets.
