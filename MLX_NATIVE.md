# MLX-Native Integration

## Scope

Version 0.3 keeps the model and graph-policy computation in one process while adding a real repository effect plane.

1. Qwen node generation runs through `mlx_lm.load` and `mlx_lm.stream_generate`.
2. Route, stop, and edge selection can run through MLX tensors and optional learned sidecar heads.
3. Git mutation, test execution, checkpoints, permissions, and promotion remain in the host runtime.

Tensor evaluation cannot provide a durable transaction log or safely authorize an external side effect, so this boundary is intentional.

## Transition pipeline

```text
compiled graph structure
        │
runtime predicates and traversal caps
        │
optional policy residual logits
        │
MLX hard mask + softmax + argmax
        │
runtime validates selected edge again
        │
durable checkpoint commit
```

The learned policy cannot unmask an invalid transition.

## Source map

```text
src/graph_model/
  runtime.py                       graph authority, active budgets, run lease
  store.py                         SQLite checkpoints, traces, same-run exclusion
  workspace.py                     Git worktrees, patch transaction, verifier runner
  operators.py                     typed graph-node implementations
  graphs/coding_supergraph.yaml    editable graph source

  mlx_native/
    provider.py                    resident direct MLX-LM provider
    graph_tables.py                graph compiler and schema hashing
    generated_coding_graph.py      immutable generated constants
    decision.py                    MLX masked softmax and argmax
    features.py                    explicit 62-value policy vector
    policy.py                      route/edge/stop/value/cost network
    controller.py                  hard-mask controller integration
    training_data.py               trace-to-policy JSONL export
    trainer.py                     multi-task MLX policy trainer
    doctor.py                      Mac and model-load diagnostics
```

## Direct MLX-LM provider

`MLXLocalProvider`:

- keeps one model and tokenizer resident
- accepts a local path or Hugging Face repository ID
- supports revision pinning across compatible MLX-LM loader signatures
- applies the tokenizer chat template when available
- falls back to explicit system/user role delimiters
- uses `make_sampler` and `stream_generate`
- serializes generation because decoding cache state is mutable
- runs blocking generation in a worker thread
- extracts the final complete JSON object from surrounding reasoning text
- records prompt and completion token counts
- defaults to 8,192 output tokens for coding patches

The provider does not enable MTP/speculative decoding in v0.3. MTP may later accelerate text generation inside a node, but it must not own graph transitions, idempotency, or recovery.

## Compiled graph tables

`graph-model compile-graph` emits:

- stable sorted node IDs
- stable edge keys
- source/target index arrays
- edge priorities and traversal limits
- condition strings
- per-node structural edge masks
- terminal mask
- graph metadata
- deterministic schema SHA-256

The generated module validates itself against the loaded YAML graph. Schema drift fails before execution.

## MLX decision backend

The production decision path is equivalent to:

```python
values = mx.array(logits, dtype=mx.float32)
allowed = mx.array(mask, dtype=mx.bool_)
masked = mx.where(allowed, values, mx.full_like(values, -1e9))
probabilities = mx.softmax(masked, axis=-1)
selected = mx.argmax(probabilities, axis=-1)
mx.eval(probabilities, selected)
```

All-false masks and dimension mismatches fail explicitly.

## Policy sidecar

The default graph uses:

```text
16 task features
34 run/repository-state features
12 one-hot node features
---------------------------
62 total inputs
```

Outputs:

```text
3 route residual logits
19 edge residual logits
4 stop residual logits
1 success logit
3 positive cost estimates
```

The sidecar config binds graph name, version, schema hash, input size, and output dimensions. Controller identity includes SHA-256 fingerprints of policy weights and config.

## Repository-aware features

In addition to budget and route state, the policy sees indicators for:

- repository workspace presence
- pending patch presence
- candidate presence
- apply-report presence
- apply pass/fail
- test-report presence
- test-induced workspace mutation
- review and diagnosis presence

Structural safety remains hardcoded even after policy training.

## Training data

`graph-model export-mlx-policy` emits route and transition records with:

- exact feature vector
- graph schema hash
- selected labels
- allowed edge and stop masks
- terminal reward
- normalized token, active-time, and tool-call cost targets

The included trainer combines route cross-entropy, masked edge cross-entropy, masked stop cross-entropy, success-value MSE, and cost MSE.

Behavioral cloning should be treated as initialization. Stronger training should use held-out evaluator scores, graph-search winners, failed alternatives, and explicit cost-aware preferences.

## Current model-level boundary

Implemented:

- resident direct MLX-LM backbone
- compiled hard graph masks
- optional trainable MLX policy
- real repository workspace and verifier integration
- durable external effects and policy trace export

Not yet implemented:

- Qwen hidden-state extraction for policy fusion
- joint LM/route/edge/stop/value/cost LoRA training
- MTP/speculative generation integration
- hostile-code process sandboxing
- offline MCTS graph promotion
- validated inference-time subgraph grafting

The next model-level step is to append a selected Qwen hidden representation to the explicit policy vector while retaining the same hard graph mask and external transaction boundary.
