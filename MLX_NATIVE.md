# MLX-Native Integration

## Scope

Version 0.2 is the first deployable MLX checkpoint. It puts two computational paths in the same process:

1. Qwen node generation through `mlx_lm.load` and `mlx_lm.stream_generate`.
2. Graph route, stop, and edge selection through MLX tensor operations.

The durable executor remains outside MLX. This is intentional. Tensor evaluation cannot commit a filesystem mutation, journal a remote side effect, or authorize a deployment.

## Authority boundary

The transition pipeline is ordered and non-bypassable:

```text
compiled graph row
      │
      ▼
runtime condition evaluation
      │
      ▼
edge traversal-limit filtering
      │
      ▼
optional learned residual logits
      │
      ▼
MLX hard mask + softmax + argmax
      │
      ▼
runtime validates selected edge again
      │
      ▼
checkpointed transition commit
```

The learned policy never receives permission to create an edge. It only ranks the candidate set produced by the graph and runtime.

## Source map

```text
src/graph_model/
  controller.py                    controller protocol and deterministic fallback
  runtime.py                       durable authority and transition validation
  provider.py                      provider protocol and JSON extraction
  graphs/coding_supergraph.yaml    editable graph source

  mlx_native/
    provider.py                    resident in-process MLX-LM provider
    graph_tables.py                graph compiler and schema hashing
    generated_coding_graph.py      generated immutable graph constants
    decision.py                    MLX hard mask, softmax, and argmax
    features.py                    explicit task/run-state feature vector
    policy.py                      MLX route/edge/stop/value/cost sidecar
    controller.py                  graph-controller integration
    training_data.py               trace-to-policy-dataset export
    trainer.py                     multi-task MLX policy trainer
    doctor.py                      target-Mac diagnostics and model-load check
```

## Generated graph tables

`graph-model compile-graph` validates the YAML graph and emits stable constants:

- sorted node IDs
- sorted edge keys
- edge source and target indices
- edge priorities
- edge traversal limits
- condition expressions
- a per-node structural edge mask
- terminal-node mask
- deterministic graph-schema SHA-256

The generated module is checked against the loaded graph at controller startup. Any graph edit that is not followed by regeneration causes a schema mismatch instead of silently running stale masks.

## MLX decision backend

The production backend performs:

```python
values = mx.array(logits, dtype=mx.float32)
allowed = mx.array(mask, dtype=mx.bool_)
masked = mx.where(allowed, values, mx.full_like(values, -1e9))
probabilities = mx.softmax(masked, axis=-1)
selected = mx.argmax(probabilities, axis=-1)
mx.eval(probabilities, selected)
```

At least one mask entry must be true. Invalid dimensions and all-false masks fail before evaluation.

The portable test backend implements the same masked-softmax/argmax semantics in Python. It is injected explicitly in tests; `--controller mlx` does not silently fall back to it.

## Sidecar policy

The v0.2 sidecar consumes an explicit feature vector, not Qwen hidden states.

```text
16 task features
28 run-state features
11 one-hot node features
---------------------------
55 inputs for the default graph
```

It emits:

```text
3 route residual logits
16 edge residual logits
4 stop residual logits
1 success logit
3 cost values
```

The success output is exposed through a sigmoid. Cost outputs are exposed through softplus. Route, edge, and stop values remain logits because they are combined with hardcoded priors before masking.

Policy configuration is bound to:

- graph name
- graph version
- graph schema hash
- input size
- route, edge, and stop output dimensions

Controller identity also includes SHA-256 fingerprints of the policy weights and config. A checkpoint cannot resume with different sidecar bytes under the same filename.

## Direct MLX-LM provider

`MLXLocalProvider`:

- loads the model and tokenizer once
- accepts a local model path or Hugging Face repository ID
- supports a pinned Hugging Face revision across MLX-LM 0.31.3 and newer loader signatures
- applies the tokenizer chat template when usable
- falls back to explicit system/user role delimiters when the template fails
- builds a sampler through `mlx_lm.sample_utils.make_sampler`
- consumes streamed text segments from `stream_generate`
- records prompt and completion token counts
- extracts the final complete JSON object from reasoning or surrounding prose
- serializes generation access because decoding mutates model/cache state
- moves blocking generation into a worker thread so the async graph runtime remains responsive

The provider intentionally does not enable speculative decoding or an MTP-specific path in v0.2. Establish graph correctness first; then benchmark MTP as a node-generation optimization.

## Durable state and resume

SQLite commits each completed node together with:

- canonical input hash
- node result
- state delta and artifacts
- stop decision
- selected edge
- candidate masks
- provider identity
- controller identity

A resume validates graph, configured provider identity, controller kind, graph hash, policy fingerprints, and policy scale before continuing. Local or remote model repositories should still be version-pinned for reproducible production runs; a repository name alone is a configuration identity, not a cryptographic hash of 27B model weights.

## Policy-data format

`graph-model export-mlx-policy` emits JSONL records with:

```json
{
  "decision_type": "transition",
  "features": [0.0],
  "route_label": -1,
  "edge_label": 4,
  "stop_label": 0,
  "allowed_edge_mask": [false, true],
  "allowed_stop_mask": [true, false, false, false],
  "reward": 1.0,
  "cost_target": [0.12, 0.08, 0.20]
}
```

`-1` means that a label is not applicable to that record. Masked cross-entropy ignores it.

The included trainer minimizes:

```text
route cross-entropy
+ masked edge cross-entropy
+ masked stop cross-entropy
+ 0.5 × success-value MSE
+ 0.25 × normalized-cost MSE
```

Behavioral cloning is only a bootstrap. A stronger policy dataset should retain graph-search winners, held-out evaluator results, cost-aware preferences, and explicit failed alternatives.

## Current boundary and next stage

Implemented now:

- direct resident MLX-LM backbone
- compiled graph schema and masks
- MLX route/stop/edge decisions
- trainable MLX sidecar policy
- durable external execution
- policy trace export and training

Not yet implemented:

- a Qwen architecture hook returning selected hidden states
- joint text-plus-policy LoRA training
- MTP/speculative generation integration
- sandboxed repository tools that mutate and test a real checkout
- offline MCTS graph promotion
- inference-time validated region grafting

The next model-level milestone is hidden-state fusion. It should append a selected Qwen representation to the existing explicit features while preserving the same hard graph mask and external transaction boundary.
