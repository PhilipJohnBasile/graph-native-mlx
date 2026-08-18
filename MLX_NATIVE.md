# MLX-Native Integration

## Scope

v0.5.1 keeps language generation, hidden extraction, and optional graph-policy inference in one process while retaining Git, tests, journals, permissions, and promotion in the host effect plane.

```text
Model plane       Qwen generation and selected hidden representations
Policy plane      explicit state + projected Qwen features + hard MLX masks
Effect plane      Git, tests, checkpoints, permissions, promotion, cleanup
```

Tensor evaluation is not a durable transaction log and cannot authorize external effects. The separation is intentional.

## Source map

```text
src/graph_model/
  runtime.py                       graph authority and durable transitions
  store.py                         SQLite checkpoints, traces, run exclusion
  workspace.py                     worktrees, patch transaction, verifier runner
  operators.py                     typed node implementations
  trace_collection.py              sequential real-repository trace batches
  graphs/coding_supergraph.yaml    editable graph source

  mlx_native/
    provider.py                    resident MLX-LM provider and hidden capture
    qwen_hidden.py                 backbone discovery and selected-layer forward
    hidden_state.py                projection, identity, artifact integrity
    decision_context.py            bounded evidence-aware hidden prompt context
    graph_tables.py                graph compiler and schema hashing
    generated_coding_graph.py      immutable generated constants
    decision.py                    masked softmax and argmax
    features.py                    explicit 62-value state vector
    policy.py                      gated route/edge/stop/value/cost sidecar
    controller.py                  masks and shadow/guarded/active deployment
    policy_audit.py                durable decision-deployment summaries
    qualification.py               M5/Metal acceptance and integrity report
    training_data.py               trace export and dataset identity checks
    trainer.py                     MLX sidecar training and validation split
    doctor.py                      Mac, model, hidden, and policy diagnostics
```

## Direct MLX-LM provider

`MLXLocalProvider`:

- loads one model and tokenizer through `mlx_lm.load`
- keeps them resident across graph nodes and trace tasks
- streams node generations through `mlx_lm.stream_generate`
- adapts to compatible MLX-LM loader signatures
- applies the tokenizer chat template with a role-delimited fallback
- serializes generation and hidden extraction under one lock
- reports prompt and completion token counts
- supports a local path or Hugging Face repository ID
- fingerprints local model identity and requires immutable remote identity for promoted hidden traces

The direct provider passes no separate draft model to `stream_generate`. Its generation mode is therefore reported as `mlx-lm-standard-stream-generate`. A loaded checkpoint may expose `mtp_forward`, but that is capability metadata, not evidence that generated tokens used MTP.

## Qwen hidden-state path

The provider renders the task without an assistant generation suffix and performs a bounded hidden forward.

Default:

```text
layers      final
pooling     last-token
projection  deterministic CountSketch
size        256
scope       task
```

Layer selectors:

```text
final
0, 7, 31
-1, -2
25%, 50%, 75%, 100%
```

`final` captures the normalized language-backbone output. `100%` refers to the raw output of the last decoder layer before final normalization.

Intermediate-layer extraction follows the discovered Qwen text backbone with `embed_tokens`, decoder layers or pipeline layers, and final normalization. Pipeline-parallel extraction supports final-only capture; selected intermediate layers require a single-process backbone.

## Task and decision scopes

`GRAPH_MODEL_MLX_HIDDEN_SCOPE=task` renders only the software task. The projected representation is reused throughout the run.

`GRAPH_MODEL_MLX_HIDDEN_SCOPE=decision` adds a deterministic JSON context containing bounded current state:

- node, route, and verdict
- step, attempt, repair, and revision counts
- edge traversals
- remaining budgets
- selected apply/test/review evidence

Known secret-bearing keys are redacted. Depth, collection sizes, strings, and evidence counts are bounded. A new hidden forward occurs only when the resulting decision-context hash changes.

## Hidden artifact

Raw selected tensors are transient. The artifact store writes:

- projected features
- model fingerprint
- extractor schema hash
- task and prompt hashes
- source backbone path and layer labels
- dimensions, pooling, and token count
- artifact content hash

It does not write raw task text, rendered prompts, or raw hidden tensors. Projected features remain task-derived private data.

## MLX policy sidecar

The sidecar fuses explicit and optional Qwen features and returns:

- route logits
- one logit per compiled edge
- stop-action logits
- completion-value estimate
- normalized token, latency, and tool-call costs

Hard masks are applied after logits are produced. A second host-side validation rejects any selected transition outside the current candidate set.

## Deployment modes

The controller always computes a hardcoded baseline. When a sidecar exists, it also computes a masked candidate.

- `shadow`: baseline selected, candidate audited
- `guarded`: candidate selected only after value/confidence/margin gates
- `active`: candidate selected whenever legal

The controller identity includes mode and thresholds, preventing a resumed run from changing deployment semantics.

## Qualification

`graph-model qualify-mac` uses one resident provider to collect:

- platform and package versions
- Metal/device information
- allocator active, peak, and cache memory
- model identity and capability metadata
- repeated hidden-feature determinism
- exact repeated-generation regression results
- required-gate verdicts
- content SHA-256

The repeated-generation probe does not pass an explicit reusable server prompt cache. It is a resident-process regression test, not a complete server cache qualification.

## Training

The Qwen backbone remains frozen. Training operates on exported projected features and explicit state.

Successful runs train route, edge, stop, value, and cost targets. Failed runs train value and cost but do not positively imitate failed actions. Train/validation partitioning occurs by run ID.

A policy config is bound to:

- graph schema
- explicit and hidden feature sizes
- model fingerprint
- hidden extractor schema
- fusion version
- policy configuration and weight hashes

A mismatch fails closed before a policy decision is used.
