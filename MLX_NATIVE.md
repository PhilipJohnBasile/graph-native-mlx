# MLX-Native Integration

## Scope

Version 0.4 keeps Qwen generation, hidden-state extraction, and graph-policy inference in one process while preserving Git, test, checkpoint, permission, and promotion authority in the host runtime.

```text
Qwen/MLX-LM                   Host runtime
--------------------------    --------------------------------
node text generation          durable SQLite/Restate journal
selected hidden states        Git worktree and patch effects
fixed-size projection         verifier process execution
policy fusion and logits      idempotency and crash recovery
hard tensor mask              final transition validation
```

Tensor evaluation cannot provide a durable transaction log or safely authorize an external side effect. The split is intentional.

## State-aware policy forward

A policy decision is conditioned on the current checkpoint, not only the original task.

The bounded deterministic policy-state prompt includes:

- task identity and bounded task text
- current graph node and decision type
- completed nodes, attempts, and edge counts
- route, verdict, repair, and plan-revision state
- remaining step, model-call, tool-call, token, and active-time budgets
- bounded plan, patch/apply, test, review, and diagnosis evidence
- artifact names and progress state

Runtime identity metadata, secrets, and complete repository artifacts are excluded. Oversized evidence is compacted to an excerpt plus SHA-256.

For one checkpoint, stop and edge selection share the same policy representation. A later checkpoint or changed evidence receives a fresh representation. Hidden observations are held in a bounded LRU cache, and the runtime charges policy-prefill tokens only when Qwen actually performs a new hidden-state forward.

## Hidden-state extraction

The provider resolves supported Qwen backbones exposed as either:

```text
model.model
model.language_model.model
```

Selectors support:

```text
final       normalized final backbone output
0           first decoder layer output
-1          last decoder layer output before final norm
25%         decoder layer at approximately 25% depth
```

Pooling options:

```text
last-token
mean
mean-last
```

The final-only path calls the model’s own normalized backbone. Selected intermediate layers use a no-cache decoder pass with the model’s attention and state-space masks. Multi-layer extraction requires pipeline size one.

Each selected view is L2-normalized, projected into a disjoint block using deterministic CountSketch, and concatenated into a fixed-size vector. The default output is 256 values.

## Artifact boundary

`HiddenStateArtifactStore` writes immutable hash-addressed JSON containing only:

- projected finite features
- model fingerprint
- extractor schema hash
- prompt/task hashes
- source path and layer labels
- dimensions, pooling, and token count

It does not persist the rendered policy prompt or raw Qwen hidden tensors. Every artifact is verified against its SHA-256 and metadata when exported for training.

## Dedicated MLX affinity worker

`MLXLocalProvider` owns one `ThreadPoolExecutor(max_workers=1)`. The following all execute on that same worker:

- model and tokenizer load
- streaming generation
- hidden-state extraction
- policy sidecar construction
- policy forward passes
- MLX masked softmax and argmax when invoked through the controller

This preserves stable thread affinity around mutable generation/cache state and prevents concurrent graph decisions from racing the resident model. The CLI closes the provider after run, resume, benchmark, trace collection, and model diagnostics.

## Transition pipeline

```text
compiled graph structure
        │
runtime predicates and traversal caps
        │
explicit 62-value state vector
        │
projected Qwen state representation
        │
gated residual policy fusion
        │
route / edge / stop residual logits + value + cost
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
  runtime.py                       graph authority, budgets, run lease
  store.py                         SQLite checkpoints and traces
  workspace.py                     Git worktrees, patch transaction, verifier
  operators.py                     typed graph-node implementations
  trace_collection.py              JSONL repository-task trace runner
  graphs/coding_supergraph.yaml    editable graph source

  mlx_native/
    provider.py                    resident MLX-LM provider and affinity worker
    qwen_hidden.py                 Qwen backbone and selected-layer forward
    hidden_state.py                state prompt, projection, immutable artifacts
    graph_tables.py                graph compiler and schema hashing
    generated_coding_graph.py      immutable generated constants
    decision.py                    MLX masked softmax and argmax
    features.py                    explicit 62-value policy vector
    policy.py                      gated route/edge/stop/value/cost network
    controller.py                  state-aware hard-mask integration
    training_data.py               trace-to-policy JSONL export
    trainer.py                     run-split multi-task MLX trainer
    doctor.py                      Mac, hidden-state, policy, and model diagnostics
```

## Direct MLX-LM provider

`MLXLocalProvider`:

- keeps one model and tokenizer resident
- accepts a local path or model repository ID
- supports revision pinning across compatible MLX-LM loader signatures
- applies the tokenizer chat template when available
- falls back to explicit role delimiters
- uses `make_sampler` and `stream_generate`
- extracts the final complete JSON object from surrounding reasoning text
- records generation prompt/completion tokens plus hidden-policy prefill tokens and policy-call latency
- defaults to 8,192 output tokens for coding patches
- exposes a state-aware hidden source to the graph controller
- caches a bounded number of identical hidden observations; cache hits add no new prefill-token charge

MTP/speculative decoding is not enabled by this runtime. It may later accelerate text generation inside a node, but it must not own graph transitions, idempotency, or recovery.

## Compiled graph tables

`graph-model compile-graph` emits stable node IDs, edge keys, source/target arrays, priorities, traversal limits, condition strings, per-node structural masks, terminal masks, and a deterministic graph schema SHA-256. Generated tables validate themselves against the YAML source.

## Policy sidecar

The default explicit input is:

```text
16 task features
34 run/repository-state features
12 one-hot node features
---------------------------
62 explicit inputs
```

The optional backbone input is a fixed projected hidden vector, 256 values by default.

Fusion:

```text
explicit -> normalize -> project ───────────────┐
                                                ├─ concatenate -> gate/candidate -> residual
Qwen projection -> normalize -> project ────────┘
```

Outputs:

```text
3 route residual logits
19 edge residual logits
4 stop residual logits
1 success logit
3 positive cost estimates
```

The config binds graph name/version/schema, explicit size, hidden size, Qwen feature size, hidden extractor schema, and model fingerprint. Controller identity also includes hashes of the sidecar weights and config.

## Trace export and training

`graph-model export-mlx-policy --require-hidden` loads and verifies every referenced hidden artifact, then emits:

- exact explicit feature vector
- projected hidden vector
- graph, model, and extractor identities
- selected route/edge/stop labels
- allowed edge and stop masks
- terminal reward
- normalized token, active-time, and tool-call targets

Mixed explicit-only/hidden datasets, mixed hidden dimensions, mixed extractor schemas, and mixed model fingerprints are rejected.

The trainer uses:

- route cross-entropy
- hard-mask-aware edge cross-entropy
- hard-mask-aware stop cross-entropy
- success-value MSE
- cost MSE
- AdamW
- deterministic run-level train/validation splitting
- early stopping on validation loss

Behavioral cloning is initialization, not proof of improvement. Promotion should depend on held-out task quality, deterministic verifier evidence, cost, and comparison against the hardcoded controller.

## Current boundary

Implemented:

- resident direct MLX-LM backbone
- state-aware Qwen hidden extraction
- deterministic projected hidden artifacts
- gated hidden/explicit policy fusion
- compiled hard graph masks
- real repository workspace and verifier integration
- durable effects, trace collection, export, and policy training

Not yet implemented:

- joint LM plus graph-policy LoRA training
- MTP/speculative generation integration
- hostile-code process sandboxing
- offline MCTS graph promotion
- validated inference-time subgraph grafting

The actual selected model must still pass the M5 Max load and hidden-capture gate; portable Linux validation cannot execute Metal.
