# Hidden-State Graph Policy

## Purpose

The hidden-state path adds semantic information from the resident Qwen model to the graph controller without generating control-flow prose and without weakening the graph boundary.

The policy combines:

```text
Qwen representation       task- or decision-scoped
explicit execution state  changes after each node
```

The sidecar emits residual preferences. Hard graph masks remain authoritative.

## Capture path

```text
task + optional bounded decision context
  │
  ▼
chat template without assistant generation suffix
  │
  ▼
bounded token sequence
  │
  ▼
selected Qwen decoder layers
  │
  ▼
last-token / mean / mean-last pooling
  │
  ▼
per-view L2 normalization
  │
  ▼
deterministic CountSketch projection
  │
  ▼
fixed-size feature vector
```

The default is the normalized final backbone representation pooled at the last token and projected to 256 values.

## Scopes

### Task scope

```bash
export GRAPH_MODEL_MLX_HIDDEN_SCOPE=task
```

The prompt contains only the task. One observation is reused for the same task/model/extractor tuple. Dynamic graph state remains available through the 62 explicit controller features.

Task scope should be the first performance baseline.

### Decision scope

```bash
export GRAPH_MODEL_MLX_HIDDEN_SCOPE=decision
```

The prompt also contains deterministic bounded state for the current decision. Context includes the node, route, verdict, attempts, repair and plan-revision counts, edge traversals, remaining budgets, and selected apply/test/review evidence.

The context builder:

- sorts mappings for deterministic output
- limits nesting, collection sizes, evidence entries, and string lengths
- redacts keys associated with tokens, passwords, credentials, private keys, sessions, authorization, and API keys
- hashes the final context to control capture reuse

Decision scope can add one bounded Qwen forward at each distinct decision state. Stop and edge decisions at an unchanged node/state share the same observation.

Redaction is defense in depth, not a complete data-loss-prevention system. Do not place secrets in tasks or repository evidence.

## Backbones

The extractor discovers normalized language backbones through common paths such as:

- `model.model`
- `model.language_model.model`
- `model.language_model`
- `model.transformer`

A supported backbone must expose `embed_tokens`, decoder `layers` or `pipeline_layers`, and `norm`.

The final-only path invokes the backbone directly. Selected intermediate layers reproduce the public Qwen layer/mask sequence with no KV cache and pipeline size one.

## Layer selectors

Selectors are resolved against the model’s decoder-layer count:

- `final`: normalized output after the final decoder layer
- `0`, `7`, `31`: zero-based decoder-layer output
- `-1`, `-2`: Python-style negative index
- `25%`, `50%`, `75%`, `100%`: layer at that depth

`100%` is the raw last decoder-layer output; `final` is the normalized final representation.

## Projection

Raw selected representations are transient. Each selected layer receives a named, disjoint projection block. Projection is deterministic across Python processes and bound into the extractor schema hash.

The artifact contains the projected vector, not the raw hidden tensor. A 256-value artifact remains small enough for traces and repeated sidecar training even when the backbone hidden width is several thousand values.

## Model and prompt binding

A policy config records:

- graph name, version, and schema hash
- explicit input size
- projected hidden feature size
- hidden extractor schema hash
- model fingerprint
- fusion version

The extractor schema includes scope and prompt version. A policy trained from task-scope features cannot silently consume decision-scope features.

Remote models should use an immutable full commit revision. Local directories receive a snapshot fingerprint using hashes of small identity files plus weight-file names, sizes, and nanosecond modification times. `GRAPH_MODEL_MLX_MODEL_DIGEST` can add an operator-supplied SHA-256 identity.

## Fusion network

```text
explicit features ─► LayerNorm ─► projection ─► hidden projection ┐
                                                                  ├─► gated residual fusion
Qwen features ─────► LayerNorm ─► backbone projection ────────────┘
                                                                       │
                               ┌──────────────┬─────────────┬───────────┤
                               ▼              ▼             ▼           ▼
                            route           edge          stop      value/cost
```

Qwen weights remain frozen; trace collection and sidecar training are separate stages.

## Training semantics

Decision labels are the executed route and transition. The terminal result provides reward.

- Successful decisions receive action-imitation weight.
- Failed decisions receive zero action-imitation weight.
- Successful and failed decisions both train success-value and cost prediction.
- Validation splitting occurs by `run_id`.
- The best validation checkpoint is retained and early stopping is supported.

This avoids teaching the controller to reproduce a path merely because it appeared in a failed trace.

## Stored artifact

Each hash-addressed JSON artifact contains:

- projected features
- feature size
- model fingerprint
- extractor schema hash
- raw hidden width and transient vector size
- task and rendered-prompt hashes
- token count
- backbone path
- selected layer labels
- pooling method

It excludes raw task text, rendered prompt text, decision-context text, and raw hidden tensors. Projected embeddings may still encode semantic information and should be treated as private model data.

## Performance profile

Task scope adds one bounded prefill-style backbone forward per distinct task/model/extractor tuple.

Decision scope adds a forward when bounded graph evidence changes. The selected-layer path costs more than final-only capture. Start with:

```text
scope   task
layers  final
pooling last-token
```

Then evaluate decision scope and intermediate layers as separate ablations. Do not change several extraction variables in one policy comparison.

## Failure behavior

Hidden-aware policy loading is fail-closed:

- missing hidden source: policy cannot activate
- feature-size mismatch: rejected
- extractor-schema mismatch: rejected
- model-fingerprint mismatch: rejected
- malformed or modified artifact: rejected
- unsupported backbone: capture fails before graph execution proceeds

The graph retains deterministic hardcoded priors when no hidden policy is configured.

## Bootstrap corpus

After M5 qualification, the source package includes a controlled route/status-diverse corpus generator and collector. It is intended to prove the end-to-end trace/export path before real-project collection:

```bash
scripts/collect-bootstrap-policy-corpus-mac.sh
```

The bootstrap collector uses a dedicated database and does not activate policy weights. It exports successful and failed records; failed records have zero route/edge/stop imitation weight in v0.5.3 but remain available for value and cost learning. Treat the fixtures as pipeline qualification data, not as a production-quality training corpus.

## Causal paired evaluation in v0.5.7

Paired evaluation separates policy effects from prompt, path, timing, and random-stream effects.

The live state retains actual run IDs and paths for checkpoint and Git safety. The prompt copy uses stable aliases and normalizes timing. Every model call stores hashes and a deterministic seed, never raw prompt text.

Policy interventions are independently controlled:

```bash
export GRAPH_MODEL_MLX_ROUTE_POLICY_SCALE=0
export GRAPH_MODEL_MLX_TRANSITION_POLICY_SCALE=0
export GRAPH_MODEL_MLX_SKIP_FORCED_POLICY=true
```

A forced transition with one graph-valid choice is not treated as a learned decision. When forced-choice skipping is enabled, no policy hidden forward or sidecar inference is performed for that transition.

The four deployment arms are:

```text
static      no policy
shadow      policy loaded, route=0, transition=0
route-only  policy loaded, route=1, transition=0
full        policy loaded, route=1, transition=1
```

Static and shadow must match exactly before route or transition effects are interpreted.
