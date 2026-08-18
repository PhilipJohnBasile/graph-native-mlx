# Architecture

## Design objective

Replace an open-ended retry loop with a durable, inspectable, graph-controlled system in which every operation has a name, state contract, allowed successors, retry cap, evidence boundary, and checkpoint.

```text
Model plane      Qwen/MLX generates typed proposals and state representations
Policy plane     explicit state + Qwen features + validated hard graph masks
Effect plane     Git, tests, journals, permissions, promotion, cleanup
```

The model and learned policy are advisory. The effect plane remains authoritative.

## Default graph

The v0.4 package retains the validated 12-node, 19-edge graph introduced in v0.3. Its schema is unchanged; v0.4 changes the policy representation and training pipeline.

| Node | Kind | Responsibility |
|---|---|---|
| `intake` | router | Select `fast`, `deep`, or `repair` route |
| `context` | tool | Prepare/resume worktree and collect bounded repository context |
| `plan` | LLM | Produce typed implementation plan |
| `plan_check` | verifier | Reject unusable plans; allow one revision |
| `implement` | LLM | Propose a unified diff; no mutation authority |
| `apply` | tool | Validate and transactionally apply pending patch |
| `tests` | verifier | Run bounded deterministic commands and fingerprint integrity check |
| `review` | LLM | Independently assess objective satisfaction after tests pass |
| `diagnose` | LLM | Convert exact apply/test/review evidence into a local diagnosis |
| `repair` | LLM | Propose one patch against the current worktree |
| `finish` | final | Export cumulative verified patch and evidence |
| `abort` | final | Export failed patch/evidence after bounded paths are exhausted |

Repairs return to `apply`, not directly to `tests`. Every proposed mutation crosses the same validation and idempotency gate.

## Policy representation

At each policy checkpoint, the controller builds two inputs:

1. A 62-value explicit vector encoding task shape, current node, budgets, progress, repository state, patch state, and verifier state.
2. A projected Qwen representation of a bounded structured rendering of the same checkpoint plus richer plan/test/review/diagnosis evidence.

Route receives a `route` representation. Stop and edge selection at one post-node checkpoint share one `transition` representation and one policy forward. A changed checkpoint invalidates the cache.

The Qwen feature pipeline is:

```text
checkpointed state
      │
bounded deterministic policy prompt
      │
selected Qwen hidden layer views
      │
per-view pooling and L2 normalization
      │
disjoint deterministic CountSketch blocks
      │
fixed projected feature vector
```

The policy applies gated residual fusion so the explicit graph state remains a direct path even when hidden features are noisy.

## Transition authority

For every non-terminal node:

1. Evaluate edge predicates from explicit state.
2. Remove edges that exhausted traversal limits.
3. Produce explicit and optional hidden policy inputs.
4. Predict residual route/edge/stop logits, value, and cost.
5. Apply the hard MLX mask to surviving choices only.
6. Validate the returned edge against the candidate set again in Python.
7. Commit the node result and transition atomically.

A policy returning a masked or invented edge raises a runtime error.

## Repository transaction

```text
model JSON patch proposal
          │
normalize fenced/plain diff
          │
validate size, path policy, and diff-header consistency
          │
write immutable patch artifact and operation intent
          │
git apply --check
          │
apply in detached worktree
          │
verify declared paths and new fingerprint
          │
write committed operation ledger
```

If validation fails after mutation, the runtime reverses the patch and verifies restoration. If the process crashes after application but before the committed ledger, replay recognizes the matching intent and already-applied state rather than applying twice.

## Worktree isolation

A new repository run requires a clean Git top-level checkout. The base reference is resolved to an immutable commit before creating a detached worktree. Paths are derived from hashes of the source path and safe run-ID components; run IDs cannot escape configured roots.

In-place mode exists for controlled use, but detached worktree mode is the default.

## Verifier integrity

Verifier commands run without a shell. The runtime parses with `shlex`, rejects shell control syntax, requires an allowlisted executable, resolves through a sanitized `PATH`, restricts Git to read-only subcommands, runs with no stdin in a new process group, enforces time/output bounds, and compares tracked workspace fingerprints before and after the verifier sequence.

A zero-exit test that mutates tracked source still fails verification.

The verifier is not a hostile-code sandbox. Repository code runs with local user permissions.

## Durable state

`RunState` stores graph identity, task, current node, explicit data/artifacts, node attempts, edge traversal counts, completed-node history, model/tool/policy calls, generation and hidden-policy token counts, active-time metrics, budgets, progress state, provider/controller identity, output, and terminal error.

SQLite commits each node result and resulting checkpoint in one transaction. Append-only event rows record route, stop, edge, masks, explicit policy vectors, hidden artifact references, and policy metrics.

Side-effecting nodes are forbidden from ordinary result caching. They implement operation-specific idempotency boundaries.

## Hidden artifact integrity

Projected hidden artifacts are immutable and content addressed. References bind:

- artifact SHA-256
- model fingerprint
- hidden extractor schema
- feature and raw dimensions
- prompt/task hashes
- selected layer labels and pooling

Export reloads each artifact, checks the hash and metadata, and rejects missing or mixed identities. Raw prompts and raw hidden tensors are not written to the hidden artifact store.

## MLX concurrency model

The direct provider owns a single affinity worker. Model load, generation, hidden extraction, policy construction, policy inference, and masked tensor selection run there. Hidden observations use a bounded LRU keyed by the complete policy-state identity; cache hits do not incur another hidden-prefill token charge. This serializes mutable model/cache access while allowing the asynchronous graph runtime to await it without blocking the event loop.

## Policy training

Training records are split by run ID. The policy predicts route, edge, stop, completion value, and normalized cost. Invalid labels are represented as `-1`; edge and stop losses use the recorded hard masks. The trainer uses AdamW and optional early stopping against held-out runs.

Policy files are bound to graph schema, model fingerprint, hidden extractor schema, and feature dimensions. Resume validation prevents silent policy substitution.

## Same-run exclusion and time semantics

The SQLite runtime holds a non-blocking per-run file lease for the full run or resume call. Individual workspace mutation and test operations also hold workspace locks. Active-time budgets accumulate execution duration only; paused wall-clock time does not consume `max_seconds`.

## Promotion boundary

A successful run exports a cumulative patch from the base commit to the final worktree. Promotion is separate from graph execution and requires completed status, matching base commit, clean source checkout, matching patch SHA-256, revalidation, and idempotent application. The graph cannot promote itself.
