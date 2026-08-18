# Architecture

## Design objective

Replace an open-ended agent retry loop with a durable, inspectable, graph-controlled system in which every operation has a name, state contract, allowed successors, retry cap, evidence boundary, and checkpoint.

The system is split into three planes:

```text
Model plane      Qwen/MLX generates typed proposals and hidden features
Policy plane     validated graph + hard priors + optional learned MLX heads
Effect plane     Git, tests, journals, permissions, promotion, cleanup
```

The model and learned policy are advisory. The effect plane remains authoritative.

## Default graph

v0.5.2 retains the 12-node, 19-edge repository graph introduced in v0.3.

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

Repairs return to `apply`, not directly to `tests`. Every proposed mutation therefore crosses the same validation and idempotency gate.

## Repository transaction

```text
model JSON patch proposal
          │
          ▼
normalize fenced/plain diff
          │
          ▼
validate bytes, file count, path policy, and diff-header consistency
          │
          ▼
write immutable patch artifact and operation intent
          │
          ▼
git apply --check
          │
          ▼
apply in detached worktree
          │
          ▼
verify changed paths and workspace fingerprint
          │
          ▼
write committed ledger
```

A post-mutation validation failure reverses the patch and verifies restoration of the original fingerprint. If the process stops after Git applies a patch but before the commit ledger is written, recovery recognizes the matching operation and completes the ledger without applying the patch twice.

## Worktree isolation

A new repository run requires a clean Git top-level checkout. The base reference is resolved to an immutable commit before a detached worktree is created.

The deterministic worktree path uses:

```text
workspace-home / sha256(source-path) / sanitized-run-id-plus-hash
```

The run ID cannot escape the configured root. Existing worktrees are accepted only when they belong to the same Git common directory and remain pinned to the expected commit.

## Verifier integrity

Verifier commands run without a shell. The runtime:

1. Parses the command with `shlex`.
2. Rejects control operators and substitutions.
3. Requires an allowlisted executable.
4. Resolves non-local executables through a sanitized `PATH` that excludes the repository.
5. Restricts Git to read-only subcommands.
6. Executes in a new process group with no stdin.
7. Applies a timeout and bounded stdout/stderr capture.
8. Stops on the first failed command.
9. Compares workspace fingerprints before and after verification.

A command sequence that exits zero but changes tracked source is still a failed verifier. The verifier is not a hostile-code sandbox.

## Durable state

`RunState` stores:

- graph name and version
- task and current node
- explicit data and artifacts
- node attempts and edge traversal counts
- completed-node history
- model/tool/token/active-time metrics
- budgets and no-progress state
- provider and controller identity
- output and terminal error

SQLite commits each node result and resulting checkpoint in one transaction. An append-only event row records route, stop, edge, masks, and policy-deployment metadata.

Side-effecting nodes do not use ordinary result caching. They implement explicit idempotency boundaries.

## Same-run exclusion

The local runtime holds a non-blocking per-run file lease for the full `run` or `resume` call. This prevents concurrent model calls and checkpoint races for one run. POSIX releases the lease automatically on process death.

Workspace mutation and verifier operations also hold a per-workspace lock.

## Time semantics

The budget records accumulated active execution duration per node. A paused run does not consume `max_seconds` while idle. Audit timestamps still preserve wall-clock start and update times.

## Transition authority

For every non-terminal node:

1. Evaluate edge predicates from explicit state.
2. Remove edges that exhausted traversal limits.
3. Compute the hardcoded baseline over surviving choices.
4. Optionally compute a learned candidate under the same mask.
5. Apply shadow, guarded, or active deployment semantics.
6. Validate the selected edge against the candidate set again.
7. Commit the node result and transition atomically.

A controller returning a masked or invented edge raises a runtime error.

## Learned-policy deployment

The controller maintains two choices:

```text
baseline  hardcoded graph prior
candidate learned policy under the same legal mask
```

- `shadow`: execute baseline; record candidate.
- `guarded`: apply candidate only when success, confidence, and margin gates pass.
- `active`: apply every masked candidate.

Deployment details are stored with each decision. The audit layer aggregates coverage, disagreements, applied choices, confidence, margins, and fallback reasons. A shadow candidate is counterfactual only; its outcome was not observed.

## Hidden-state policy inputs

The optional policy combines:

- 62 explicit task, budget, progress, repository, verifier, and node features
- a fixed-size projected Qwen representation, 256 values by default

Two scopes are available:

```text
task      one representation for the task/model/extractor tuple
decision  representation includes bounded current graph state and evidence
```

Decision context is deterministic, depth- and size-bounded, and redacts known secret-bearing keys. Raw prompts and raw hidden tensors are not persisted. The projected artifact is hash-addressed and bound to the model fingerprint and extractor schema.

The policy predicts route, edge, stop, success value, and normalized costs through gated residual fusion. It cannot unmask an edge, bypass a verifier, increase a traversal limit, or commit an effect.

## Policy identity and resume

Controller identity includes:

- graph schema hash
- decision backend
- policy/config/weight identity
- policy mode, scale, prior weight, and guard thresholds
- hidden extractor identity and scope

A resumed run must match its stored provider and controller identity. Changing model, policy weights, hidden scope, or deployment mode requires a new run rather than silently changing semantics mid-execution.

## Qualification boundary

The M5 qualification command checks platform, immutable model identity, model load, capability discovery, hidden-feature determinism, repeated-generation behavior, and MLX memory counters. It writes a content-hashed report.

An exposed `mtp_forward` method is treated as capability only. The direct v0.5.2 provider uses ordinary `mlx_lm.stream_generate` and reports MTP activation separately.

## Promotion boundary

A successful run exports a cumulative patch from the base commit to the final worktree. Promotion is separate from graph execution and requires:

- completed status
- matching base commit in the source checkout
- clean source worktree
- matching patch SHA-256
- successful revalidation and idempotent application

The graph cannot promote itself.
