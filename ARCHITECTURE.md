# Architecture

## Design objective

Replace an open-ended agent retry loop with a durable, inspectable, graph-controlled system in which every operation has a name, state contract, allowed successors, retry cap, evidence boundary, and checkpoint.

The system is intentionally split into three planes.

```text
Model plane      Qwen/MLX generates typed proposals and semantic judgments
Policy plane     validated graph + optional learned MLX decision heads
Effect plane     Git, tests, journals, permissions, promotion, cleanup
```

The model and learned policy are advisory. The effect plane remains authoritative.

## Default graph

The v0.3 graph contains 12 nodes and 19 edges.

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
validate byte count, file count, path policy, diff-header consistency
          │
          ▼
write immutable patch artifact
          │
          ▼
write operation intent ledger
          │
          ▼
git apply --check
          │
          ▼
apply patch in detached worktree
          │
          ▼
verify declared changed paths and new workspace fingerprint
          │
          ▼
write committed ledger
```

If validation fails after mutation, the runtime reverses the patch and verifies that the original fingerprint was restored. If the process crashes after application but before the committed ledger, the next invocation recognizes the matching intent and reverse-applicability, then commits the recovered operation without applying the patch again.

## Worktree isolation

A new repository run requires a clean Git top-level checkout. The base reference is resolved to an immutable commit before creating a detached worktree.

The deterministic worktree path uses:

```text
workspace-home / sha256(source-path) / sanitized-run-id-plus-hash
```

The run ID cannot escape the configured root. Existing deterministic worktrees are accepted only when they belong to the same Git common directory and remain pinned to the expected commit.

In-place mode exists for controlled use, but detached worktree mode is the default.

## Verifier integrity

Verifier commands run without a shell. The runtime:

1. Parses the command with `shlex`.
2. Rejects control operators and substitutions.
3. Requires an allowlisted executable.
4. Resolves non-local executables through a sanitized `PATH` that excludes the repository.
5. Restricts Git to read-only subcommands.
6. Executes in a new process group with no stdin.
7. Applies a per-command timeout and bounded stdout/stderr capture.
8. Stops on the first failed command.
9. Compares workspace fingerprints before and after the entire verifier sequence.

A command sequence that exits zero but changes tracked source is still a failed verifier result.

The verifier is not a hostile-code sandbox. Repository code runs with local user permissions.

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

SQLite commits each node result and the resulting checkpoint in one transaction. An append-only event row records transition decisions and masks.

Side-effecting nodes are forbidden from using ordinary result caching. They implement their own idempotency boundary instead.

## Same-run exclusion

The local SQLite runtime holds a non-blocking per-run file lease for the full `run` or `resume` call. This prevents two processes from issuing duplicate model calls or racing checkpoint sequence numbers for the same run. POSIX releases the lease automatically on process death.

Individual workspace mutation and test operations also hold a per-workspace lock.

## Time semantics

The local budget records accumulated active execution duration per node. A run paused for hours or days does not consume its `max_seconds` budget while idle. The audit timestamps still preserve wall-clock start and update times.

## Transition authority

For every non-terminal node:

1. Evaluate edge predicates from explicit state.
2. Remove edges that exhausted traversal limits.
3. Ask the controller to rank only the surviving candidates.
4. Apply the hard MLX mask when the MLX controller is used.
5. Validate the returned edge against the candidate set again.
6. Commit the node result and transition atomically.

A controller returning a masked or invented edge raises a runtime error.

## Policy learning

The optional policy network receives 62 values for the default graph:

- 16 task features
- 34 budget, progress, verdict, workspace, patch, apply, test, and review features
- 12 current-node one-hot values

It predicts route, edge, stop, success value, and normalized costs. Policy files are bound to the graph schema hash and included in controller identity, preventing silent sidecar changes on resume.

## Promotion boundary

A successful run exports a cumulative patch from the base commit to the final worktree. Promotion is separate from graph execution and requires:

- completed status
- matching base commit in the source checkout
- clean source worktree
- matching patch SHA-256
- successful revalidation and idempotent application

The graph cannot promote itself.
