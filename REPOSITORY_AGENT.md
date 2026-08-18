# Repository Agent

## End-to-end flow

```bash
# 1. Validate the graph and Mac environment
graph-model validate
graph-model mlx-doctor --load-model

# 2. Run in a detached worktree
graph-model run \
  --provider mlx \
  --run-id issue-123 \
  --repo /absolute/path/to/repository \
  --task 'Fix issue 123, add focused regression coverage, and verify the result.'

# 3. Inspect evidence
graph-model trace --run-id issue-123

# 4. Promote only after review
graph-model apply-result --run-id issue-123

# 5. Remove the detached worktree
graph-model cleanup --run-id issue-123
```

## Preconditions

- `--repo` must point to the Git top level.
- The source checkout must be clean.
- `--base-ref` must resolve to a commit.
- The workspace and artifact roots must be outside the source repository.
- The repository and configured test commands must be trusted; tests are not OS-sandboxed.

## Context collection

The context node records:

- base commit, branch, status, and workspace fingerprint
- bounded file tree
- language profile
- task-ranked file excerpts
- detected verifier commands
- patch and command limits

Large files, binaries, caches, build outputs, vendor directories, and node modules are excluded. Context is bounded by file count, per-file bytes, and total bytes.

## Patch contract

Patch-generating nodes return one JSON object:

```json
{
  "summary": "What the patch changes",
  "patch": "diff --git a/... b/...\n...",
  "assumptions": [],
  "no_changes_needed": false
}
```

The patch must be a complete text unified diff against the current worktree. The model must not claim tests passed.

## Verification

`git diff --check` always runs first. Additional commands are auto-detected or supplied with repeated `--test-command` arguments.

Auto-detection covers common Python, Node, Rust, Go, Swift, and existing CMake build layouts. A `package.json` only triggers a package-manager test command when it contains an actual `scripts.test` entry.

The verifier captures exact argv, exit code, duration, timeout state, bounded stdout/stderr, and workspace fingerprints.

## Repair semantics

A failure does not restart the task. The graph preserves context, plan, and already-applied valid changes.

```text
apply/test/review failure
          │
          ▼
exact evidence + current diff + changed-file context
          │
          ▼
local diagnosis
          │
          ▼
repair diff against current worktree
          │
          ▼
same transactional apply gate
```

The default graph permits two repair proposals. Exhaustion ends at an explicit abort node.

## Artifacts

By default, artifacts are stored outside the repository under:

```text
~/.graph-model/artifacts/<sanitized-run-id-hash>/
```

They include patch proposals, apply reports, test reports, diagnoses, reviews, operation intent/commit ledgers, and the final verified or failed cumulative patch.

The SQLite database defaults to `.graph-model/runs.sqlite3` relative to the command’s current directory. Set `GRAPH_MODEL_DB` or `--db` to place it elsewhere.

## Security model

v0.5.2 protects the source checkout and constrains model-proposed effects, but it assumes the repository itself is trusted. A malicious test suite can still access anything available to the current OS user.

For untrusted repositories, run the entire process inside a disposable VM or container with restricted credentials, network, filesystem mounts, and resource limits.
