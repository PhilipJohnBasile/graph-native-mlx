# Graph-Native MLX Architecture

## Thesis

A loop is not inherently wrong. An unbounded loop with implicit state, self-judged progress, replayed side effects, and no terminal policy is wrong.

This system treats the workflow as a versioned executable control model:

- **nodes** are typed operations
- **edges** are the only legal transitions
- **state** is explicit and checkpointed
- **verifiers** produce evidence rather than vague reflection
- **cycles** have traversal limits, progress checks, and abort paths
- **the language model proposes work but does not own control flow**

The graph is the control plane. A loop is a named, bounded, observable back-edge.

## Production shape

```text
request
  │
  ▼
resident Qwen MLX backbone ───────────────► typed node artifact
  │                                                │
  │                                                ▼
  │                                    runtime conditions and limits
  │                                                │
  ▼                                                ▼
explicit task/run-state features ───────► MLX policy residuals
                                                   │
compiled structural graph mask ────────────────────┤
                                                   ▼
                                      hard-masked MLX decision
                                                   │
                                                   ▼
                                      durable executor commits
                                      tools / tests / checkpoint
```

The model and policy may rank valid actions. Only the runtime can commit a transition or terminal status.

## Four learning layers

1. **Node policy** — how a planner, executor, verifier, retriever, or repair node behaves.
2. **Route policy** — which validated task-level path should run.
3. **Transition policy** — which condition-valid edge is best and whether expected progress justifies continuing.
4. **Graph policy** — which nodes and edges should be promoted for a task family after offline evaluation.

Version 0.2 implements the first three control interfaces. Graph-structure optimization remains offline and bounded.

## Typed run state

Each run records:

- run ID, graph name, immutable version, and schema hash
- task, route, difficulty estimate, and current node
- completed-node path and edge-traversal counts
- artifacts, verifier evidence, and repair state
- LLM-call, tool-call, token, time, step, and no-progress budgets
- a repeated-progress signature
- stable idempotency keys for external effects
- provider and controller identities
- policy config and weights fingerprints when a sidecar is active

A node returns a `NodeResult` containing a state delta, artifacts, optional output, verdict, progress key, token accounting, and notes. It cannot name or jump to the next node.

## Compiled graph authority

The editable YAML graph is compiled into immutable Python tables. The compiler emits stable IDs, adjacency rows, traversal limits, terminal masks, and a deterministic SHA-256 over the normalized graph.

Transition selection has two masks:

1. **Structural mask:** an edge must originate at the current node in the compiled graph.
2. **Runtime mask:** its predicate must pass and its traversal count must remain below the cap.

Optional policy logits are added before the hard mask. The runtime validates the selected edge against the candidate set again before committing it. A policy cannot create a node, unmask an edge, bypass a verifier, or raise its own budget.

## Default coding supergraph

The graph supports three routes:

- **fast:** context → implement → tests → semantic review
- **deep:** context → plan → plan verifier → implement → tests → semantic review
- **repair:** planned execution followed by bounded diagnose/repair when evidence rejects the candidate

The plan can be revised within its configured edge cap. Candidate repair is also bounded. Exhaustion reaches an explicit `abort` terminal rather than continuing until a model happens to declare success.

## Verifier ordering

Cheap deterministic verification precedes expensive semantic review:

1. schema and output-shape checks
2. compilation, unit tests, static analysis, or domain constraints
3. independent semantic review
4. human approval for configured high-risk effects

The executor cannot mark its own work as verified. A failed verifier must leave evidence that a diagnose node can convert into targeted repair state.

## MLX-native model boundary

### Inside MLX

- Qwen language generation through MLX-LM
- explicit task/run-state feature tensors
- route, edge, stop, success-value, and cost heads
- hard masking, softmax, and argmax
- policy-weight inference and training

### Outside MLX

- SQLite or Restate journal
- shell commands and test processes
- filesystem, GitHub, database, email, and deployment effects
- secrets, permissions, and human approvals
- idempotency records
- final transition commit

A tensor graph is not a transaction log. External durability is therefore a required part of the model system, not an incidental wrapper.

## Durability model

The local runner writes atomic SQLite checkpoints using canonical node-input hashes. A completed cacheable node can be replayed without re-running model or tool work.

There remains a commit gap around an external operation: a process can fail after a remote system accepts the request but before local success is recorded. Side-effecting tools must accept stable idempotency keys.

The optional Restate adapter journals node actions and replays completed results during recovery. Temporal can serve the same architectural role in another deployment.

## Implemented model roadmap

### Stage 1 — Direct MLX compound model: implemented

- One resident Qwen model/tokenizer via `mlx_lm.load`
- In-process streamed generation
- External typed graph and durable checkpointing
- Deterministic verifier gates

### Stage 2 — MLX graph-policy sidecar: implemented

- Explicit task and run-state features
- Route, edge, stop, value, and cost heads
- MLX hard masks
- Trace export and multi-task trainer
- Graph-schema-bound Safetensors

### Stage 3 — Hidden-state fusion and graph-aware LoRA: next

Add a Qwen-specific, tested hidden-state hook and concatenate selected hidden representations with explicit graph features. Train the controller and a small LoRA to:

- emit concise typed node outputs
- preserve evidence across nodes
- distinguish claims from verifier proof
- predict calibrated progress and cost
- avoid a back-edge when state has not materially changed

A candidate objective is:

```text
L = L_text
  + α CE(route)
  + β CE(masked_edge)
  + γ CE(masked_stop)
  + δ BCE(success)
  + ε Huber(cost)
  + ζ pairwise_rank(winning_trace, losing_trace)
```

The external graph remains authoritative after distillation.

### Stage 4 — Speed and structural optimization

- benchmark MTP/speculative generation inside LLM nodes
- search graph variants offline with a cost-aware evaluator
- promote only validated graph versions
- permit at most one compatible local region graft when runtime evidence proves the current region is structurally insufficient

MTP accelerates token generation. It does not own graph transitions, checkpoints, or side-effect replay.

## Offline graph optimization

Candidate mutations may:

- insert or remove an optional verifier region
- swap a node prompt, adapter, model, or decoding policy
- change a bounded traversal cap
- alter decomposition or parallelization motifs
- bypass a proven redundant node

Score candidates on held-out tasks:

```text
reward = task_quality
       - λ_token × tokens
       - λ_time × latency
       - λ_tool × tool_calls
       - λ_risk × unsafe_or_duplicate_effects
```

Promotion requires regression tests, paraphrase stability, tool-failure injection, schema-drift tests, and duplicate-effect tests.

## Evaluation

The retry-from-the-top baseline is bounded and uses the same provider. Report:

- success or deterministic evaluator score
- cost per successful task
- model and tool calls
- token and wall-clock cost
- path length and repeated upstream nodes
- duplicate external effects
- crash/restart recovery
- verifier false accepts and false rejects
- route regret
- path stability under paraphrase
- no-progress and budget termination rates
- behavior under tool failure and schema drift

Mock-provider results validate control flow only. Model capability claims require real held-out tasks, repeated seeds, stable evaluators, matched budgets, and identical tool environments.
