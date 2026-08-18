# Graph-Native Model Architecture

## Thesis

A loop is not inherently wrong. An unbounded loop with implicit state, self-judged progress, replayed side effects, and no terminal policy is wrong.

The system therefore treats the workflow as a versioned executable model:

- **nodes** are typed operations
- **edges** are allowed transitions
- **state** is explicit and checkpointed
- **verifiers** produce evidence rather than vague reflection
- **cycles** have traversal limits, progress checks, and abort paths
- **the language model proposes work but does not own control flow**

The correct amount of dynamism is the minimum required by the workload. Start with a validated scaffold, route or prune per task, generate a new graph only when known paths are structurally insufficient, and edit only a bounded failed region when runtime evidence requires it.

## Three learning layers

1. **Node policy** — how a planner, executor, verifier, retriever, or repair node behaves.
2. **Routing policy** — which validated path and model tier a task should use.
3. **Graph policy** — which nodes and edges should exist for a task family.

The first release externalizes all three so they are inspectable. Later releases train the route policy and distill winning graph traces into the backbone.

## Production shape

```text
request
  │
  ▼
feature extraction ──► constrained route policy
  │                              │
  │                              ▼
  └────────────────────► versioned supergraph
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
                fast path    planned path   repair path
                    │            │            │
                    └────────────┴─────► verifier gates
                                           │
                               pass ────────┴────── fail
                                │                    │
                                ▼                    ▼
                              finish          bounded local graft
```

A loop is represented as a back-edge in this graph. It is never an invisible `while not done` instruction delegated to the model.

## Typed run state

Each run records:

- run ID, graph name, and immutable graph version
- task, constraints, route, and difficulty estimate
- current node and completed-node path
- artifacts, evidence, and verifier decisions
- repair and edge-traversal counts
- model-call, tool-call, token, time, and step budgets
- a repeated-progress signature
- stable idempotency keys for external effects
- model, prompt, adapter, and tool-schema versions when integrated

A node returns a `NodeResult` containing a state delta, artifacts, optional terminal output, verdict, progress key, token accounting, and notes. It cannot directly jump to another node. The runtime alone selects a matching validated edge.

## Initial coding supergraph

The default graph supports three task-level routes:

- **fast:** context → implement → tests → semantic review
- **deep:** context → plan → plan verifier → implement → tests → semantic review
- **repair:** the planned path, followed by a bounded diagnose/repair region when evidence rejects the candidate

The plan can be revised once. Candidate repair can traverse its back-edge twice. When the limit is exhausted, the run reaches an explicit failed terminal rather than continuing until a model happens to say it is done.

## Verifier ordering

Cheap, deterministic verification comes before expensive semantic review:

1. schema and output-shape checks
2. compilation, unit tests, static analysis, or domain constraints
3. independent semantic review
4. human approval only for configured high-risk effects

The executor cannot mark its own work as verified. Verifier failure must include evidence that a diagnose node can transform into targeted repair state.

## Durability model

The local runner uses SQLite for atomic state/checkpoint commits and stable node input hashes. This supports local resume and avoids redoing a committed node.

There remains an unavoidable commit gap around external effects: a process can die after a remote system accepts a request but before the local transaction records success. Side-effecting tools must therefore accept idempotency keys.

The Restate adapter closes more of this gap by journaling each node action and replaying its result during recovery. The graph loop uses deterministic journaled time rather than process-local wall-clock time.

## Trainable routing controller

The included route model is a constrained softmax classifier with stable hashed text features. It exists to establish the full learning loop without requiring another large dependency:

```text
(task text) -> P(fast), P(deep), P(repair)
```

It is trained from exported production traces. A confidence gate retains the deterministic fallback. The controller cannot generate node IDs or edges outside the graph.

The next controller should add visible-state and budget features:

```text
(task, state summary, allowed edges, remaining budget) -> selected edge/model tier
```

## Model integration roadmap

### Stage 1 — Graph-controlled compound model

Use the existing Qwen3.8-27B MLX model as the shared planner, executor, and semantic verifier. Keep routing, checkpointing, and hard verification external. This stage is implemented in the starter.

### Stage 2 — Learned route, cost, and stop policies

Train:

- a route head over allowed next edges
- a success/progress head estimating whether another step has positive expected value
- a cost head predicting tokens, latency, and tool calls
- an escalation head predicting when a stronger model or human review is justified

A practical objective is:

```text
L = L_text
  + α CE(route)
  + β BCE(success)
  + γ Huber(cost)
  + δ pairwise_rank(winning_trace, losing_trace)
  + ε consistency(paraphrase_routes)
```

### Stage 3 — Graph-aware distillation

Use offline graph search to produce positive and negative execution traces. LoRA/QLoRA-train the backbone to:

- emit concise typed node outputs
- preserve evidence and constraints across nodes
- choose calibrated stop/escalate proposals
- distinguish executor claims from verifier evidence
- avoid requesting a back-edge when progress is unchanged

The external graph remains authoritative after distillation.

## Offline graph optimization

Search is performed against held-out tasks, not unrestricted live production requests. Candidate mutations may:

- insert or remove an optional verifier region
- swap a node prompt, adapter, model, or decoding policy
- adjust a bounded traversal limit
- alter decomposition or parallelization motifs
- bypass redundant nodes

Score candidates with a cost-aware objective:

```text
reward = task_quality
       - λ_token * tokens
       - λ_time * latency
       - λ_tool * tool_calls
       - λ_risk * unsafe_or_duplicate_effects
```

Promote a graph only after regression tests, paraphrase stability, tool-failure injection, schema-drift testing, and side-effect deduplication tests.

## Inference-time local grafting

Do not regenerate an entire graph for every request. Start with a globally validated supergraph and select a subgraph. When execution evidence proves that one region failed, replace only a single-entry/single-exit region with another compatible region from a validated library.

A local graft must satisfy:

- identical typed boundary inputs and outputs
- a strict search and token budget
- verifier improvement above a configured margin
- no degradation of evidence support across the boundary
- a cache key based on failure and task signature
- an edit cap and explicit fallback

Successful upstream state remains checkpointed.

## Graph-versus-loop evaluation

The project includes a bounded retry-from-the-top baseline. The benchmark must report more than final answer quality:

- success or deterministic evaluator score
- cost per successful task
- LLM and tool calls
- token and wall-clock cost
- path length and repeated upstream nodes
- duplicate external effects
- crash/restart recovery
- verifier false accepts and false rejects
- route regret versus the best validated path
- path stability under paraphrase
- no-progress and budget termination rates
- performance under tool failure and schema drift

Mock-provider benchmarks validate control flow only. Capability claims require a real model, held-out tasks, stable evaluators, repeated seeds, and matched budgets.

## Immediate build sequence for MTPLX/oMLX

1. Point the provider at the current local Qwen3.8 endpoint.
2. Replace simulated `run_tests` with a sandboxed repository/test operator.
3. Add repository retrieval and patch-application nodes with idempotent tool contracts.
4. Run the graph and retry-loop baseline on the same internal coding task set.
5. Train the route policy on winning traces and measure route regret.
6. Add constrained MCTS mutations for prompt, verifier placement, model tier, and repair depth.
7. Promote only graph versions that improve held-out quality at equal or lower cost.
8. Distill winning node traces into role adapters on the local backbone.
