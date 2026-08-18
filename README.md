# Graph-Native Model Starter

A working starter for a graph-controlled AI system that uses a local or hosted language model as its reasoning backbone.

The central rule is:

> The graph is the control plane. A loop is only one possible edge, and every cyclic edge must be named, bounded, observable, and able to terminate explicitly.

This is the first practical layer of building an AI model around **graph policy rather than retry-until-success behavior**. It is not a new foundation-model pretraining run. It is a compound model with a typed workflow policy, a trainable router, verifier gates, durable state, and an offline graph-search hook. Winning traces can later be distilled into adapters or new model heads.

## What is implemented

- A typed, versioned YAML supergraph
- Fast, planned, and repair paths selected per task
- A trainable constrained route model with deterministic fallback
- Independent deterministic and semantic verifier nodes
- Bounded plan revision and diagnose/repair cycles
- Explicit abort behavior after the repair budget is exhausted
- SQLite checkpoints and an append-only execution trace
- Resume from the next unfinished node after interruption
- Stable idempotency keys for side-effecting nodes
- An OpenAI-compatible provider for MTPLX, oMLX, vLLM Metal, LM Studio, or another compatible endpoint
- Router-training trace export
- An AFlow-style constrained MCTS search scaffold
- A graph-versus-Ralph-loop benchmark harness
- An optional Restate workflow adapter for production durable execution

## Default coding graph

```text
intake/router
      │
      ▼
   context
   ┌──┴─────────────────────┐
   │ fast                    │ deep or repair
   ▼                         ▼
implement                  plan
   │                         │
   │                     plan_check ──fail──► revise plan (bounded)
   │                         │ pass
   └──────────────┬──────────┘
                  ▼
                tests ──fail──► diagnose ─► repair ─┐
                  │                                 │
                  │ pass                            └──► tests (bounded)
                  ▼
                review ──fail──► diagnose/repair
                  │ pass
                  ▼
                finish

Any exhausted failure path goes to abort.
```

A successful `context` or `plan` node is not rerun merely because a later verifier rejects the candidate.

## Install and test

```bash
cd graph-native-model-starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest

graph-model validate
```

## Run the proof of concept

```bash
# Fast route
python -m graph_model run \
  --task "quick fix: rename one variable"

# Planned route with one intentional test failure and one local repair
python -m graph_model run \
  --task "fix failing CI [force-fail-once]" \
  --run-id demo-repair

# Inspect every checkpoint, decision, and state delta
python -m graph_model trace --run-id demo-repair
```

The bracketed `force-fail` markers are only deterministic test fixtures for the starter.

## Compare the graph with a retry loop

The included baseline repeatedly runs `plan → implement → test → retry from the top`. It is capped at three attempts so the benchmark itself cannot become an unbounded loop.

```bash
python -m graph_model benchmark \
  --input examples/benchmark_tasks.jsonl \
  --output examples/benchmark_report.mock.json
```

Mock mode tests routing, reuse, stopping, and call economics. It does **not** prove that one system is more intelligent. For a real claim, replace the mock provider with your local model and use held-out coding tasks with deterministic tests.

## Connect the local Qwen/MLX backbone

```bash
export GRAPH_MODEL_BASE_URL=http://127.0.0.1:8000/v1
export GRAPH_MODEL_NAME=Qwen3.8-27B-JANG_4L
export GRAPH_MODEL_API_KEY=local
export GRAPH_MODEL_TIMEOUT_SECONDS=300

python -m graph_model run \
  --provider openai \
  --task "Implement a production feature with tests and an explicit migration plan"
```

The planner, executor, and semantic reviewer can initially share one backbone. They receive distinct node instructions and cannot choose arbitrary transitions or certify their own output.

## Train the route policy

First accumulate real graph runs, then export them:

```bash
python -m graph_model export \
  --output data/router-traces.jsonl
```

Train the included hashed linear softmax controller:

```bash
python -m graph_model train-router \
  --input data/router-traces.jsonl \
  --output models/router-v1.json \
  --success-only \
  --epochs 60

python -m graph_model predict-route \
  --model models/router-v1.json \
  --task "fix a failing GitHub Actions regression"
```

Activate it for normal runs:

```bash
export GRAPH_MODEL_ROUTER_PATH=$PWD/models/router-v1.json
export GRAPH_MODEL_ROUTER_MIN_CONFIDENCE=0.60
```

The learned policy can choose only `fast`, `deep`, or `repair`. Low-confidence predictions fall back to the deterministic router, and the graph still validates every transition.

## Demonstrate interruption and resume

```bash
python -m graph_model run \
  --run-id checkpoint-demo \
  --stop-after-steps 3 \
  --task "Implement a production feature"

python -m graph_model resume --run-id checkpoint-demo
```

The resumed run begins at the checkpointed next node. It does not rerun `intake`, `context`, or `plan`. Resume with the same provider and model identity used by the original run; for an OpenAI-compatible run, include `--provider openai` and keep the endpoint/model environment variables unchanged.

## Offline graph optimization

`graph_model.optimizer.mcts_optimize` searches a constrained mutation library rather than letting an optimizer emit arbitrary Python. Candidate graphs are parsed into the typed schema and rejected when invalid.

Useful mutations include:

- changing node prompts, adapters, or model tiers
- inserting or removing optional verifier regions
- adjusting bounded traversal limits
- pruning redundant paths
- changing decomposition or parallelization motifs

The evaluator should run held-out tasks and score quality, token cost, latency, tool calls, and unsafe or duplicate side effects.

## Production durability with Restate

The SQLite runner demonstrates graph semantics and local checkpointing. It cannot prevent a network side effect from being repeated if the process dies after the external system accepts the request but before SQLite commits. External tools therefore receive stable idempotency keys.

For journaled production execution, the optional adapter wraps each node in a Restate durable step:

```bash
brew install restatedev/tap/restate-server restatedev/tap/restate
python -m pip install -e '.[restate]'

restate-server
python -m graph_model.integrations.restate_service
restate deployments register http://localhost:9080
```

The adapter uses deterministic Restate time, journals completed node results, and replays those results instead of reissuing completed model or tool calls after a crash.

## What makes this a model project

The external graph is Stage 1, not the endpoint.

1. **Node policy:** improve planner, executor, verifier, retrieval, and repair behavior using role prompts and adapters.
2. **Routing policy:** train the included controller, then replace it with a small neural route/cost/stop head.
3. **Graph policy:** search validated graph variants offline and permit only bounded local region replacement during inference.
4. **Distillation:** construct preference pairs from winning and losing traces and LoRA-train the Qwen backbone to emit typed state transitions and calibrated stop/escalate decisions.
5. **Authority remains external:** even after distillation, a model prediction is a proposal. The runtime owns allowed edges, budgets, side effects, and terminal status.

See `ARCHITECTURE.md` for the full design and `REFERENCES.md` for the research basis.
