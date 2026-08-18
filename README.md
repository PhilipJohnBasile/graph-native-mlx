# Graph-Native MLX Model

A graph-controlled AI model runtime for Apple Silicon. Version 0.2 moves both language generation and graph-policy decisions into MLX while keeping durability, tool effects, permissions, and checkpoints outside the tensor graph.

The central rule is:

> The graph is the control plane. A loop is only a named, bounded, observable back-edge.

This is not a new foundation-model pretraining run. It turns an existing MLX language model into a compound model with:

- a direct, in-process `mlx_lm` provider
- immutable compiled node and edge tables
- hard edge masks evaluated with MLX tensors
- route, edge, stop, success-value, and cost sidecar heads
- an external durable executor that retains authority over tools and terminal status

## Implemented in v0.2

- An 11-node, 16-edge typed coding supergraph
- Generated Python graph tables with a deterministic SHA-256 schema hash
- Per-node allowed-edge matrices and traversal limits
- Runtime condition filtering before every controller decision
- MLX-native masked softmax and argmax for route, stop, and edge selection
- Optional trainable MLX sidecar heads loaded from Safetensors
- A direct `MLXLocalProvider` using `mlx_lm.load` and `stream_generate`
- One resident model/tokenizer per provider instance
- Chat-template rendering with an explicit role-delimited fallback
- Robust extraction of the final complete JSON object from model output
- SQLite checkpoints and append-only execution traces
- Provider and controller identity checks on resume
- Stable idempotency keys for side-effecting operations
- Bounded plan revision and diagnose/repair cycles
- Explicit finish and abort terminals
- MLX policy trace export and a multi-task MLX trainer
- Graph-versus-retry-loop benchmark harness
- Optional Restate production adapter

## Architecture

```text
                         MLX language backbone
                  planner / executor / semantic reviewer
                                   │
                                   ▼
                         typed node result JSON
                                   │
                                   ▼
                condition-valid, non-exhausted edge set
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
          generated graph tables        explicit state features
          node/edge IDs and masks                 │
                    │                             ▼
                    │                 MLX route/edge/stop/value/cost
                    │                       sidecar policy heads
                    └──────────────┬──────────────┘
                                   ▼
                         hard MLX tensor mask
                                   │
                                   ▼
                       selected valid transition
                                   │
                                   ▼
                  SQLite / Restate durable executor
                   tools, tests, effects, checkpoint
```

A policy can add logits to allowed choices. It cannot add a node, invent an edge, bypass a verifier, exceed an edge traversal cap, or mark a run complete.

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
   │                     plan_check ──fail──► revise plan (once)
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

Exhausted failure paths ─────────────────────────► abort
```

## Install on the M5 Max

```bash
unzip graph-native-model-mlx-v0.2.0.zip
cd graph-native-model-mlx-v0.2.0

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,mlx]'

pytest
```

The base package can be installed and tested on non-MLX machines. The `mlx` extra is intended for Apple Silicon.

### Configure the direct model

Set `GRAPH_MODEL_MLX_MODEL` to a local MLX model directory or a Hugging Face repository ID. For a Hub model, pin `GRAPH_MODEL_MLX_REVISION` to a commit hash after the first successful load:

```bash
export GRAPH_MODEL_MLX_MODEL='AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP'
# Recommended for reproducibility: set a Hugging Face commit hash.
export GRAPH_MODEL_MLX_REVISION='<commit-sha>'
export GRAPH_MODEL_MLX_MAX_TOKENS=2048
export GRAPH_MODEL_MLX_TEMPERATURE=0.1
export GRAPH_MODEL_MLX_TOP_P=1.0
export GRAPH_MODEL_MLX_MIN_P=0.0
export GRAPH_MODEL_MLX_TOP_K=0
```

For a local download:

```bash
export GRAPH_MODEL_MLX_MODEL="$HOME/.cache/huggingface/hub/<model-directory>"
```

Model architecture support comes from the installed `mlx-lm` version. When a model requires changes newer than the latest packaged release, install the current upstream code in the same environment:

```bash
python -m pip install --upgrade 'mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git'
```

Check the environment and optionally perform a real model load:

```bash
graph-model mlx-doctor
graph-model mlx-doctor --load-model
```

## Run the MLX-native graph

`--provider mlx` automatically selects the MLX graph controller:

```bash
graph-model run \
  --provider mlx \
  --run-id first-mlx-run \
  --task 'Inspect the repository, implement the requested production change, run verification, and report exact evidence.'
```

The model is loaded once and remains resident for all node calls made through that provider instance. Generation runs in-process rather than through an OpenAI-compatible HTTP endpoint.

Inspect the recorded route, stop, masks, and edge decision:

```bash
graph-model trace --run-id first-mlx-run
```

Each non-terminal node event records:

- the stop action and probability distribution
- allowed stop actions
- the selected edge and probability distribution
- every currently allowed edge key
- the graph schema hash
- the explicit policy feature vector
- sigmoid success probability and positive cost predictions when trained weights are active

## Interruption and resume

```bash
graph-model run \
  --provider mlx \
  --run-id checkpoint-demo \
  --stop-after-steps 3 \
  --task 'Implement a production feature with tests'

graph-model resume \
  --provider mlx \
  --run-id checkpoint-demo
```

Resume requires the same graph, configured provider/model identity, adapter identity, controller, policy-file fingerprints, and policy scale. This prevents a partially completed run from silently changing its execution policy. Pin remote model revisions separately for fully reproducible production runs.

## Compile the graph into constants

The bundled graph is already compiled into `generated_coding_graph.py`.

Regenerate it after intentionally changing the YAML:

```bash
graph-model compile-graph \
  --graph src/graph_model/graphs/coding_supergraph.yaml \
  --output src/graph_model/mlx_native/generated_coding_graph.py
```

The generated module contains:

- stable integer node IDs
- stable integer edge IDs
- source and target arrays
- edge priorities
- edge traversal limits
- condition strings
- per-node allowed-edge masks
- terminal masks
- a graph schema hash

At runtime, predicates and traversal counts can only remove entries from this structural mask. They cannot add entries.

## MLX policy heads

The sidecar network receives explicit task, budget, verifier, artifact, node, and traversal features. It emits residual logits and estimates:

```text
route_logits:  fast / deep / repair
edge_logits:   one logit per compiled edge
stop_logits:   continue / repair / finish / abort
value:         estimated probability of successful completion
cost:          predicted token / latency / tool-call cost
```

The sidecar is deliberately separate from the 27B backbone weights:

```text
model directory/
  model-*.safetensors
  config.json
  tokenizer.json

policy directory/
  graph_policy.safetensors
  graph_policy.json
  training_summary.json
```

Activate trained policy weights:

```bash
export GRAPH_MODEL_MLX_POLICY_WEIGHTS="$PWD/models/graph-policy-v1/graph_policy.safetensors"
export GRAPH_MODEL_MLX_POLICY_CONFIG="$PWD/models/graph-policy-v1/graph_policy.json"
export GRAPH_MODEL_MLX_POLICY_SCALE=1.0
```

The config is bound to the graph schema hash. Loading weights trained for a different graph fails before execution.

## Collect and train policy traces

First run real tasks using the MLX controller. The initial controller uses hardcoded priors while recording the exact feature vectors and masks required for training.

Export decision records:

```bash
graph-model export-mlx-policy \
  --output data/mlx-policy-traces.jsonl
```

Restrict the first behavioral-cloning dataset to successful runs:

```bash
graph-model export-mlx-policy \
  --success-only \
  --output data/mlx-policy-success.jsonl
```

Train the sidecar on the Mac:

```bash
graph-model train-mlx-policy \
  --input data/mlx-policy-success.jsonl \
  --output-dir models/graph-policy-v1 \
  --hidden-size 128 \
  --epochs 100 \
  --learning-rate 0.001
```

The trainer uses a multi-task objective over route, masked edge, masked stop, success value, and normalized cost targets. Successful behavioral cloning teaches the policy to reproduce validated decisions. To make it smarter than the initial rules, supply traces selected by held-out quality, graph search, preference ranking, or human review rather than indiscriminately cloning every run.

## Phase boundary: what is and is not inside MLX

Inside MLX:

- local language-model inference
- policy feature tensors
- route, edge, stop, value, and cost heads
- structural and runtime edge masks
- softmax and argmax selection
- policy weight loading

Outside MLX:

- SQLite or Restate journaling
- shell commands and test processes
- repository and filesystem mutation
- GitHub, email, database, and deployment effects
- secrets and permissions
- idempotency records
- human approval gates
- final authority to commit a transition

A tensor computation cannot be the transaction log for an external side effect. The executor therefore commits evidence and state after each node and resumes from the first unfinished node.

## MTP and MTPLX

Version 0.2 intentionally uses normal `mlx_lm.stream_generate` for node generation. It does not enable a draft model or an MTP-specific decode path yet.

This isolates two independent questions:

1. Does graph-controlled execution improve correctness, cost, and stopping behavior?
2. Does MTP improve token generation speed without changing those decisions?

After the graph baseline passes real repository benchmarks, MTP can accelerate generation inside each LLM node. It should not own graph transitions, checkpoint semantics, or side-effect replay.

## Compare against the retry-from-the-top baseline

```bash
graph-model benchmark \
  --provider mlx \
  --controller mlx \
  --input examples/benchmark_tasks.jsonl \
  --output examples/benchmark_report.mlx.json
```

The included loop baseline is intentionally bounded. It rebuilds planning and execution on every attempt, whereas the graph preserves successful upstream state and repairs only the failed region.

Mock mode validates control flow, not intelligence:

```bash
graph-model benchmark \
  --input examples/benchmark_tasks.jsonl \
  --output examples/benchmark_report.mock.json
```

A meaningful model result requires held-out tasks, deterministic evaluators, repeated seeds, matched budgets, and identical tool environments.

## Existing HTTP provider

The original OpenAI-compatible path remains available for MTPLX, oMLX, LM Studio, or another server:

```bash
export GRAPH_MODEL_BASE_URL=http://127.0.0.1:8000/v1
export GRAPH_MODEL_NAME=local-model
export GRAPH_MODEL_API_KEY=local

graph-model run \
  --provider openai \
  --controller deterministic \
  --task 'Implement a production feature with verification'
```

This is useful for A/B testing direct MLX against an optimized serving engine.

## Production durability with Restate

SQLite demonstrates checkpoint and replay semantics locally. External effects must still accept the stable idempotency keys supplied by the runtime.

For a journaled production workflow:

```bash
python -m pip install -e '.[restate]'
restate-server
python -m graph_model.integrations.restate_service
restate deployments register http://localhost:9080
```

## Test status

Run all portable tests:

```bash
pytest
python -m compileall -q src tests
```

The v0.2.0 release contains 41 portable tests. The suite uses a numerically equivalent Python decision backend and an injected fake MLX-LM backend. This verifies masks, routing, trace content, provider residency, prompt rendering, JSON parsing, policy data export, resume guards, graph compilation, and packaging without requiring a 27B model in CI.

Use `graph-model mlx-doctor --load-model` on the target Mac for the final hardware/model compatibility check.

See `ARCHITECTURE.md` and `MLX_NATIVE.md` for design details.
