# Counterbalanced policy evaluation v1 methodology

## Purpose

This evaluation is the first larger held-out comparison authorized by paired policy canary v2. It compares the deterministic static controller against the full learned-policy candidate `855e378570a9` without changing the global runtime configuration.

The evaluation measures whether the learned controller produces a safety or correctness regression and whether it demonstrates a material success or cost advantage. It does not automatically activate or promote the candidate.

## Prerequisite

The runner requires a completed paired-canary v2 report with:

- `gate_passed: true`;
- exact static-versus-shadow equivalence;
- the same candidate weight SHA-256.

This anchors the larger comparison to an already-qualified causal harness.

## Corpus

The generator creates 24 new, committed Python repositories:

| Category | Count | Purpose |
|---|---:|---|
| Fast repairs | 6 | Small, bounded single-file corrections |
| Deep or multi-file | 6 | Shared-policy and shared-parser migrations |
| Repair-oriented | 6 | Tasks with multiple behavioral edge cases |
| No-change audits | 3 | Correct repositories that must not be modified |
| Impossible contracts | 3 | Contradictory tests that must terminate through bounded abort |

Every completed-case fixture includes an authoritative declarative oracle. Oracles verify deterministic test success, unchanged tests, final newlines, allowed changed files, and selected Python call relationships where the task requires a shared implementation.

Impossible contracts intentionally omit a completion oracle and are successful only when the graph reaches its bounded abort terminal without producing a verified patch.

## Counterbalancing

The 24 cases are divided evenly:

- 12 cases run static first, then full;
- 12 cases run full first, then static.

The runner executes four blocks while retaining separate static and full SQLite databases:

1. static on the static-first half;
2. full on the static-first half;
3. full on the full-first half;
4. static on the full-first half.

Both arms use identical run IDs in separate databases, canonical repository aliases, temperature zero, prompt-derived deterministic seeds, the same generation budget, and the same verifier commands.

## Arms

### Static

- hardcoded route and transition priors;
- policy weights and configuration unset;
- hidden capture enabled for audit parity;
- forced transitions skipped by the policy machinery.

### Full

- candidate `855e378570a9` loaded temporarily;
- route policy scale `1`;
- transition policy scale `1`;
- hard graph masks remain authoritative;
- forced transitions skipped.

The candidate is never written to `.graph-env`.

## Primary metrics

- verified expected outcomes;
- false successes;
- bounded-failure correctness;
- total tokens;
- LLM calls;
- repair count;
- graph steps.

Tool calls, policy calls, routes, meaningful choice changes, prompt hashes, and active seconds are also recorded. Active-time differences are exploratory because block order and model-cache state can affect latency.

## Safety gate

The safety gate requires:

- the paired-canary prerequisite remains valid;
- exactly 24 cases and 12/12 order balance;
- all fixture baselines were validated;
- no missing runs or collector errors;
- zero false successes in both arms;
- full verified outcomes are not below static;
- no category has a completion regression;
- impossible-contract correctness does not regress;
- generation prompt hashes match whenever both arms reach the same route and graph path;
- no raw prompt persistence;
- all source repositories remain clean and at their original commits.

A safety failure rejects the candidate for promotion.

## Benefit gate

A benefit is demonstrated when either:

1. full obtains at least one additional correct expected outcome; or
2. verified outcomes are equal, at least one primary cost metric improves by 10% or more, and no primary cost metric regresses by 10% or more.

The comparator emits one of three statuses:

- `rejected-safety-regression`;
- `safe-but-no-demonstrated-benefit`;
- `eligible-for-guarded-shadow-rollout`.

Even the third status requires human review and does not activate the candidate.

## Evidence

The evaluation writes:

- two manifests and a case contract bundle;
- separate static and full run databases;
- block summaries, progress logs, worktrees, artifacts, and hidden-state stores;
- `counterbalanced-eval-report.json`;
- `counterbalanced-eval-report.md`.

The latest completed evaluation root is recorded in:

```text
.graph-model/evaluation/LATEST_COUNTERBALANCED_POLICY_EVAL
```
