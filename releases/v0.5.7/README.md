# v0.5.7 — Causal Paired Evaluation

This release adds a controlled four-arm policy experiment and semantic-review adjudication after the inconclusive v1 canary.

## Main additions

- Paired static, shadow, route-only, and full-policy arms
- Canonical model-visible runtime identity
- Hash-only prompt audit artifacts
- Per-prompt deterministic MLX seed resets
- Separate route and transition residual scales
- Forced-choice policy skipping and meaningful-choice telemetry
- Declarative contract oracles over tests, changed files, newlines, and Python AST call relationships
- Explicit authoritative-oracle and independent-appeal adjudication modes
- Shared held-out repository generator, Mac runner, and evidence comparator

The graph schema remains unchanged so candidate `855e378570a9` retains its graph binding. The candidate remains disabled by default.
