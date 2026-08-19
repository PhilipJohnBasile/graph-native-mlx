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

## Portable validation

- 135 tests passed in isolated source-checkout groups on Linux/Python 3.13
- source archive import and graph validation passed
- wheel direct import and CLI graph validation passed
- graph schema remained `1536e86a64cbf09a1c0308ab72985a67eba47eb748f3b4f7ba23accc25e7543f`

Actual M5 Max validation is the next gate.

## Preserved artifacts

- `graph-native-model-mlx-v0.5.7.zip` — `9ba4facb31ed1e7aa45a87fce66362b63fc4d69298c2890593fdd479c6115239`
- `graph_native_model-0.5.7-py3-none-any.whl` — `a4f9584c22df67a5bacb0e1f72cdfdea0c3e739529269657c46e93bbef4e80f4`
- `graph-native-model-mlx-v0.5.7-release.json` — `0cfca138b6f9f9f8e037614f3f35cfb72f107fa27c582245d5f09f4f35443a26`
