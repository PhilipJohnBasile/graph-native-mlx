# v0.5.4 — Observable, fail-fast collection

Fixed lexical venv paths in manifests, added bounded JSON recovery, terminalized collector exceptions, and printed per-task progress.

## Problem observed

False pytest failures, malformed JSON, and swallowed exceptions left stale running rows and apparently frozen terminals.

## Result

Tasks 1-8 completed cleanly and collector failures became explicit and resumable.

## Preserved artifacts

- `graph-native-model-mlx-v0.5.4.zip` — `db0b747d686c60e7f1010f02cadcf8710d4874830598de73397414174fd42d6c`
- `graph_native_model-0.5.4-py3-none-any.whl` — `36f01ba5585d6da222cec847efd10193a2a93360e14191e55b071dec496d5fb1`
- `graph-native-model-mlx-v0.5.4-SHA256.txt` — `7cbff068f8c5736af63be8f1d1a93fdacf033d3cd1b218b92fa07b9b99ee063d`
- `graph-native-model-mlx-v0.5.4-release.json` — `793ba92771592d74130bf97c3e74de7510857af70392dc4e59cf589773bf652d`

The source tree at Git tag `v0.5.4` corresponds to this release archive.
