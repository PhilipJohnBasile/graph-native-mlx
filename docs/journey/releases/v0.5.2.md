# v0.5.2 — Deterministic verifier bytecode

Added a fresh external PYTHONPYCACHEPREFIX and disabled user-site imports for every verifier command.

## Problem observed

Same-second, same-size repairs could reuse stale timestamp-based .pyc bytecode.

## Result

Every verifier run evaluated current source and the repaired-candidate regression became deterministic.

## Preserved artifacts

- `graph-native-model-mlx-v0.5.2.zip` — `fed833e0f6ad521168ec809a12de0a7754da6fb936117a7de49b452e8ecc71bd`
- `graph_native_model-0.5.2-py3-none-any.whl` — `3a031a8077b32020331c9ee5c967d94c686bcaafbf8f8748db6a578d35e79d39`
- `graph-native-model-mlx-v0.5.2-SHA256.txt` — `15d4e73b30d99a8d8ac72e345afb7a385b997190dca2f5527af2b24460800bb4`
- `graph-native-model-mlx-v0.5.2-release.json` — `0ac33dff70bacb0141b8d67a14b783a0daab3f144c5e9252d142d6b898547803`

The source tree at Git tag `v0.5.2` corresponds to this release archive.
