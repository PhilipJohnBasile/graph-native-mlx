# v0.5.5 — Patch-native structured output

Introduced GRAPH_PATCH_V1: compact JSON metadata plus a raw unified diff outside JSON, with one deterministic recovery.

## Problem observed

Deep multi-file diffs were fragile when embedded as escaped multiline JSON strings.

## Result

The previously failing deep email-refactor task completed through tests, review, and verified finish.

## Preserved artifacts

- `graph-native-model-mlx-v0.5.5.zip` — `c76219d85b92cb799b43ad345002cf00e0e08163fbcdeb44d55e63f9635f86dd`
- `graph_native_model-0.5.5-py3-none-any.whl` — `e7bb8f62680aa10151da519eeac6de75652c8deb69f758daee15a50087586a03`
- `graph-native-model-mlx-v0.5.5-SHA256.txt` — `8d1f9e5cb13a875a730ba82ceba9c0e1de1f36f0fea50e3229bf718c6fa3f8cc`
- `graph-native-model-mlx-v0.5.5-release.json` — `9653dfb6bd4b14bb1838c9e686072b4dcaa0883abe6c521143d55fb475b42b50`

The source tree at Git tag `v0.5.5` corresponds to this release archive.
