# v0.4.0 — Qwen hidden-state policy features

Added selected Qwen hidden-state capture, deterministic 5,120-to-256 projection, hidden-aware policy fusion, artifact integrity, and repository trace collection.

## Problem observed

Static rules had no compact representation of what the model internally understood at each graph checkpoint.

## Result

Real Qwen representations could inform graph decisions without persisting raw prompts or raw hidden tensors.

## Preserved artifacts

- `graph-native-model-mlx-v0.4.0.zip` — `b1dc77926e935b1dd0cc046f77874488de3be8cf2f77e739663d3513fb14c367`
- `graph_native_model-0.4.0-py3-none-any.whl` — `3da628353144aa79e09f806ea521f314df99cef20ded040115ace38d4bcf98a6`
- `graph-native-model-mlx-v0.4.0-SHA256.txt` — `41e31ee0d96b31f71f4d7e3f8bd884e33f904617a8f920d84d32aeb5a9648ab4`
- `graph-native-model-mlx-v0.4.0-release.json` — `80c02ea62e42fa3b4b7e300c835b5ed2ebce8d6fec9a84bf808b5f4401e4face`

The source tree at Git tag `v0.4.0` corresponds to this release archive.
