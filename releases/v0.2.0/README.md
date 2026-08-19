# v0.2.0 — MLX-native controller foundation

Added direct MLX-LM inference, generated graph tables, hard-masked MLX decisions, feature extraction, and trainable policy sidecars.

## Problem observed

The graph existed outside the model but could not yet use MLX-native representations for control.

## Result

The language backbone and graph controller could share a resident MLX execution path while the graph remained authoritative.

## Preserved artifacts

- `graph-native-model-mlx-v0.2.0.zip` — `383d7a45f68341212473f88fa601eae3cb601d9ae309d523c00689c757c87d7f`
- `graph_native_model-0.2.0-py3-none-any.whl` — `675a652c8d364d059dffd83ca415b438314b7229bab462c6c99645786561a24c`
- `graph-native-model-mlx-v0.2.0-SHA256.txt` — `1d7961c8bc51a98dfa827e07eb0bd09cec9691e66ee51cbb248bab2c946c3817`

The source tree at Git tag `v0.2.0` corresponds to this release archive.
