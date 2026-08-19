# v0.5.3 — Clean failed-run learning semantics

Prevented failed runs from teaching route/edge/stop imitation and added a controlled 16-repository bootstrap corpus.

## Problem observed

Failed executions were still being imitated as if their actions were desirable.

## Result

Failures train value and cost only; the first balanced fast/deep/repair/no-change/abort corpus became collectable.

## Preserved artifacts

- `graph-native-model-mlx-v0.5.3.zip` — `19a936373c830e641823aa11546ec6ccdc4cfda06170edfd5942bf93fd38b606`
- `graph_native_model-0.5.3-py3-none-any.whl` — `41441e5d40715753f2d1c6df8cd4277154cff95d895cb642acb89e98fcb562c9`
- `graph-native-model-mlx-v0.5.3-SHA256.txt` — `b2f6137c3a0461be8dd3e80acbb6ced4a600783bd6c9bd6094deb7ef653910f7`
- `graph-native-model-mlx-v0.5.3-release.json` — `056039699ea81c04309b5e10122418a2a1acd5856d0f31dc0b68efb2201d8f62`

The source tree at Git tag `v0.5.3` corresponds to this release archive.
