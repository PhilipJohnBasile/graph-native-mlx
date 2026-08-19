# v0.5.1 — macOS virtual-environment correctness

Preserved the lexical venv Python invocation, added recent-run discovery, concise reports, and a state-preserving Mac upgrader.

## Problem observed

Resolving venv/bin/python to the Homebrew framework binary bypassed venv site-packages and made pytest disappear.

## Result

Verifier commands used the exact configured venv interpreter; reporting no longer required manual SQLite queries.

## Preserved artifacts

- `graph-native-model-mlx-v0.5.1.zip` — `b9ac79ec8c0870e3f6437c0f512497de1deaf54fd3cd9a93182c15182b9c5427`
- `graph_native_model-0.5.1-py3-none-any.whl` — `da9f676aaac02fe8d3c6e17d63539d28c08ae21056c23ce463b7e4ce335cc380`
- `graph-native-model-mlx-v0.5.1-SHA256.txt` — `9ed85a404840ab4ed28b87a58be8e281fb0c161099f3a0926ad9dec5d3c3b8ee`
- `graph-native-model-mlx-v0.5.1-release.json` — `dfe7947d9a2c3cd3f90846c8e8a580a7381cef13cab3048b9d23d2f03082acd7`

The source tree at Git tag `v0.5.1` corresponds to this release archive.
