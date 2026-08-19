# v0.5.6 — Bounded truncation recovery and clean corpus

Disabled Qwen thinking for structured artifacts by default and added one truncation-only continuation for GRAPH_PATCH_V1.

## Problem observed

A deep migration task spent the full 4,096-token allowance before emitting the actual patch envelope.

## Result

The complete 16-task corpus passed: 14 verified completions, 2 intentional bounded aborts, 165 policy records, one model fingerprint, and homogeneous 256-dimensional features.

## Preserved artifacts

- `graph-native-model-mlx-v0.5.6.zip` — `65035c6981364843c1cbe8617f6b326a1569f24a42ec37394754f26d40ffa910`
- `graph_native_model-0.5.6-py3-none-any.whl` — `641b53837912b78a8e90086db55b5eabf2b6fa1175a8f292a25f5e6ce6b136c1`
- `graph-native-model-mlx-v0.5.6-SHA256.txt` — `006eeabe9ce92f7725451328e6bb49a207d2a887216a76e19e95599e180c1fb6`
- `graph-native-model-mlx-v0.5.6-release.json` — `8bbf63608bc70c924247ce6919f190b0556c1df1adb7c40a3460901000190bc2`

The source tree at Git tag `v0.5.6` corresponds to this release archive.
