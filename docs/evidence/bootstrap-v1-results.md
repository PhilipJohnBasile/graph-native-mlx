# Bootstrap v1 result

The clean v0.5.6 corpus completed on an Apple M5 Max using the pinned Qwen3.8 MLX checkpoint.

| Measure | Result |
|---|---:|
| Repository tasks | 16 |
| Verified completions | 14 |
| Intentional bounded aborts | 2 |
| Collector errors | 0 |
| Policy records | 165 |
| Route records | 16 |
| Transition records | 149 |
| Positive reward records | 133 |
| Zero reward records | 32 |
| Raw Qwen hidden size | 5,120 |
| Projected policy size | 256 |
| Model fingerprints | 1 |
| Outcome mismatches | 0 |

No learned policy was activated. This dataset is the bootstrap input for the next training and held-out-evaluation gate.

The two failed tasks were intentionally impossible contracts and reached the explicit bounded `abort` node after exhausting the permitted repair paths.
