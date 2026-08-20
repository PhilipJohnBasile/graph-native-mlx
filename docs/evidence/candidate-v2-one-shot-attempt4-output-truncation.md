# Candidate-v2 one-shot — Stage 03 attempt 4

## Result

The budget-safe implementation pass reached the model successfully but failed closed because the generated patch envelope was truncated at the provider's per-generation output cap.

```text
ProviderError: model did not return a valid patch envelope after one bounded recovery and one truncation continuation
chars=69648
completion_tokens=8192
max_tokens=8192
likely_truncated=true
GRAPH_PATCH_V1=yes
GRAPH_PATCH_META_BEGIN=yes
GRAPH_PATCH_META_END=yes
GRAPH_PATCH_DIFF_BEGIN=yes
GRAPH_PATCH_DIFF_END=no
continuation_used=true
```

This is distinct from the earlier failures:

1. workspace validator rejected `max_context_files=300`;
2. repository prompt exceeded the model context window;
3. aggregate graph execution exceeded the default 64k run budget;
4. **current attempt:** aggregate budget was fixed, but the actual patch generation reached the 8,192-token per-generation cap before emitting `GRAPH_PATCH_DIFF_END`.

## Interpretation

The model produced the patch envelope header and metadata, entered the diff body, and consumed the complete 8,192-token generation allowance. A bounded continuation was attempted but the complete envelope still did not terminate. No patch was applied or promoted.

This is an orchestration/output-size gate, not candidate-v2 safety or benefit evidence. Candidate-v2 code and weights still do not exist.

## Next orchestrator

Keep the successful repository-context and aggregate-runtime bounds:

```text
context files:       60
context file bytes:  80,000
context bytes:       400,000
aggregate tokens:    512,000
steps:               32
LLM calls:           16
tool calls:          40
active time:         3,600 seconds
```

Increase only the source-implementation provider output cap from 8,192 to 16,384 tokens. This remains bounded and is comfortably inside the model context window with the reduced repository-context cap.

Output-safe orchestrator SHA-256:

`39bd4429b7e1485856ed0ea3a312da09ff88078ffb0abe04546aa7032e1156e6`

If the core implementation still truncates at 16,384, do **not** continue raising output limits indefinitely. Split the core implementation contract into smaller verified passes instead.

Candidate v1 remains frozen and globally disabled. Candidate v2 remains unimplemented, untrained, and disabled.
