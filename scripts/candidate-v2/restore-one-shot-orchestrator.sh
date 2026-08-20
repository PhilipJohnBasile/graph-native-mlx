#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$ROOT/one-shot-orchestrator-budget-safe.sh.gz.b64"
OUTPUT="${1:-$ROOT/graph-native-mlx-candidate-v2-one-shot-budget-safe.sh}"
EXPECTED_SHA="f506b193a10be30fdb448652c0472ba44500649ebb5546c395b0c7f09213092e"

if [[ ! -f "$SOURCE" ]]; then
  echo "ERROR: encoded orchestrator is missing:"
  echo "  $SOURCE"
  exit 1
fi

python3 - "$SOURCE" "$OUTPUT" <<'PY'
from __future__ import annotations

import base64
import gzip
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
raw = base64.b64decode(source.read_text(encoding="utf-8"))
output.parent.mkdir(parents=True, exist_ok=True)
output.write_bytes(gzip.decompress(raw))
PY

ACTUAL_SHA="$(shasum -a 256 "$OUTPUT" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "ERROR: restored orchestrator checksum mismatch."
  echo "Expected: $EXPECTED_SHA"
  echo "Actual:   $ACTUAL_SHA"
  rm -f "$OUTPUT"
  exit 1
fi

chmod +x "$OUTPUT"
echo "Restored:"
echo "  $OUTPUT"
echo "SHA-256:"
echo "  $ACTUAL_SHA"
