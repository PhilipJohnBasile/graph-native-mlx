#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${GRAPH_NATIVE_MLX_HOME:-$HOME/graph-native-mlx}"
BACKUP_ROOT="${GRAPH_NATIVE_MLX_BACKUP_ROOT:-$HOME/graph-native-mlx-backups}"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="$BACKUP_ROOT/v0.4-to-v0.5.1-$STAMP"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: this upgrade helper requires Apple Silicon macOS." >&2
  exit 1
fi

if [[ ! -d "$TARGET" || ! -f "$TARGET/pyproject.toml" ]]; then
  echo "ERROR: existing installation not found at $TARGET" >&2
  exit 1
fi
if [[ ! -x "$TARGET/.venv/bin/python" ]]; then
  echo "ERROR: existing virtual environment not found at $TARGET/.venv" >&2
  exit 1
fi
if [[ ! -f "$TARGET/.graph-env" ]]; then
  echo "ERROR: existing $TARGET/.graph-env was not found." >&2
  exit 1
fi

mkdir -p "$BACKUP_ROOT" "$BACKUP"
rsync -a \
  --exclude='.venv/' \
  --exclude='.graph-model/' \
  --exclude='build/' \
  --exclude='dist/' \
  --exclude='*.egg-info/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  "$TARGET/" "$BACKUP/"

rsync -a --delete \
  --exclude='.venv/' \
  --exclude='.graph-model/' \
  --exclude='.graph-env' \
  --exclude='.git/' \
  --exclude='build/' \
  --exclude='dist/' \
  --exclude='*.egg-info/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  "$SOURCE_DIR/" "$TARGET/"

cd "$TARGET"
source .venv/bin/activate
source .graph-env

python -m pip install -e '.[dev,mlx]'
python -m pytest -q
graph-model validate

echo
echo "Recent runs retained from the previous installation:"
graph-model runs --limit 10 || true

echo
echo "Latest completed-run report:"
graph-model report --latest --latest-status completed || true

echo
echo "Running exact Mac/model qualification:"
graph-model qualify-mac --output-dir .graph-model/qualification

echo
echo "Upgrade complete."
echo "Project: $TARGET"
echo "Backup:  $BACKUP"
echo "Report:  $TARGET/.graph-model/qualification/mlx-m5-qualification.md"
