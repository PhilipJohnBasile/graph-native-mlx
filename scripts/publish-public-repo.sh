#!/usr/bin/env bash
set -euo pipefail

OWNER="${GITHUB_OWNER:-PhilipJohnBasile}"
REPO="${GITHUB_REPO:-graph-native-mlx}"
VISIBILITY="public"

if ! command -v gh >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install gh
  else
    echo "GitHub CLI is required: https://cli.github.com/"
    exit 1
  fi
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "Authenticate GitHub CLI, then rerun this script:"
  echo "  gh auth login"
  exit 2
fi

if gh repo view "$OWNER/$REPO" >/dev/null 2>&1; then
  echo "Repository already exists: $OWNER/$REPO"
else
  gh repo create "$OWNER/$REPO" --public --description     "Graph-controlled MLX coding-agent runtime with durable execution, hidden-state policy features, and reproducible release history."
fi

REMOTE="https://github.com/$OWNER/$REPO.git"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE"
else
  git remote add origin "$REMOTE"
fi

git push -u origin main
git push origin --tags

for dir in releases/v*; do
  version="${dir##*/}"
  if gh release view "$version" --repo "$OWNER/$REPO" >/dev/null 2>&1; then
    echo "Release exists: $version"
    continue
  fi
  notes="$dir/README.md"
  assets=()
  while IFS= read -r -d '' file; do assets+=("$file"); done < <(
    find "$dir" -maxdepth 1 -type f       ! -name README.md ! -name 'SHA256.normalized.txt' -print0
  )
  gh release create "$version" "${assets[@]}"     --repo "$OWNER/$REPO" --title "Graph-Native MLX $version" --notes-file "$notes"
done

echo "Published: https://github.com/$OWNER/$REPO"
