#!/bin/bash

set -e

HOOKS_DIR=".git/hooks"
REPO_HOOKS_DIR="scripts/hooks"

echo "Setting up Git hooks..."

for hook in pre-commit pre-push; do
  src="$REPO_HOOKS_DIR/$hook"
  dest="$HOOKS_DIR/$hook"
  if [ -f "$src" ]; then
    if [ ! -f "$dest" ] || ! cmp -s "$src" "$dest"; then
      cp "$src" "$dest"
      chmod +x "$dest"
      echo "✔ Installed or updated $hook hook"
    else
      echo "✔ $hook already up-to-date"
    fi
  fi
done

echo "All hooks installed successfully."