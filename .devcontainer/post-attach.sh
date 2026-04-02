#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f .env ]; then
  echo "[devcontainer] .env detected"
else
  echo "[devcontainer] .env missing; copy values into .env or set Codespaces secrets"
fi

if command -v python >/dev/null 2>&1; then
  python --version
fi

if command -v uv >/dev/null 2>&1; then
  uv --version
fi

if command -v claude >/dev/null 2>&1; then
  echo "[devcontainer] Claude Code available"
else
  echo "[devcontainer] Claude Code not found"
fi

if command -v bt >/dev/null 2>&1; then
  echo "[devcontainer] Braintrust CLI available"
else
  echo "[devcontainer] Braintrust CLI not found"
fi
