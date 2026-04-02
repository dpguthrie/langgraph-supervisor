#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "[devcontainer] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$HOME/.local/bin" sh
fi

echo "[devcontainer] syncing Python environment with uv"
uv sync --extra dev

if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo "[devcontainer] created .env from .env.example"
fi

if command -v npm >/dev/null 2>&1 && ! command -v claude >/dev/null 2>&1; then
  echo "[devcontainer] installing Claude Code"
  npm install -g @anthropic-ai/claude-code
fi

if ! command -v bt >/dev/null 2>&1; then
  echo "[devcontainer] installing Braintrust CLI"
  curl -fsSL https://bt.dev/cli/install.sh | sh
fi

if command -v bt >/dev/null 2>&1 && bt setup skills --help >/dev/null 2>&1; then
  if command -v claude >/dev/null 2>&1; then
    echo "[devcontainer] configuring Braintrust skills for Claude if credentials are available"
    bt setup skills --agent claude || true
  fi
fi
