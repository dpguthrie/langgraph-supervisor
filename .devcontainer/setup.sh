#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

run_optional() {
  local label="$1"
  shift

  echo "[devcontainer] ${label}"
  if "$@"; then
    echo "[devcontainer] ${label} complete"
  else
    echo "[devcontainer] ${label} failed; continuing"
  fi
}

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
  run_optional "installing Claude Code" npm install -g --no-audit --no-fund @anthropic-ai/claude-code
fi

if ! command -v bt >/dev/null 2>&1; then
  install_bt() {
    curl -fsSL https://bt.dev/cli/install.sh | bash
  }

  run_optional "installing Braintrust CLI" install_bt
fi

if command -v bt >/dev/null 2>&1 && bt setup skills --help >/dev/null 2>&1; then
  if command -v claude >/dev/null 2>&1; then
    run_optional "configuring Braintrust skills for Claude if credentials are available" bt setup skills --agent claude
  fi
fi
