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

with_timeout() {
  local duration="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout "$duration" "$@"
  else
    "$@"
  fi
}

if ! command -v uv >/dev/null 2>&1; then
  run_optional "installing uv" with_timeout 5m bash -o pipefail -c 'curl -LsSf --connect-timeout 15 --max-time 300 https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$HOME/.local/bin" sh'
fi

if command -v uv >/dev/null 2>&1; then
  run_optional "syncing Python environment with uv" with_timeout 30m uv sync --frozen --extra dev
else
  echo "[devcontainer] uv not available; skipping Python environment sync"
fi

if command -v npm >/dev/null 2>&1 && ! command -v claude >/dev/null 2>&1; then
  run_optional "installing Claude Code" with_timeout 10m npm install -g --no-audit --no-fund @anthropic-ai/claude-code
fi

if ! command -v bt >/dev/null 2>&1; then
  run_optional "installing Braintrust CLI" with_timeout 10m bash -o pipefail -c 'curl -fsSL --connect-timeout 15 --max-time 600 https://bt.dev/cli/install.sh | bash'
fi

if command -v bt >/dev/null 2>&1 && bt setup skills --help >/dev/null 2>&1; then
  if command -v claude >/dev/null 2>&1; then
    run_optional "configuring Braintrust skills for Claude" bt setup skills --agent claude --global --no-input
  fi
fi
