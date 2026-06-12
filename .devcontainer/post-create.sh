#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ "${CODESPACES:-}" = "true" ]; then
  echo "[devcontainer] Codespaces detected; expecting API keys from Codespaces secrets"
  for name in BRAINTRUST_API_KEY OPENAI_API_KEY TAVILY_API_KEY; do
    if [ -n "${!name:-}" ]; then
      echo "[devcontainer] ${name} is set"
    else
      echo "[devcontainer] ${name} is missing; add it as a Codespaces secret if needed"
    fi
  done
elif [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo "[devcontainer] created .env from .env.example"
fi
