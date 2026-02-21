#!/usr/bin/env bash
set -e

# Azure App Service sets $PORT; default to 8000 for local runs
exec uv run gunicorn azure_app:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind "0.0.0.0:${PORT:-8000}" \
    --timeout 3600 \
    --workers "${GUNICORN_WORKERS:-2}"
