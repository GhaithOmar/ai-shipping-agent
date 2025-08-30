#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting…"
APP_DIR="/app"
DB_DIR="${APP_DIR}/qdrant_db"
DATA_DIR="${APP_DIR}/rag/data"

# 1) ingest only if db is missing or empty
if [ ! -d "$DB_DIR" ] || [ -z "$(ls -A "$DB_DIR" 2>/dev/null || true)" ]; then
  echo "[entrypoint] qdrant_db is empty → ingesting KB…"
  if [ -d "$DATA_DIR" ]; then
    python rag/ingest.py
  else
    echo "[entrypoint] WARNING: rag/data not found. Skipping ingest."
  fi
else
  echo "[entrypoint] qdrant_db already populated. Skipping ingest."
fi

# 2) run the API
echo "[entrypoint] launching uvicorn…"
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000