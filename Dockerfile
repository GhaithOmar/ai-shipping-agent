# ---------- Base image ----------
FROM python:3.11-slim AS base

# System deps: tini for proper signal handling, curl for healthcheck,
# and a few build tools some Python wheels may need.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini curl ca-certificates build-essential git \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/hf_cache

# App directory
WORKDIR /app

# Copy only files needed to install deps first (better cache)
# If you have requirements.txt, we’ll use it; otherwise we’ll install a minimal runtime set later.
COPY requirements.txt /app/requirements.txt

# Try to install from requirements.txt if present
RUN if [ -f /app/requirements.txt ]; then \
        python -m pip install --upgrade pip && \
        pip install --no-cache-dir -r /app/requirements.txt ; \
    else \
        echo "No requirements.txt; installing minimal runtime set…" && \
        python -m pip install --upgrade pip && \
        pip install --no-cache-dir \
            fastapi "uvicorn[standard]" pydantic \
            transformers accelerate peft \
            sentence-transformers qdrant-client \
            langchain langgraph ; \
    fi

# Copy application code
# (Your .dockerignore excludes heavy stuff like notebooks, qdrant_db, data, etc.)
COPY backend/      /app/backend/
COPY rag/          /app/rag/
COPY scripts/      /app/scripts/
COPY docker/       /app/docker/
COPY warm_agent.py /app/warm_agent.py
COPY README.md     /app/README.md

# Generate .env.example inside the image (fallback if host file is tricky)
RUN printf '%s\n' \
 'BASE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0' \
 'FALLBACK_BASE=TinyLlama/TinyLlama-1.1B-Chat-v1.0' \
 'ADAPTER_ID=' \
 'HUGGINGFACE_TOKEN=' \
 'HF_TOKEN=' \
 'AGENT_OFFLINE=0' \
 'AGENT_ENABLE=true' \
 'AGENT_TOP_K=4' \
 'QDRANT_HOST=127.0.0.1' \
 'QDRANT_PORT=6333' \
 'QDRANT_COLLECTION=shipping_kb' \
 'PORT=8000' \
 'TOKENIZERS_PARALLELISM=false' \
 'HF_HUB_OFFLINE=0' \
 'TRANSFORMERS_OFFLINE=0' \
 'HF_HOME=/app/hf_cache' \
 > /app/.env.example


# Ensure entrypoint is executable and LF-normalized (handles CRLF from Windows)
RUN chmod +x /app/docker/entrypoint.sh \
 && sed -i 's/\r$//' /app/docker/entrypoint.sh

# Expose API port
EXPOSE 8000

# Healthcheck hits /health (your API defines it)
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Use tini as PID 1 for clean shutdowns
ENTRYPOINT ["/usr/bin/tini", "--"]

# Launch via entrypoint (starts uvicorn)
CMD ["bash", "/app/docker/entrypoint.sh"]
