# ----------------------------
# Stage 1: builder (compile wheels, cache deps)
# ----------------------------
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for building wheels (add gcc if some libs require it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels

# Copy only requirements first for better Docker layer caching
COPY requirements.txt /wheels/requirements.txt

# Build wheels for all deps to speed up installs in the final image
RUN pip wheel --no-cache-dir -r /wheels/requirements.txt -w /wheels

# ----------------------------
# Stage 2: runtime (slim, non-root)
# ----------------------------
FROM python:3.11-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app

# Minimal OS packages for runtime (add if your libs need more)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl tini \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user and app dirs
RUN useradd -m appuser
WORKDIR ${APP_HOME}

# Copy prebuilt wheels from builder and install
COPY --from=builder /wheels /tmp/wheels
RUN pip install --no-cache-dir --find-links=/tmp/wheels -r /tmp/wheels/requirements.txt \
 && rm -rf /tmp/wheels

# Copy application code (keep order for better caching)
# We deliberately do NOT copy qdrant_db/ or rag/data/ (mounted at runtime)
COPY backend/ backend/
COPY rag/ rag/
#COPY tests/ tests/
COPY docker/ docker/
RUN chmod +x docker/entrypoint.sh
COPY README.md .
# helpful for users; safe to include
COPY .env.example .env.example



# Create runtime directories and adjust permissions
RUN mkdir -p qdrant_db && chown -R appuser:appuser ${APP_HOME}
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Healthcheck: hit the root or a lightweight ping endpoint if you add one
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1


# Use tini as init for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Entrypoint script will:
#  1) ingest KB if qdrant_db is empty
#  2) start uvicorn backend.main:app
CMD ["bash", "docker/entrypoint.sh"]
