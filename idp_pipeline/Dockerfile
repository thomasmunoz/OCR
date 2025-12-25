# IDP Pipeline - Production Dockerfile
# Supports both CPU and GPU (NVIDIA)
# Multi-stage build for smaller image

# ============================================
# Stage 1: Base with dependencies
# ============================================
FROM python:3.11-slim AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ============================================
# Stage 2: Dependencies
# ============================================
FROM base AS deps

# Copy requirements first for layer caching
COPY requirements.txt .
COPY docker/requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-docker.txt

# ============================================
# Stage 3: Production image
# ============================================
FROM deps AS production

# Copy application code
COPY . /app/

# Create directories and non-root user
RUN mkdir -p /app/uploads /app/output /app/models /data/uploads /data/output /data/models \
    && useradd -m -u 1000 idp \
    && chown -R idp:idp /app /data
USER idp

# Environment
ENV IDP_UPLOAD_DIR=/data/uploads \
    IDP_OUTPUT_DIR=/data/output \
    IDP_MODELS_DIR=/data/models \
    IDP_QUEUE_TYPE=redis \
    IDP_REDIS_URL=redis://redis:6379/0 \
    IDP_PORT=8080 \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${IDP_PORT}/health || exit 1

# Expose port
EXPOSE 8080

# Default command (API server)
CMD ["python", "run.py", "serve", "--host", "0.0.0.0", "--port", "8080"]

# ============================================
# Stage 4: Worker image (for scaling)
# ============================================
FROM production AS worker

# Disable the API healthcheck (worker doesn't serve HTTP)
HEALTHCHECK NONE

# Worker command
CMD ["python", "run.py", "worker"]

# ============================================
# Stage 5: GPU image (optional)
# ============================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS gpu-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    libmagic1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/dist-packages
COPY . /app/

RUN mkdir -p /data/uploads /data/output /data/models \
    && useradd -m -u 1000 idp && chown -R idp:idp /app /data

USER idp

ENV IDP_UPLOAD_DIR=/data/uploads \
    IDP_OUTPUT_DIR=/data/output \
    IDP_MODELS_DIR=/data/models \
    IDP_QUEUE_TYPE=redis \
    IDP_REDIS_URL=redis://redis:6379/0 \
    NVIDIA_VISIBLE_DEVICES=all \
    PYTHONPATH=/app

EXPOSE 8080
CMD ["python", "run.py", "serve", "--host", "0.0.0.0", "--port", "8080"]
