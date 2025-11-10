# FastAPI AI Microservice - Multi-stage Docker Build
# Optimized for production deployment with UV package manager

# ===== Builder Stage =====
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies for compiling Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files and app structure
COPY pyproject.toml .
COPY app/__init__.py app/__init__.py

# Create virtual environment and install dependencies using UV
# Increase timeout for large packages like TensorFlow (615MB)
ENV UV_HTTP_TIMEOUT=300
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python .

# ===== Runtime Stage =====
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and ONNX
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY app /app/app

# Fix executable stack issue for onnxruntime
RUN find /app/.venv/lib -name "*.so" -exec patchelf --clear-execstack {} \; 2>/dev/null || true

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health')" || exit 1

# Run with Uvicorn for FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
