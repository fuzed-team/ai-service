# Python AI Microservice - Docker Image
# Optimized for Railway, Fly.io, Google Cloud Run

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and build tools
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    gcc \
    g++ \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install onnxruntime first with specific flags to handle executable stack issue
RUN pip install --no-cache-dir onnxruntime==1.17.1

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Fix executable stack issue for onnxruntime on Docker Desktop/WSL2
RUN find /usr/local/lib/python3.11/site-packages/onnxruntime/capi -name "*.so" -exec patchelf --clear-execstack {} \; 2>/dev/null || true

# Pre-download InsightFace model (optional - reduces cold start time)
# RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(providers=['CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640, 640))"

# Copy application code
COPY app.py .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run with gunicorn for production (better than Flask dev server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]

# For development, use:
# CMD ["python", "app.py"]
