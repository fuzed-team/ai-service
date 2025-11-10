# Flask to FastAPI Migration Plan

**Project**: AI Face Matching Service
**Date**: 2025-11-09
**Status**: In Progress

## Executive Summary

Complete migration of 859-line monolithic Flask application to FastAPI following best practices from [zhanymkanov/fastapi-best-practices](https://github.com/zhanymkanov/fastapi-best-practices) and migrating to UV package manager.

### Key Goals
1. **Framework Migration**: Flask → FastAPI with async support
2. **Package Management**: pip/requirements.txt → UV (10-100x faster)
3. **Architecture**: Monolithic → Domain-based modular structure
4. **Performance**: Process pool offloading for CPU-intensive ML workloads
5. **Validation**: Manual checks → Pydantic automatic validation
6. **Documentation**: Auto-generated OpenAPI/Swagger docs

---

## Current State Analysis

### Application Overview
- **Framework**: Flask 3.1.2
- **Structure**: Single 859-line `app.py` file
- **Endpoints**: 6 routes (health, verify-face, extract-embedding, compare-faces, batch-extract, analyze-face-advanced)
- **ML Libraries**: InsightFace 0.7.3, DeepFace 0.0.95, TensorFlow 2.15.0
- **Auth**: Custom API key validation (Bearer token)
- **Deployment**: Docker + Gunicorn (1 worker, 2 threads)

### Critical Issues Identified

#### Architecture Issues
- 859-line monolithic file with no separation of concerns
- No dependency injection
- Global state management with lazy model loading
- Mixed business logic and routing
- No API versioning

#### Code Quality Issues
- Massive code duplication (image loading repeated 5+ times)
- No type hints
- Mixed error handling patterns
- No input validation framework
- Inconsistent logging

#### Performance Issues
- Blocking ML operations in request handlers
- No async support
- Limited concurrency (Gunicorn thread pool)
- Models loaded on first request (not startup)

#### Security Issues
- CORS allows all origins
- No rate limiting
- No request size limits
- No input sanitization

#### DevOps Issues
- No tests (0% coverage)
- No CI/CD
- No dependency locking (no reproducible builds)
- Large Docker image with no optimization

---

## Target Architecture

### Directory Structure

```
ai-service/
├── .agent/
│   └── tasks/
│       └── fastapi-migration-plan.md
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app initialization
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # Pydantic BaseSettings
│   │   ├── security.py              # API key validation
│   │   └── exceptions.py            # Custom exception handlers
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py          # Auth & model loading dependencies
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py            # Main v1 router
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── health.py        # GET /health
│   │           ├── face.py          # POST /verify-face, /extract-embedding, /analyze-face-advanced
│   │           └── compare.py       # POST /compare-faces, /batch-extract
│   ├── models/
│   │   ├── __init__.py
│   │   ├── face.py                  # Pydantic request/response schemas
│   │   └── common.py                # Common schemas (health, errors)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── face_detection.py        # InsightFace wrapper
│   │   ├── face_analysis.py         # DeepFace wrapper
│   │   ├── embedding.py             # Embedding extraction logic
│   │   └── image_processing.py      # Image utilities (base64, multipart, URL)
│   └── workers/
│       ├── __init__.py
│       └── ml_worker.py             # ProcessPoolExecutor for ML inference
├── pyproject.toml                   # UV project configuration
├── uv.lock                          # Dependency lockfile
├── Dockerfile                       # Multi-stage build with UV
├── .dockerignore
├── .env.example
├── .gitignore
├── README.md
└── app.py                           # LEGACY - TO BE REMOVED
```

---

## Implementation Phases

### Phase 1: UV Setup & Dependency Migration (30 min)

#### 1.1 Install UV Package Manager
```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

#### 1.2 Initialize UV Project
```bash
uv init --no-readme
```

#### 1.3 Create pyproject.toml
Configure project metadata and dependencies:
```toml
[project]
name = "ai-face-matching-service"
version = "1.0.0"
description = "FastAPI service for AI-powered face detection, verification, and analysis"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "python-multipart>=0.0.9",
    "insightface==0.7.3",
    "onnxruntime==1.23.2",
    "opencv-python-headless==4.10.0.84",
    "numpy==1.26.4",
    "protobuf==3.20.3",
    "deepface==0.0.95",
    "tensorflow==2.15.0",
    "tf-keras==2.15.0",
    "scikit-learn==1.7.2",
    "scipy==1.16.3",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.8.0",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.28.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

#### 1.4 Install Dependencies
```bash
uv sync
```

#### 1.5 Update .gitignore
Add UV-specific entries:
```
# UV
.venv/
uv.lock
__pypackages__/
```

---

### Phase 2: Core Infrastructure (1 hour)

#### 2.1 Configuration Management (core/config.py)

**Principles Applied**:
- Pydantic BaseSettings for environment variables
- Type validation and defaults
- Separate concerns (app, CORS, ML, workers)

**Implementation**:
```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Configuration
    api_key: str = Field(..., description="API authentication key")
    debug: bool = Field(default=False, description="Debug mode")

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # CORS Configuration
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )

    # ML Configuration
    face_det_thresh: float = Field(default=0.5, description="Face detection threshold")

    # Worker Configuration
    max_workers: int = Field(default=2, description="ML process pool workers")
    worker_timeout: int = Field(default=120, description="Worker timeout in seconds")


settings = Settings()
```

#### 2.2 Security Layer (core/security.py)

**Principles Applied**:
- FastAPI dependency injection
- Reusable security dependencies
- Proper HTTP exception handling

**Implementation**:
```python
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.core.config import settings

security = HTTPBearer()


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    """Verify API key from Bearer token."""
    if credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
```

#### 2.3 Exception Handling (core/exceptions.py)

**Principles Applied**:
- Custom exceptions for domain errors
- Global exception handlers
- Structured error responses

**Implementation**:
```python
from fastapi import Request, status
from fastapi.responses import JSONResponse


class FaceDetectionError(Exception):
    """Raised when face detection fails."""
    pass


class InvalidImageError(Exception):
    """Raised when image is invalid or cannot be processed."""
    pass


class EmbeddingExtractionError(Exception):
    """Raised when embedding extraction fails."""
    pass


async def face_detection_error_handler(
    request: Request, exc: FaceDetectionError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Face detection failed", "detail": str(exc)},
    )


async def invalid_image_error_handler(
    request: Request, exc: InvalidImageError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid image", "detail": str(exc)},
    )


# Register handlers in main.py
```

---

### Phase 3: ML Process Pool (1.5 hours)

#### 3.1 Worker Implementation (workers/ml_worker.py)

**Principles Applied**:
- CPU-intensive work in separate processes
- Async wrappers for non-blocking execution
- Proper resource cleanup

**Key Considerations**:
- Initialize models in worker processes (not main)
- Use ProcessPoolExecutor (not ThreadPoolExecutor)
- Handle process lifecycle via FastAPI lifespan

**Implementation**:
```python
import asyncio
import base64
import io
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# Worker-local model instance (initialized per process)
_face_app = None


def _init_worker():
    """Initialize ML models in worker process."""
    global _face_app
    _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    _face_app.prepare(ctx_id=0, det_size=(640, 640))


def _detect_face_sync(image_data: bytes) -> dict[str, Any]:
    """Synchronous face detection (runs in worker process)."""
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image")

    faces = _face_app.get(img)

    if len(faces) == 0:
        raise ValueError("No face detected")

    face = faces[0]
    return {
        "embedding": face.embedding.tolist(),
        "bbox": face.bbox.tolist(),
        "det_score": float(face.det_score),
    }


class MLWorkerPool:
    """Process pool for ML inference."""

    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.executor: ProcessPoolExecutor | None = None

    async def start(self):
        """Start worker pool."""
        self.executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_init_worker,
        )

    async def stop(self):
        """Stop worker pool."""
        if self.executor:
            self.executor.shutdown(wait=True)

    async def detect_face(self, image_data: bytes) -> dict[str, Any]:
        """Async face detection using worker pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            _detect_face_sync,
            image_data,
        )
```

---

### Phase 4: Service Layer (1.5 hours)

#### 4.1 Image Processing Utilities (services/image_processing.py)

**Goal**: Eliminate code duplication (image loading repeated 5+ times in original)

**Principles Applied**:
- DRY (Don't Repeat Yourself)
- Single responsibility
- Type hints for clarity

**Implementation**:
```python
import base64
import io
from typing import BinaryIO

import cv2
import numpy as np
from fastapi import UploadFile

from app.core.exceptions import InvalidImageError


async def load_image_from_upload(file: UploadFile) -> bytes:
    """Load image from multipart upload."""
    try:
        content = await file.read()
        if not content:
            raise InvalidImageError("Empty file")
        return content
    except Exception as e:
        raise InvalidImageError(f"Failed to read uploaded file: {e}")


def load_image_from_base64(base64_string: str) -> bytes:
    """Load image from base64 string."""
    try:
        # Remove data URI prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        image_data = base64.b64decode(base64_string)
        if not image_data:
            raise InvalidImageError("Empty base64 data")

        return image_data
    except Exception as e:
        raise InvalidImageError(f"Failed to decode base64: {e}")


def validate_image(image_data: bytes) -> np.ndarray:
    """Validate and decode image data."""
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise InvalidImageError("Failed to decode image")

        return img
    except Exception as e:
        raise InvalidImageError(f"Invalid image data: {e}")
```

#### 4.2 Other Services

Similar implementations for:
- `services/face_detection.py` - InsightFace operations
- `services/face_analysis.py` - DeepFace operations
- `services/embedding.py` - Embedding comparison logic

---

### Phase 5: Pydantic Models (1 hour)

#### 5.1 Face Models (models/face.py)

**Principles Applied**:
- Comprehensive validation
- Field constraints and examples
- Separate models for different input types
- OpenAPI schema examples

**Implementation Examples**:
```python
from pydantic import BaseModel, Field, field_validator


class FaceVerificationRequest(BaseModel):
    """Request to verify if image contains a detectable face."""

    image: str = Field(
        ...,
        description="Base64-encoded image or multipart file",
        examples=["data:image/jpeg;base64,/9j/4AAQSkZJRg..."],
    )

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("Image cannot be empty")
        return v


class EmbeddingResponse(BaseModel):
    """Response containing face embedding."""

    embedding: list[float] = Field(
        ...,
        description="512-dimensional face embedding vector",
        min_length=512,
        max_length=512,
    )
    bbox: list[float] = Field(
        ...,
        description="Bounding box [x1, y1, x2, y2]",
        min_length=4,
        max_length=4,
    )
    confidence: float = Field(
        ...,
        description="Detection confidence score",
        ge=0.0,
        le=1.0,
    )


class ComparisonRequest(BaseModel):
    """Request to compare two face embeddings."""

    embedding_a: list[float] = Field(
        ...,
        description="First 512D embedding",
        min_length=512,
        max_length=512,
    )
    embedding_b: list[float] = Field(
        ...,
        description="Second 512D embedding",
        min_length=512,
        max_length=512,
    )
```

#### 5.2 Common Models (models/common.py)

```python
class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")
```

---

### Phase 6: FastAPI Endpoints (1.5 hours)

#### 6.1 Health Endpoint (api/v1/endpoints/health.py)

```python
from fastapi import APIRouter
from app.models.common import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint (no authentication required)."""
    return HealthResponse()
```

#### 6.2 Face Endpoints (api/v1/endpoints/face.py)

```python
from fastapi import APIRouter, Depends, File, UploadFile
from app.api.dependencies import get_ml_worker
from app.core.security import verify_api_key
from app.models.face import (
    EmbeddingResponse,
    FaceVerificationRequest,
    AdvancedAnalysisResponse,
)
from app.services.image_processing import load_image_from_upload
from app.workers.ml_worker import MLWorkerPool

router = APIRouter()


@router.post(
    "/verify-face",
    response_model=dict,
    tags=["Face Detection"],
    dependencies=[Depends(verify_api_key)],
)
async def verify_face(
    file: UploadFile = File(...),
    worker: MLWorkerPool = Depends(get_ml_worker),
):
    """Verify if uploaded image contains a detectable face."""
    image_data = await load_image_from_upload(file)
    result = await worker.detect_face(image_data)
    return {"face_detected": True, "confidence": result["det_score"]}


@router.post(
    "/extract-embedding",
    response_model=EmbeddingResponse,
    tags=["Face Detection"],
    dependencies=[Depends(verify_api_key)],
)
async def extract_embedding(
    file: UploadFile = File(...),
    worker: MLWorkerPool = Depends(get_ml_worker),
):
    """Extract 512D face embedding from uploaded image."""
    image_data = await load_image_from_upload(file)
    result = await worker.detect_face(image_data)

    return EmbeddingResponse(
        embedding=result["embedding"],
        bbox=result["bbox"],
        confidence=result["det_score"],
    )
```

#### 6.3 Router Setup (api/v1/router.py)

```python
from fastapi import APIRouter
from app.api.v1.endpoints import health, face, compare

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(face.router, prefix="/face", tags=["Face"])
api_router.include_router(compare.router, prefix="/compare", tags=["Compare"])
```

---

### Phase 7: Main Application (45 min)

#### 7.1 FastAPI App (app/main.py)

**Principles Applied**:
- Lifespan context manager for startup/shutdown
- Proper CORS configuration
- Global exception handlers
- API versioning with prefix

**Implementation**:
```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies import ml_worker_pool
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import (
    FaceDetectionError,
    InvalidImageError,
    face_detection_error_handler,
    invalid_image_error_handler,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    await ml_worker_pool.start()
    yield
    # Shutdown
    await ml_worker_pool.stop()


app = FastAPI(
    title="AI Face Matching Service",
    description="FastAPI service for face detection, verification, and analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
app.add_exception_handler(FaceDetectionError, face_detection_error_handler)
app.add_exception_handler(InvalidImageError, invalid_image_error_handler)

# Include routers
app.include_router(api_router, prefix="/api/v1")
```

---

### Phase 8: Docker Optimization (45 min)

#### 8.1 Multi-stage Dockerfile

**Optimizations**:
- Multi-stage build (builder + runtime)
- UV for fast dependency installation
- Layer caching
- Minimal runtime image

**Implementation**:
```dockerfile
# Builder stage
FROM python:3.11-slim AS builder

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies to /app/.venv
RUN uv sync --frozen --no-dev

# Runtime stage
FROM python:3.11-slim

# Install system dependencies for OpenCV and ONNX
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Fix onnxruntime executable stack issue
RUN apt-get update && apt-get install -y execstack \
    && find /usr/local/lib -name "*.so*" -exec execstack -c {} \; 2>/dev/null || true \
    && apt-get remove -y execstack \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY app /app/app

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health')"

# Run with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

#### 8.2 Updated .env.example

```
# API Configuration
API_KEY=your-secure-api-key-here
DEBUG=False

# Server Configuration
HOST=0.0.0.0
PORT=8000

# CORS Configuration (comma-separated)
CORS_ORIGINS=http://localhost:3000,https://your-frontend.com

# ML Configuration
FACE_DET_THRESH=0.5

# Worker Configuration
MAX_WORKERS=2
WORKER_TIMEOUT=120
```

---

## Migration Checklist

### Phase 1: UV Setup
- [ ] Install UV package manager
- [ ] Create `pyproject.toml`
- [ ] Migrate dependencies
- [ ] Generate `uv.lock`
- [ ] Update `.gitignore`

### Phase 2: Core Infrastructure
- [ ] Implement `core/config.py`
- [ ] Implement `core/security.py`
- [ ] Implement `core/exceptions.py`

### Phase 3: ML Process Pool
- [ ] Create `workers/ml_worker.py`
- [ ] Implement ProcessPoolExecutor
- [ ] Add async wrappers
- [ ] Test worker initialization

### Phase 4: Service Layer
- [ ] Extract `services/image_processing.py`
- [ ] Extract `services/face_detection.py`
- [ ] Extract `services/face_analysis.py`
- [ ] Extract `services/embedding.py`
- [ ] Verify DRY (no duplication)

### Phase 5: Pydantic Models
- [ ] Create `models/face.py` schemas
- [ ] Create `models/common.py` schemas
- [ ] Add validation rules
- [ ] Add OpenAPI examples

### Phase 6: FastAPI Endpoints
- [ ] Create `api/v1/endpoints/health.py`
- [ ] Create `api/v1/endpoints/face.py`
- [ ] Create `api/v1/endpoints/compare.py`
- [ ] Set up `api/v1/router.py`
- [ ] Create `api/dependencies.py`

### Phase 7: Main Application
- [ ] Implement `app/main.py`
- [ ] Configure CORS
- [ ] Add exception handlers
- [ ] Test lifespan events

### Phase 8: Docker & Deployment
- [ ] Create multi-stage `Dockerfile`
- [ ] Update `.dockerignore`
- [ ] Update `.env.example`
- [ ] Test Docker build
- [ ] Test Docker run

### Testing & Validation
- [ ] Test `/api/v1/health` endpoint
- [ ] Test `/api/v1/face/verify-face` endpoint
- [ ] Test `/api/v1/face/extract-embedding` endpoint
- [ ] Test `/api/v1/compare/compare-faces` endpoint
- [ ] Test `/api/v1/face/batch-extract` endpoint
- [ ] Test `/api/v1/face/analyze-face-advanced` endpoint
- [ ] Verify OpenAPI docs at `/docs`
- [ ] Test concurrent requests
- [ ] Verify authentication
- [ ] Test error handling

---

## Success Metrics

### Code Quality
- ✅ 9 modular files vs 1 monolithic file
- ✅ 100% type hints coverage
- ✅ Zero code duplication
- ✅ Separation of concerns (MVC-like)

### Performance
- ✅ Non-blocking ML inference via process pool
- ✅ 10-100x faster dependency installation
- ✅ Optimized Docker image layers
- ✅ Concurrent request handling

### Developer Experience
- ✅ Auto-generated API docs at `/docs`
- ✅ Automatic request/response validation
- ✅ Type-safe code (IDE autocomplete)
- ✅ Reproducible builds (uv.lock)

### Production Readiness
- ✅ Proper error handling
- ✅ Structured logging
- ✅ Health checks
- ✅ Docker optimization
- ✅ Environment-based configuration

---

## Post-Migration Tasks (Future)

1. **Testing**
   - Add pytest test suite
   - Unit tests for services
   - Integration tests for endpoints
   - Load testing

2. **Security Enhancements**
   - Add rate limiting (slowapi)
   - Request size limits
   - Input sanitization
   - API key rotation mechanism

3. **Observability**
   - Structured logging (structlog)
   - Metrics (Prometheus)
   - Tracing (OpenTelemetry)
   - Error monitoring (Sentry)

4. **CI/CD**
   - GitHub Actions workflow
   - Automated testing
   - Docker image building
   - Deployment automation

---

## References

- [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)
- [UV Documentation](https://github.com/astral-sh/uv)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
