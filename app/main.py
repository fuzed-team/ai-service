"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies import ml_worker_pool
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import (
    EmbeddingExtractionError,
    FaceDetectionError,
    InvalidImageError,
    ModelNotLoadedError,
    embedding_extraction_error_handler,
    face_detection_error_handler,
    invalid_image_error_handler,
    model_not_loaded_error_handler,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle (startup and shutdown).

    This handles initialization and cleanup of the ML worker pool.
    """
    # Startup
    logger.info("Starting up AI Face Matching Service...")
    await ml_worker_pool.start()
    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down AI Face Matching Service...")
    await ml_worker_pool.stop()
    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="FastAPI service for AI-powered face detection, verification, and analysis using InsightFace and DeepFace",
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register exception handlers
app.add_exception_handler(FaceDetectionError, face_detection_error_handler)
app.add_exception_handler(InvalidImageError, invalid_image_error_handler)
app.add_exception_handler(EmbeddingExtractionError, embedding_extraction_error_handler)
app.add_exception_handler(ModelNotLoadedError, model_not_loaded_error_handler)

# Include API routers
app.include_router(api_router, prefix="/api/v1")

logger.info(f"FastAPI application initialized (debug={settings.debug})")
