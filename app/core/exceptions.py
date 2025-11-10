"""Custom exceptions and exception handlers."""

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


class ModelNotLoadedError(Exception):
    """Raised when ML model is not loaded."""

    pass


# Exception handlers


async def face_detection_error_handler(
    request: Request, exc: FaceDetectionError
) -> JSONResponse:
    """Handle face detection errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Face detection failed", "detail": str(exc)},
    )


async def invalid_image_error_handler(
    request: Request, exc: InvalidImageError
) -> JSONResponse:
    """Handle invalid image errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid image", "detail": str(exc)},
    )


async def embedding_extraction_error_handler(
    request: Request, exc: EmbeddingExtractionError
) -> JSONResponse:
    """Handle embedding extraction errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Embedding extraction failed", "detail": str(exc)},
    )


async def model_not_loaded_error_handler(
    request: Request, exc: ModelNotLoadedError
) -> JSONResponse:
    """Handle model not loaded errors."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "ML model not loaded", "detail": str(exc)},
    )
