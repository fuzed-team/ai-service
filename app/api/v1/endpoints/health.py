"""Health check endpoint."""

from fastapi import APIRouter

from app.models.common import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check if the service is running and healthy. Does not require authentication.",
)
async def health_check():
    """Health check endpoint (no authentication required)."""
    return HealthResponse(
        status="healthy",
        model="insightface",
        model_loaded=True,
        version="1.0.0",
    )
