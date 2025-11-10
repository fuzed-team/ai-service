"""Face embedding comparison endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import verify_api_key
from app.models.face import ComparisonRequest, ComparisonResponse
from app.services.embedding import calculate_cosine_similarity

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/compare-faces",
    response_model=ComparisonResponse,
    tags=["Comparison"],
    summary="Compare face embeddings",
    description="Compare two 512D face embeddings and return similarity score.",
    dependencies=[Depends(verify_api_key)],
)
async def compare_faces(request: ComparisonRequest):
    """
    Compare two face embeddings and return similarity score.

    Args:
        request: Comparison request with two embeddings

    Returns:
        Similarity and distance scores

    Raises:
        HTTPException: If embeddings are invalid
    """
    try:
        # Calculate cosine similarity
        similarity, distance = calculate_cosine_similarity(
            request.embedding_a,
            request.embedding_b,
        )

        logger.info(f"Compared embeddings: similarity={similarity:.3f}")

        return ComparisonResponse(
            similarity=similarity,
            distance=distance,
        )

    except ValueError as e:
        logger.warning(f"Comparison failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error comparing faces: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )
