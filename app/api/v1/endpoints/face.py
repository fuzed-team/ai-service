"""Face detection and analysis endpoints."""

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import get_ml_worker
from app.core.exceptions import FaceDetectionError, InvalidImageError
from app.core.security import verify_api_key
from app.models.face import (
    AdvancedAnalysisResponse,
    Base64ImageRequest,
    BatchExtractRequest,
    BatchExtractResponse,
    BatchExtractResult,
    EmbeddingResponse,
    Expression,
    FaceVerificationResponse,
    GeometryRatios,
    PoseMetrics,
    QualityMetrics,
    SkinTone,
)
from app.services.face_analysis import (
    calculate_blur_score,
    calculate_geometry_ratios,
    calculate_illumination,
    calculate_symmetry_score,
    detect_emotion,
    extract_skin_tone,
    lab_to_hex,
)
from app.services.image_processing import load_image_from_base64, load_image_from_upload
from app.workers.ml_worker import MLWorkerPool

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/verify-face",
    response_model=FaceVerificationResponse,
    tags=["Face Detection"],
    summary="Verify face presence",
    description="Verify if an image contains a detectable face without extracting embedding.",
    dependencies=[Depends(verify_api_key)],
)
async def verify_face(
    file: UploadFile = File(..., description="Image file (JPEG, PNG)"),
    worker: MLWorkerPool = Depends(get_ml_worker),
):
    """Verify if uploaded image contains a detectable face."""
    try:
        # Load image
        image_data = await load_image_from_upload(file)

        # Detect face
        result = await worker.detect_face(image_data)

        logger.info(f"Face verified with confidence: {result['det_score']:.3f}")

        return FaceVerificationResponse(
            face_detected=True,
            confidence=result["det_score"],
            bbox=result["bbox"],
            message="Face detected successfully",
        )

    except ValueError as e:
        logger.warning(f"Face verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error verifying face: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post(
    "/extract-embedding",
    response_model=EmbeddingResponse,
    tags=["Face Detection"],
    summary="Extract face embedding",
    description="Extract 512-dimensional face embedding from uploaded image.",
    dependencies=[Depends(verify_api_key)],
)
async def extract_embedding(
    file: UploadFile = File(None, description="Image file (JPEG, PNG)"),
    image_base64: str = Form(None, description="Base64-encoded image (alternative to file)"),
    worker: MLWorkerPool = Depends(get_ml_worker),
):
    """Extract 512D face embedding from uploaded image or base64."""
    try:
        # Load image from file or base64
        if file:
            image_data = await load_image_from_upload(file)
        elif image_base64:
            image_data = load_image_from_base64(image_base64)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'file' or 'image_base64' must be provided",
            )

        # Extract embedding
        result = await worker.detect_face(image_data)

        logger.info(f"Embedding extracted with confidence: {result['det_score']:.3f}")

        return EmbeddingResponse(
            face_detected=True,
            embedding=result["embedding"],
            bbox=result["bbox"],
            confidence=result["det_score"],
        )

    except ValueError as e:
        logger.warning(f"Embedding extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post(
    "/batch-extract",
    response_model=BatchExtractResponse,
    tags=["Face Detection"],
    summary="Batch extract embeddings",
    description="Extract embeddings from multiple images in a single request.",
    dependencies=[Depends(verify_api_key)],
)
async def batch_extract_embeddings(
    request: BatchExtractRequest,
    worker: MLWorkerPool = Depends(get_ml_worker),
):
    """Extract embeddings from multiple base64-encoded images."""
    results = []
    successful = 0
    failed = 0

    for idx, img_base64 in enumerate(request.images):
        try:
            # Load and process image
            image_data = load_image_from_base64(img_base64)
            result = await worker.detect_face(image_data)

            results.append(
                BatchExtractResult(
                    index=idx,
                    face_detected=True,
                    embedding=result["embedding"],
                    bbox=result["bbox"],
                    confidence=result["det_score"],
                    error=None,
                )
            )
            successful += 1

        except Exception as e:
            logger.warning(f"Batch item {idx} failed: {e}")
            results.append(
                BatchExtractResult(
                    index=idx,
                    face_detected=False,
                    embedding=None,
                    bbox=None,
                    confidence=None,
                    error=str(e),
                )
            )
            failed += 1

    logger.info(f"Batch processing: {successful}/{len(request.images)} successful")

    return BatchExtractResponse(
        results=results,
        total=len(request.images),
        successful=successful,
        failed=failed,
    )


@router.post(
    "/analyze-face-advanced",
    response_model=AdvancedAnalysisResponse,
    tags=["Face Analysis"],
    summary="Advanced facial analysis",
    description="Extract comprehensive facial attributes including age, gender, emotions, quality metrics, and more.",
    dependencies=[Depends(verify_api_key)],
)
async def analyze_face_advanced(
    file: UploadFile = File(None, description="Image file (JPEG, PNG)"),
    image_base64: str = Form(None, description="Base64-encoded image (alternative to file)"),
    worker: MLWorkerPool = Depends(get_ml_worker),
):
    """Extract comprehensive facial attributes for advanced matching."""
    try:
        # Load image
        if file:
            image_data = await load_image_from_upload(file)
        elif image_base64:
            image_data = load_image_from_base64(image_base64)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'file' or 'image_base64' must be provided",
            )

        # Perform advanced analysis
        result = await worker.analyze_face_advanced(image_data)

        # Extract stored image for additional processing
        img = result.pop("_image")
        bbox = result["bbox"]
        landmarks = result["landmarks_68"]

        # Calculate quality metrics
        blur_score = calculate_blur_score(img, bbox)
        illumination = calculate_illumination(img, bbox)
        overall_quality = (blur_score + illumination) / 2.0

        # Calculate symmetry
        symmetry_score = calculate_symmetry_score(landmarks)

        # Extract skin tone
        skin_tone_lab = extract_skin_tone(img, bbox)
        hex_color = lab_to_hex(skin_tone_lab)

        # Detect emotion
        dominant_emotion, emotion_scores = detect_emotion(img, bbox)

        # Calculate geometry
        geometry = calculate_geometry_ratios(landmarks)

        logger.info(
            f"Advanced analysis complete: age={result['age']}, "
            f"gender={result['gender']}, expression={dominant_emotion}, "
            f"quality={overall_quality:.2f}"
        )

        return AdvancedAnalysisResponse(
            face_detected=True,
            embedding=result["embedding"],
            bbox=result["bbox"],
            confidence=result["det_score"],
            age=result["age"],
            gender=result["gender"],
            landmarks_68=landmarks,
            pose=PoseMetrics(**result["pose"]),
            quality=QualityMetrics(
                blur_score=blur_score,
                illumination=illumination,
                overall=overall_quality,
            ),
            symmetry_score=symmetry_score,
            skin_tone=SkinTone(dominant_color_lab=skin_tone_lab, hex=hex_color),
            expression=Expression(
                dominant=dominant_emotion,
                confidence=emotion_scores.get(dominant_emotion, 0.5),
                emotions=emotion_scores,
            ),
            geometry=GeometryRatios(**geometry),
        )

    except ValueError as e:
        logger.warning(f"Advanced analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in advanced analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )
