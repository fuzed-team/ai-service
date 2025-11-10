"""ML worker pool for CPU-intensive inference tasks."""

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.core.config import settings

logger = logging.getLogger(__name__)

# Worker-local model instance (initialized per process)
_face_app: FaceAnalysis | None = None


def _init_worker():
    """Initialize ML models in worker process."""
    global _face_app
    logger.info("Initializing InsightFace model in worker process...")
    _face_app = FaceAnalysis(providers=["CPUExecutionProvider"])
    _face_app.prepare(ctx_id=0, det_size=settings.det_size)
    logger.info("InsightFace model loaded successfully in worker")


def _detect_face_sync(image_data: bytes) -> dict[str, Any]:
    """
    Synchronous face detection (runs in worker process).

    Args:
        image_data: Raw image bytes

    Returns:
        Dictionary containing embedding, bbox, and confidence score

    Raises:
        ValueError: If image is invalid or no face detected
    """
    if _face_app is None:
        raise ValueError("Face app not initialized in worker")

    # Decode image
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image")

    # Detect faces
    faces = _face_app.get(img)

    if len(faces) == 0:
        raise ValueError("No face detected in image")

    # Get first face (highest confidence)
    face = faces[0]

    return {
        "embedding": face.embedding.tolist(),
        "bbox": face.bbox.tolist(),
        "det_score": float(face.det_score),
        "age": int(face.age) if hasattr(face, "age") else None,
        "gender": "male" if (hasattr(face, "gender") and face.gender == 1) else "female",
    }


def _analyze_face_advanced_sync(image_data: bytes) -> dict[str, Any]:
    """
    Advanced face analysis with comprehensive attributes (runs in worker process).

    Args:
        image_data: Raw image bytes

    Returns:
        Dictionary containing comprehensive facial analysis

    Raises:
        ValueError: If image is invalid or no face detected
    """
    if _face_app is None:
        raise ValueError("Face app not initialized in worker")

    # Decode image
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image")

    # Detect faces
    faces = _face_app.get(img)

    if len(faces) == 0:
        raise ValueError("No face detected in image")

    # Get first face
    face = faces[0]

    # Extract basic attributes
    result = {
        "embedding": face.embedding.tolist(),
        "bbox": face.bbox.tolist(),
        "det_score": float(face.det_score),
        "age": int(face.age) if hasattr(face, "age") else 25,
        "gender": "male" if (hasattr(face, "gender") and face.gender == 1) else "female",
    }

    # Extract landmarks
    if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        result["landmarks_68"] = face.landmark_2d_106[:68].tolist()
    elif hasattr(face, "kps") and face.kps is not None:
        result["landmarks_68"] = face.kps.tolist()
    else:
        result["landmarks_68"] = None

    # Extract pose
    if hasattr(face, "pose") and face.pose is not None:
        pose_array = face.pose if isinstance(face.pose, (list, np.ndarray)) else [0, 0, 0]
        result["pose"] = {
            "yaw": float(pose_array[0]) if len(pose_array) > 0 else 0.0,
            "pitch": float(pose_array[1]) if len(pose_array) > 1 else 0.0,
            "roll": float(pose_array[2]) if len(pose_array) > 2 else 0.0,
        }
    else:
        result["pose"] = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    # Store image for additional processing
    result["_image"] = img  # Will be used by services layer

    return result


class MLWorkerPool:
    """Process pool for ML inference."""

    def __init__(self, max_workers: int | None = None):
        """
        Initialize worker pool.

        Args:
            max_workers: Maximum number of worker processes (default from settings)
        """
        self.max_workers = max_workers or settings.max_workers
        self.executor: ProcessPoolExecutor | None = None
        logger.info(f"MLWorkerPool initialized with max_workers={self.max_workers}")

    async def start(self):
        """Start worker pool."""
        logger.info("Starting ML worker pool...")
        self.executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_init_worker,
        )
        logger.info("ML worker pool started successfully")

    async def stop(self):
        """Stop worker pool."""
        logger.info("Stopping ML worker pool...")
        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("ML worker pool stopped")

    async def detect_face(self, image_data: bytes) -> dict[str, Any]:
        """
        Async face detection using worker pool.

        Args:
            image_data: Raw image bytes

        Returns:
            Dictionary containing face detection results

        Raises:
            RuntimeError: If executor is not initialized
            ValueError: If face detection fails
        """
        if not self.executor:
            raise RuntimeError("Worker pool not initialized. Call start() first.")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            _detect_face_sync,
            image_data,
        )

    async def analyze_face_advanced(self, image_data: bytes) -> dict[str, Any]:
        """
        Async advanced face analysis using worker pool.

        Args:
            image_data: Raw image bytes

        Returns:
            Dictionary containing comprehensive facial analysis

        Raises:
            RuntimeError: If executor is not initialized
            ValueError: If analysis fails
        """
        if not self.executor:
            raise RuntimeError("Worker pool not initialized. Call start() first.")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            _analyze_face_advanced_sync,
            image_data,
        )


# Global worker pool instance
ml_worker_pool = MLWorkerPool()
