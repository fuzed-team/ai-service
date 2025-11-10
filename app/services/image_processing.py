"""Image processing utilities (DRY - Don't Repeat Yourself)."""

import base64
import logging

import cv2
import numpy as np
from fastapi import UploadFile

from app.core.exceptions import InvalidImageError

logger = logging.getLogger(__name__)


async def load_image_from_upload(file: UploadFile) -> bytes:
    """
    Load image from multipart upload.

    Args:
        file: Uploaded file from FastAPI

    Returns:
        Raw image bytes

    Raises:
        InvalidImageError: If file is empty or cannot be read
    """
    try:
        content = await file.read()
        if not content:
            raise InvalidImageError("Empty file uploaded")
        logger.info(f"Loaded image from upload: {file.filename}")
        return content
    except Exception as e:
        raise InvalidImageError(f"Failed to read uploaded file: {e}")


def load_image_from_base64(base64_string: str) -> bytes:
    """
    Load image from base64 string.

    Args:
        base64_string: Base64-encoded image (with or without data URI prefix)

    Returns:
        Raw image bytes

    Raises:
        InvalidImageError: If base64 is invalid or empty
    """
    try:
        # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,")
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        image_data = base64.b64decode(base64_string)
        if not image_data:
            raise InvalidImageError("Empty base64 data")

        logger.info("Loaded image from base64")
        return image_data
    except Exception as e:
        raise InvalidImageError(f"Failed to decode base64: {e}")


def validate_image(image_data: bytes) -> np.ndarray:
    """
    Validate and decode image data to numpy array.

    Args:
        image_data: Raw image bytes

    Returns:
        Decoded image as numpy array (BGR format)

    Raises:
        InvalidImageError: If image cannot be decoded
    """
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise InvalidImageError("Failed to decode image - invalid format")

        logger.debug(f"Image validated: shape={img.shape}")
        return img
    except Exception as e:
        raise InvalidImageError(f"Invalid image data: {e}")


def encode_image_to_base64(img: np.ndarray, format: str = ".jpg") -> str:
    """
    Encode numpy array image to base64 string.

    Args:
        img: Image as numpy array
        format: Image format (default: .jpg)

    Returns:
        Base64-encoded image string

    Raises:
        InvalidImageError: If encoding fails
    """
    try:
        success, buffer = cv2.imencode(format, img)
        if not success:
            raise InvalidImageError("Failed to encode image")

        base64_str = base64.b64encode(buffer).decode("utf-8")
        return base64_str
    except Exception as e:
        raise InvalidImageError(f"Failed to encode image to base64: {e}")
