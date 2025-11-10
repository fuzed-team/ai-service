"""Advanced facial analysis services using DeepFace and CV algorithms."""

import logging

import cv2
import numpy as np
from deepface import DeepFace
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def calculate_blur_score(image: np.ndarray, bbox: list[float]) -> float:
    """
    Detect image blur using Laplacian variance.

    Args:
        image: Input image as numpy array
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        Blur score 0.0-1.0 (1.0 = sharp, 0.0 = very blurry)
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        face_region = image[y1:y2, x1:x2]

        if face_region.size == 0:
            return 0.5

        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize: variance > 500 is sharp, < 100 is blurry
        blur_score = min(variance / 500.0, 1.0)

        return float(blur_score)

    except Exception as e:
        logger.warning(f"Error calculating blur: {e}")
        return 0.5


def calculate_illumination(image: np.ndarray, bbox: list[float]) -> float:
    """
    Check lighting quality using histogram analysis.

    Args:
        image: Input image as numpy array
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        Illumination score 0.0-1.0 (1.0 = well-lit, 0.0 = poor lighting)
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        face_region = image[y1:y2, x1:x2]

        if face_region.size == 0:
            return 0.5

        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Calculate brightness statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # Good lighting: mean around 100-150, std > 30
        contrast_score = min(std_brightness / 50.0, 1.0)
        brightness_score = 1.0 - abs(mean_brightness - 130.0) / 130.0

        illumination = contrast_score * max(brightness_score, 0.0)

        return float(max(illumination, 0.0))

    except Exception as e:
        logger.warning(f"Error calculating illumination: {e}")
        return 0.5


def extract_skin_tone(image: np.ndarray, bbox: list[float]) -> list[float]:
    """
    Extract dominant skin color using K-means clustering in CIELAB color space.

    Args:
        image: Input image as numpy array
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        [L, a, b] values in CIELAB color space
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        face_region = image[y1:y2, x1:x2]

        if face_region.size == 0:
            return [65.0, 10.0, 20.0]  # Default skin tone

        # Convert to LAB color space
        lab_image = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)

        # Reshape for K-means
        pixels = lab_image.reshape(-1, 3).astype(np.float32)

        # Use K-means to find 3 dominant colors
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10, max_iter=100)
        kmeans.fit(pixels)

        # Get the dominant color
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_cluster = labels[np.argmax(counts)]
        dominant_color = kmeans.cluster_centers_[dominant_cluster]

        return dominant_color.tolist()

    except Exception as e:
        logger.warning(f"Error extracting skin tone: {e}")
        return [65.0, 10.0, 20.0]


def lab_to_hex(lab: list[float]) -> str:
    """
    Convert LAB color to hex.

    Args:
        lab: [L, a, b] values in CIELAB

    Returns:
        Hex color string (e.g., "#d4a373")
    """
    try:
        lab_pixel = np.uint8([[lab]])
        bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
        b, g, r = bgr_pixel[0][0]
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return "#d4a373"  # Default skin color


def detect_emotion(image: np.ndarray, bbox: list[float]) -> tuple[str, dict[str, float]]:
    """
    Detect facial expression using DeepFace.

    Args:
        image: Input image as numpy array
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        Tuple of (dominant_emotion, emotion_scores_dict)
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        face_region = image[y1:y2, x1:x2]

        if face_region.size == 0:
            return "neutral", {"neutral": 1.0}

        # Analyze emotion
        result = DeepFace.analyze(
            face_region,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )

        emotions = result[0]["emotion"]
        dominant = result[0]["dominant_emotion"]

        # Normalize emotion scores to 0-1
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v / 100.0 for k, v in emotions.items()}

        return dominant, emotions

    except Exception as e:
        logger.warning(f"Error detecting emotion: {e}")
        return "neutral", {"neutral": 1.0}


def calculate_symmetry_score(landmarks: list[list[float]] | None) -> float:
    """
    Calculate facial symmetry by comparing left vs right features.

    Args:
        landmarks: List of [x, y] landmark points

    Returns:
        Symmetry score 0.0-1.0 (1.0 = perfect symmetry)
    """
    if landmarks is None or len(landmarks) < 68:
        return 0.75  # Default moderate symmetry

    try:
        landmarks = np.array(landmarks)

        # Split into left and right halves
        left_half = landmarks[:34]
        right_half = landmarks[34:68]

        # Mirror right half horizontally
        mirrored_right = np.copy(right_half)
        center_x = np.mean(landmarks[:, 0])
        mirrored_right[:, 0] = 2 * center_x - mirrored_right[:, 0]

        # Calculate average distance
        if len(left_half) != len(mirrored_right):
            min_len = min(len(left_half), len(mirrored_right))
            left_half = left_half[:min_len]
            mirrored_right = mirrored_right[:min_len]

        distance = np.mean(np.linalg.norm(left_half - mirrored_right, axis=1))

        # Normalize: typical face width is ~200px, max asymmetry ~50px
        symmetry = 1.0 - min(distance / 50.0, 1.0)

        return float(max(symmetry, 0.0))

    except Exception as e:
        logger.warning(f"Error calculating symmetry: {e}")
        return 0.75


def calculate_geometry_ratios(landmarks: list[list[float]] | None) -> dict[str, float]:
    """
    Calculate facial proportions from landmarks.

    Args:
        landmarks: List of [x, y] landmark points

    Returns:
        Dictionary of key facial ratios
    """
    if landmarks is None or len(landmarks) < 68:
        return {
            "face_width_height_ratio": 0.75,
            "eye_spacing_face_width": 0.42,
            "jawline_width_face_width": 0.68,
            "nose_width_face_width": 0.25,
        }

    try:
        landmarks = np.array(landmarks)

        # Calculate key dimensions (68-point landmark indices)
        face_width = np.linalg.norm(landmarks[16] - landmarks[0])

        if len(landmarks) > 27:
            face_height = np.linalg.norm(landmarks[8] - landmarks[27])
        else:
            face_height = face_width * 1.3

        if len(landmarks) > 45:
            eye_spacing = np.linalg.norm(landmarks[45] - landmarks[36])
        else:
            eye_spacing = face_width * 0.42

        if len(landmarks) > 14:
            jawline_width = np.linalg.norm(landmarks[14] - landmarks[2])
        else:
            jawline_width = face_width * 0.68

        if len(landmarks) > 35:
            nose_width = np.linalg.norm(landmarks[35] - landmarks[31])
        else:
            nose_width = face_width * 0.25

        return {
            "face_width_height_ratio": float(face_width / max(face_height, 1)),
            "eye_spacing_face_width": float(eye_spacing / max(face_width, 1)),
            "jawline_width_face_width": float(jawline_width / max(face_width, 1)),
            "nose_width_face_width": float(nose_width / max(face_width, 1)),
        }

    except Exception as e:
        logger.warning(f"Error calculating geometry ratios: {e}")
        return {
            "face_width_height_ratio": 0.75,
            "eye_spacing_face_width": 0.42,
            "jawline_width_face_width": 0.68,
            "nose_width_face_width": 0.25,
        }
