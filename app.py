"""
Python AI Microservice - Face Recognition API

Minimal Flask API for face embedding extraction using InsightFace.
This service is called by Next.js API routes for AI-powered face matching.

Endpoints:
    GET  /health - Health check
    POST /verify-face - Verify if face is detected in image
    POST /extract-embedding - Extract face embedding from image
    POST /compare-faces - Compare two face embeddings
    POST /batch-extract - Batch extract embeddings from multiple images
    POST /analyze-face-advanced - NEW: Extract comprehensive facial attributes
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import cv2
import os
import base64
import logging
from sklearn.cluster import KMeans
from deepface import DeepFace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Security: Require API key for all endpoints
API_KEY = os.getenv("API_KEY", "change-me-in-production")

# Global model variable - will be loaded on first request
face_app = None
model_loading = False
model_loaded = False


def get_face_app():
    """
    Lazy load InsightFace model on first request
    This allows the server to start quickly and respond to health checks
    """
    global face_app, model_loading, model_loaded

    if model_loaded:
        return face_app

    if model_loading:
        # Wait for model to finish loading (up to 120 seconds)
        import time
        for _ in range(120):
            if model_loaded:
                return face_app
            time.sleep(1)
        raise Exception("Model loading timeout")

    # Start loading model
    model_loading = True
    logger.info("Loading InsightFace model (this may take 30-60 seconds)...")
    face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    model_loaded = True
    model_loading = False
    logger.info("✓ InsightFace model loaded successfully!")

    return face_app


def verify_api_key():
    """
    Verify API key from Authorization header

    Expected format: Authorization: Bearer <api-key>
    """
    auth_header = request.headers.get('Authorization')

    if not auth_header or not auth_header.startswith('Bearer '):
        logger.warning("Missing or invalid Authorization header")
        return False

    token = auth_header.split(' ')[1]

    if token != API_KEY:
        logger.warning("Invalid API key provided")
        return False

    return True


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring

    Returns:
        JSON with status and model info
    """
    return jsonify({
        "status": "healthy",
        "model": "insightface",
        "model_loaded": model_loaded,
        "model_loading": model_loading,
        "version": "1.0.0"
    })


@app.route('/extract-embedding', methods=['POST'])
def extract_embedding():
    """
    Extract 512D face embedding from uploaded image

    Request formats:
        1. Multipart form data with 'file' field (JPEG, PNG)
        2. JSON with 'image_base64' field (base64-encoded image)

    Response:
        {
            "face_detected": true,
            "embedding": [512 float values],
            "bbox": [x, y, width, height],
            "confidence": 0.99
        }

    Error Response:
        {
            "face_detected": false,
            "error": "No face detected in image"
        }
    """
    # Verify API key
    if not verify_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    try:
        # Handle multipart file upload
        if 'file' in request.files:
            file = request.files['file']
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            logger.info(f"Processing uploaded file: {file.filename}")

        # Handle base64 image
        elif request.json and 'image_base64' in request.json:
            img_data = base64.b64decode(request.json['image_base64'])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            logger.info("Processing base64 image")

        else:
            return jsonify({"error": "No image provided"}), 400

        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Get face detection model (lazy load on first call)
        face_detector = get_face_app()

        # Detect faces in image
        faces = face_detector.get(img)

        if len(faces) == 0:
            logger.warning("No face detected in image")
            return jsonify({
                "face_detected": False,
                "error": "No face detected in image"
            }), 400

        # Use first detected face (highest confidence)
        face = faces[0]
        embedding = face.embedding.tolist()  # Convert numpy array to list

        logger.info(f"✓ Face detected with confidence: {face.det_score:.3f}")

        return jsonify({
            "face_detected": True,
            "embedding": embedding,
            "bbox": face.bbox.tolist(),
            "confidence": float(face.det_score)
        })

    except Exception as e:
        logger.error(f"Error extracting embedding: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/verify-face', methods=['POST'])
def verify_face():
    """
    Verify if a face is detected in an image (without extracting embedding)

    Request formats:
        1. Multipart form data with 'file' field (JPEG, PNG)
        2. JSON with 'image_base64' field (base64-encoded image)

    Response:
        {
            "face_detected": true,
            "confidence": 0.99,
            "bbox": [x, y, width, height],
            "message": "Face detected successfully"
        }

    Error Response:
        {
            "face_detected": false,
            "error": "No face detected in image"
        }
    """
    # Verify API key
    if not verify_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    try:
        # Handle multipart file upload
        if 'file' in request.files:
            file = request.files['file']
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            logger.info(f"Verifying face in uploaded file: {file.filename}")

        # Handle base64 image
        elif request.json and 'image_base64' in request.json:
            img_data = base64.b64decode(request.json['image_base64'])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            logger.info("Verifying face in base64 image")

        else:
            return jsonify({"error": "No image provided"}), 400

        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Get face detection model (lazy load on first call)
        face_detector = get_face_app()

        # Detect faces in image
        faces = face_detector.get(img)

        if len(faces) == 0:
            logger.warning("No face detected in image")
            return jsonify({
                "face_detected": False,
                "error": "No face detected in image"
            }), 400

        # Use first detected face (highest confidence)
        face = faces[0]

        logger.info(f"✓ Face verified with confidence: {face.det_score:.3f}")

        return jsonify({
            "face_detected": True,
            "confidence": float(face.det_score),
            "bbox": face.bbox.tolist(),
            "message": "Face detected successfully"
        })

    except Exception as e:
        logger.error(f"Error verifying face: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/compare-faces', methods=['POST'])
def compare_faces():
    """
    Compare two face embeddings and return similarity score

    Request body (JSON):
        {
            "embedding_a": [512 floats],
            "embedding_b": [512 floats]
        }

    Response:
        {
            "similarity": 0.85,  # 0-1, higher = more similar
            "distance": 0.15     # cosine distance
        }
    """
    # Verify API key
    if not verify_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.json

        if not data or 'embedding_a' not in data or 'embedding_b' not in data:
            return jsonify({"error": "Missing embeddings in request"}), 400

        emb_a = np.array(data['embedding_a'])
        emb_b = np.array(data['embedding_b'])

        # Validate embedding dimensions
        if len(emb_a) != 512 or len(emb_b) != 512:
            return jsonify({"error": "Embeddings must be 512-dimensional"}), 400

        # Compute cosine similarity
        # Cosine similarity = dot(a, b) / (||a|| * ||b||)
        similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
        distance = 1 - similarity

        logger.info(f"Compared embeddings: similarity={similarity:.3f}")

        return jsonify({
            "similarity": float(similarity),
            "distance": float(distance)
        })

    except Exception as e:
        logger.error(f"Error comparing faces: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/batch-extract', methods=['POST'])
def batch_extract_embeddings():
    """
    Extract embeddings from multiple images in batch

    Request body (JSON):
        {
            "images": [
                "base64_image_1",
                "base64_image_2",
                ...
            ]
        }

    Response:
        {
            "results": [
                {
                    "index": 0,
                    "face_detected": true,
                    "embedding": [...],
                    "confidence": 0.99
                },
                ...
            ],
            "total": 10,
            "successful": 8,
            "failed": 2
        }
    """
    # Verify API key
    if not verify_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.json

        if not data or 'images' not in data:
            return jsonify({"error": "Missing images array"}), 400

        images = data['images']
        results = []
        successful = 0
        failed = 0

        for idx, img_base64 in enumerate(images):
            try:
                # Decode image
                img_data = base64.b64decode(img_base64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Get face detection model (lazy load on first call)
                face_detector = get_face_app()

                # Detect faces
                faces = face_detector.get(img)

                if len(faces) == 0:
                    results.append({
                        "index": idx,
                        "face_detected": False,
                        "error": "No face detected"
                    })
                    failed += 1
                    continue

                # Get first face
                face = faces[0]

                results.append({
                    "index": idx,
                    "face_detected": True,
                    "embedding": face.embedding.tolist(),
                    "bbox": face.bbox.tolist(),
                    "confidence": float(face.det_score)
                })
                successful += 1

            except Exception as e:
                results.append({
                    "index": idx,
                    "face_detected": False,
                    "error": str(e)
                })
                failed += 1

        logger.info(f"Batch processing: {successful}/{len(images)} successful")

        return jsonify({
            "results": results,
            "total": len(images),
            "successful": successful,
            "failed": failed
        })

    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ===== ADVANCED FACIAL ANALYSIS HELPERS =====

def extract_landmarks_68(face):
    """Extract 68-point facial landmarks from InsightFace face object"""
    # InsightFace provides 106 landmarks, we use first 68 for compatibility
    if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
        return face.landmark_2d_106[:68].tolist()
    elif hasattr(face, 'kps') and face.kps is not None:
        # Fallback to 5-point landmarks if 106 not available
        return face.kps.tolist()
    return None


def calculate_symmetry_score(landmarks):
    """
    Calculate facial symmetry by comparing left vs right features
    Returns: 0.0-1.0 (1.0 = perfect symmetry)
    """
    if landmarks is None or len(landmarks) < 68:
        return 0.75  # Default moderate symmetry if landmarks unavailable

    landmarks = np.array(landmarks)

    # Split landmarks into left and right halves
    # Approximate split (landmarks 0-33 = left, 34-67 = right for 68-point model)
    left_half = landmarks[:34]
    right_half = landmarks[34:68]

    # Mirror right half horizontally
    mirrored_right = np.copy(right_half)
    center_x = np.mean(landmarks[:, 0])
    mirrored_right[:, 0] = 2 * center_x - mirrored_right[:, 0]

    # Calculate average distance between mirrored halves
    if len(left_half) != len(mirrored_right):
        # If different sizes, use minimum
        min_len = min(len(left_half), len(mirrored_right))
        left_half = left_half[:min_len]
        mirrored_right = mirrored_right[:min_len]

    distance = np.mean(np.linalg.norm(left_half - mirrored_right, axis=1))

    # Normalize: typical face width is ~200px, max asymmetry ~50px
    symmetry = 1.0 - min(distance / 50.0, 1.0)

    return float(max(symmetry, 0.0))


def extract_skin_tone(image, bbox):
    """
    Extract dominant skin color using K-means clustering in CIELAB color space
    Returns: [L, a, b] values in CIELAB
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)

        # Extract face region
        face_region = image[y1:y2, x1:x2]

        if face_region.size == 0:
            return [65.0, 10.0, 20.0]  # Default skin tone

        # Convert to LAB color space (perceptually uniform)
        lab_image = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)

        # Reshape for K-means
        pixels = lab_image.reshape(-1, 3).astype(np.float32)

        # Use K-means to find 3 dominant colors
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10, max_iter=100)
        kmeans.fit(pixels)

        # Get the dominant color (assume largest cluster)
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_cluster = labels[np.argmax(counts)]
        dominant_color = kmeans.cluster_centers_[dominant_cluster]

        return dominant_color.tolist()

    except Exception as e:
        logger.warning(f"Error extracting skin tone: {e}")
        return [65.0, 10.0, 20.0]  # Default skin tone


def calculate_blur_score(image, bbox):
    """
    Detect image blur using Laplacian variance
    Returns: 0.0-1.0 (1.0 = sharp, 0.0 = very blurry)
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


def calculate_illumination(image, bbox):
    """
    Check lighting quality using histogram analysis
    Returns: 0.0-1.0 (1.0 = well-lit, 0.0 = poor lighting)
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
        # Score based on standard deviation (contrast) and reasonable brightness
        contrast_score = min(std_brightness / 50.0, 1.0)
        brightness_score = 1.0 - abs(mean_brightness - 130.0) / 130.0

        illumination = contrast_score * max(brightness_score, 0.0)

        return float(max(illumination, 0.0))

    except Exception as e:
        logger.warning(f"Error calculating illumination: {e}")
        return 0.5


def detect_emotion(image, bbox):
    """
    Detect facial expression using DeepFace
    Returns: (dominant_emotion, emotion_scores_dict)
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        face_region = image[y1:y2, x1:x2]

        if face_region.size == 0:
            return "neutral", {"neutral": 1.0}

        # Analyze emotion
        result = DeepFace.analyze(
            face_region,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )

        emotions = result[0]['emotion']
        dominant = result[0]['dominant_emotion']

        # Normalize emotion scores to 0-1
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/100.0 for k, v in emotions.items()}

        return dominant, emotions

    except Exception as e:
        logger.warning(f"Error detecting emotion: {e}")
        return "neutral", {"neutral": 1.0}


def calculate_geometry_ratios(landmarks):
    """
    Calculate facial proportions from landmarks
    Returns: Dictionary of key ratios
    """
    if landmarks is None or len(landmarks) < 68:
        # Return default ratios if landmarks unavailable
        return {
            "face_width_height_ratio": 0.75,
            "eye_spacing_face_width": 0.42,
            "jawline_width_face_width": 0.68,
            "nose_width_face_width": 0.25
        }

    try:
        landmarks = np.array(landmarks)

        # Calculate key facial dimensions (68-point landmark indices)
        # Face outline: 0-16 (left to right jawline)
        face_width = np.linalg.norm(landmarks[16] - landmarks[0])

        # Face height: chin (8) to forehead (approximate from eye level)
        if len(landmarks) > 27:
            face_height = np.linalg.norm(landmarks[8] - landmarks[27])
        else:
            face_height = face_width * 1.3  # Approximate ratio

        # Eye spacing: right eye outer corner (45) to left eye outer corner (36)
        if len(landmarks) > 45:
            eye_spacing = np.linalg.norm(landmarks[45] - landmarks[36])
        else:
            eye_spacing = face_width * 0.42

        # Jawline width (approximate from lower face)
        if len(landmarks) > 14:
            jawline_width = np.linalg.norm(landmarks[14] - landmarks[2])
        else:
            jawline_width = face_width * 0.68

        # Nose width (approximate from landmarks)
        if len(landmarks) > 35:
            nose_width = np.linalg.norm(landmarks[35] - landmarks[31])
        else:
            nose_width = face_width * 0.25

        return {
            "face_width_height_ratio": float(face_width / max(face_height, 1)),
            "eye_spacing_face_width": float(eye_spacing / max(face_width, 1)),
            "jawline_width_face_width": float(jawline_width / max(face_width, 1)),
            "nose_width_face_width": float(nose_width / max(face_width, 1))
        }

    except Exception as e:
        logger.warning(f"Error calculating geometry ratios: {e}")
        return {
            "face_width_height_ratio": 0.75,
            "eye_spacing_face_width": 0.42,
            "jawline_width_face_width": 0.68,
            "nose_width_face_width": 0.25
        }


# ===== ADVANCED ENDPOINT =====

@app.route('/analyze-face-advanced', methods=['POST'])
def analyze_face_advanced():
    """
    Extract comprehensive facial attributes for advanced matching

    Request formats:
        1. Multipart form data with 'file' field (JPEG, PNG)
        2. JSON with 'image_base64' field (base64-encoded image)

    Response:
        {
            "face_detected": true,
            "embedding": [512 floats],
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.99,
            "age": 25,
            "gender": "male",
            "landmarks_68": [[x, y], ...],
            "pose": {"yaw": 5.2, "pitch": -2.1, "roll": 0.8},
            "quality": {
                "blur_score": 0.85,
                "illumination": 0.75,
                "overall": 0.80
            },
            "symmetry_score": 0.88,
            "skin_tone": {
                "dominant_color_lab": [65, 10, 20],
                "hex": "#d4a373"
            },
            "expression": {
                "dominant": "smile",
                "confidence": 0.92,
                "emotions": {"happy": 0.85, "neutral": 0.10, ...}
            },
            "geometry": {
                "face_width_height_ratio": 0.75,
                "eye_spacing_face_width": 0.42,
                "jawline_width_face_width": 0.68,
                "nose_width_face_width": 0.25
            }
        }
    """
    # Verify API key
    if not verify_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    try:
        # Handle multipart file upload
        if 'file' in request.files:
            file = request.files['file']
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            logger.info(f"Processing uploaded file (advanced): {file.filename}")

        # Handle base64 image
        elif request.json and 'image_base64' in request.json:
            img_data = base64.b64decode(request.json['image_base64'])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            logger.info("Processing base64 image (advanced)")

        else:
            return jsonify({"error": "No image provided"}), 400

        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Get face detection model
        face_detector = get_face_app()

        # Detect faces
        faces = face_detector.get(img)

        if len(faces) == 0:
            logger.warning("No face detected in image (advanced)")
            return jsonify({
                "face_detected": False,
                "error": "No face detected in image"
            }), 400

        # Use first detected face
        face = faces[0]

        # Extract basic attributes
        embedding = face.embedding.tolist()
        bbox = face.bbox.tolist()
        confidence = float(face.det_score)

        # Extract age and gender from InsightFace
        age = int(face.age) if hasattr(face, 'age') else 25
        gender = "male" if (hasattr(face, 'gender') and face.gender == 1) else "female"

        # Extract landmarks
        landmarks_68 = extract_landmarks_68(face)

        # Extract pose (if available)
        pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        if hasattr(face, 'pose') and face.pose is not None:
            pose_array = face.pose if isinstance(face.pose, (list, np.ndarray)) else [0, 0, 0]
            pose = {
                "yaw": float(pose_array[0]) if len(pose_array) > 0 else 0.0,
                "pitch": float(pose_array[1]) if len(pose_array) > 1 else 0.0,
                "roll": float(pose_array[2]) if len(pose_array) > 2 else 0.0
            }

        # Calculate quality metrics
        blur_score = calculate_blur_score(img, bbox)
        illumination = calculate_illumination(img, bbox)
        overall_quality = (blur_score + illumination) / 2.0

        # Calculate symmetry
        symmetry_score = calculate_symmetry_score(landmarks_68)

        # Extract skin tone
        skin_tone_lab = extract_skin_tone(img, bbox)

        # Convert LAB to hex (approximate)
        try:
            # Create a 1x1 LAB image and convert to BGR
            lab_pixel = np.uint8([[skin_tone_lab]])
            bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
            b, g, r = bgr_pixel[0][0]
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
        except:
            hex_color = "#d4a373"  # Default skin color

        # Detect emotion
        dominant_emotion, emotion_scores = detect_emotion(img, bbox)

        # Calculate geometry ratios
        geometry = calculate_geometry_ratios(landmarks_68)

        logger.info(f"✓ Advanced analysis complete: age={age}, gender={gender}, expression={dominant_emotion}, quality={overall_quality:.2f}")

        return jsonify({
            "face_detected": True,
            "embedding": embedding,
            "bbox": bbox,
            "confidence": confidence,
            "age": age,
            "gender": gender,
            "landmarks_68": landmarks_68,
            "pose": pose,
            "quality": {
                "blur_score": blur_score,
                "illumination": illumination,
                "overall": overall_quality
            },
            "symmetry_score": symmetry_score,
            "skin_tone": {
                "dominant_color_lab": skin_tone_lab,
                "hex": hex_color
            },
            "expression": {
                "dominant": dominant_emotion,
                "confidence": float(emotion_scores.get(dominant_emotion, 0.5)),
                "emotions": emotion_scores
            },
            "geometry": geometry
        })

    except Exception as e:
        logger.error(f"Error in advanced face analysis: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting server on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)
