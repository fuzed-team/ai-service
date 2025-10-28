"""
Python AI Microservice - Face Recognition API

Minimal Flask API for face embedding extraction using InsightFace.
This service is called by Next.js API routes for AI-powered face matching.

Endpoints:
    GET  /health - Health check
    POST /extract-embedding - Extract face embedding from image
    POST /compare-faces - Compare two face embeddings
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
