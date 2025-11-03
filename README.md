# Python AI Microservice - Face Recognition API

Minimal Flask API for face embedding extraction using InsightFace. This service is called by Next.js API routes for AI-powered face matching.

## Features

- ✅ **Face Detection Verification** - Quick check if face exists in image
- ✅ **Face Embedding Extraction** - 512D embeddings from InsightFace
- ✅ **Face Comparison** - Cosine similarity between embeddings
- ✅ **Batch Processing** - Process multiple images at once
- ✅ **API Key Authentication** - Secure endpoint access
- ✅ **Docker Support** - Ready for containerized deployment
- ✅ **Production Ready** - Gunicorn server with health checks

## API Endpoints

### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model": "insightface",
  "version": "1.0.0"
}
```

### `POST /verify-face`
Verify if a face is detected in an image (lightweight, no embedding extraction)

**Request (multipart/form-data):**
```bash
curl -X POST http://localhost:5000/verify-face \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@face.jpg"
```

**Request (JSON with base64):**
```bash
curl -X POST http://localhost:5000/verify-face \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "base64_encoded_image_data"
  }'
```

**Success Response:**
```json
{
  "face_detected": true,
  "confidence": 0.99,
  "bbox": [100, 150, 200, 250],
  "message": "Face detected successfully"
}
```

**Error Response:**
```json
{
  "face_detected": false,
  "error": "No face detected in image"
}
```

### `POST /extract-embedding`
Extract face embedding from image

**Request (multipart/form-data):**
```bash
curl -X POST http://localhost:5000/extract-embedding \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@face.jpg"
```

**Request (JSON with base64):**
```bash
curl -X POST http://localhost:5000/extract-embedding \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "base64_encoded_image_data"
  }'
```

**Response:**
```json
{
  "face_detected": true,
  "embedding": [0.123, -0.456, ...],  // 512 values
  "bbox": [100, 150, 200, 250],
  "confidence": 0.99
}
```

### `POST /compare-faces`
Compare two face embeddings

**Request:**
```bash
curl -X POST http://localhost:5000/compare-faces \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "embedding_a": [0.123, ...],
    "embedding_b": [0.456, ...]
  }'
```

**Response:**
```json
{
  "similarity": 0.85,
  "distance": 0.15
}
```

### `POST /batch-extract`
Extract embeddings from multiple images

**Request:**
```json
{
  "images": [
    "base64_image_1",
    "base64_image_2"
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "index": 0,
      "face_detected": true,
      "embedding": [...],
      "confidence": 0.99
    }
  ],
  "total": 2,
  "successful": 2,
  "failed": 0
}
```

## Local Development

### Prerequisites

- Python 3.11+
- pip

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export API_KEY="your-secure-api-key"
export PORT=8000
export DEBUG=True

# Run server
python app.py
```

Server will start at `http://localhost:5000`

### Test Endpoints

```bash
# Health check
curl http://localhost:5000/health

# Extract embedding (replace with your API key)
curl -X POST http://localhost:5000/extract-embedding \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@test_face.jpg"
```

## Docker Deployment

### Build Image

```bash
docker build -t ai-face-service .
```

### Run Container

```bash
docker run -p 5000:5000 -e API_KEY="your-secure-api-key" ai-face-service
```

### Test

```bash
curl http://localhost:5000/health
```

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `API_KEY` | API key for authentication | Yes | `change-me-in-production` |
| `PORT` | Server port | No | `8000` |
| `DEBUG` | Enable debug mode | No | `False` |

## Security

### API Key Generation

Generate secure API key:
```bash
openssl rand -base64 32
```

### Next.js Integration

Add to Next.js `.env.local`:
```env
PYTHON_AI_SERVICE_URL=https://your-service.railway.app
PYTHON_AI_SERVICE_API_KEY=your-api-key
```

## Performance

### Cold Start

- **First request:** ~3-5 seconds (model loading)
- **Subsequent requests:** <1 second

### Optimization Tips

1. **Pre-download model in Dockerfile** (faster cold starts):
   ```dockerfile
   RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(providers=['CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640, 640))"
   ```

2. **Use GPU** (much faster):
   - Change `providers=['CPUExecutionProvider']` to `['CUDAExecutionProvider', 'CPUExecutionProvider']`
   - Add CUDA dependencies to Dockerfile
   - Use GPU-enabled hosting (more expensive)

3. **Increase workers** (only if you have multiple CPU cores):
   ```bash
   gunicorn --workers=2 --threads=2 app:app
   ```

## Monitoring

### Health Checks

```bash
curl https://your-service.railway.app/health
```

### Logs

**Railway:**
```bash
railway logs
```

**Fly.io:**
```bash
fly logs
```

**Docker:**
```bash
docker logs <container-id>
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'cv2'"
- Install system dependencies: `apt-get install libgl1-mesa-glx`

### "No face detected in image"
- Image quality too low
- Face too small or angled
- Try higher resolution image

### "Timeout errors"
- Increase gunicorn timeout: `--timeout 120`
- Cold start takes longer on first request

### High memory usage
- InsightFace model uses ~500MB RAM
- Reduce workers if memory limited
- Consider serverless deployment (Cloud Run)

## Development

### Project Structure

```
python-ai-service/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image
├── .env.example        # Environment template
├── .dockerignore       # Docker ignore rules
└── README.md           # This file
```

### Adding New Endpoints

1. Add route in `app.py`:
   ```python
   @app.route('/new-endpoint', methods=['POST'])
   def new_endpoint():
       if not verify_api_key():
           return jsonify({"error": "Unauthorized"}), 401
       # Your logic here
       return jsonify({"result": "success"})
   ```

2. Update README with endpoint documentation

3. Test locally before deploying

## Next Steps

1. Deploy service to Railway/Fly.io
2. Get deployment URL
3. Add URL and API key to Next.js `.env.local`
4. Implement Next.js API routes that call this service
5. Test face upload flow end-to-end

## Resources

- [InsightFace Documentation](https://github.com/deepinsight/insightface)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Railway Documentation](https://docs.railway.app/)
- [Fly.io Documentation](https://fly.io/docs/)

---

**Created:** 2025-10-27
**Status:** ✅ Ready for deployment
