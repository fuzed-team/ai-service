# FastAPI AI Microservice - Face Recognition API

**FastAPI-based** AI service for face embedding extraction using InsightFace and DeepFace. This service provides comprehensive face detection, analysis, and matching capabilities.

> **⚠️ Migration Notice**: This service has been migrated from Flask to FastAPI. See [MIGRATION.md](MIGRATION.md) for details.

## Features

- ✅ **Face Detection Verification** - Quick check if face exists in image
- ✅ **Face Embedding Extraction** - 512D embeddings from InsightFace
- ✅ **Face Comparison** - Cosine similarity between embeddings
- ✅ **Advanced Face Analysis** - Age, gender, emotions, quality metrics, geometry
- ✅ **Batch Processing** - Process multiple images at once
- ✅ **API Key Authentication** - Secure Bearer token authentication
- ✅ **Docker Support** - Multi-stage optimized builds
- ✅ **Production Ready** - Uvicorn ASGI server with process pool
- ✅ **Auto-Generated Docs** - Interactive API docs at `/docs`
- ✅ **Type Safety** - Full Pydantic validation
- ✅ **Async Architecture** - Non-blocking ML inference

## Quick Start

### Interactive API Documentation
Visit http://localhost:8000/docs for interactive Swagger UI to test all endpoints.

## API Endpoints

> **Base URL**: All endpoints are prefixed with `/api/v1`

### `GET /api/v1/health`
Health check endpoint (no authentication required)

**Response:**
```json
{
  "status": "healthy",
  "model": "insightface",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### `POST /api/v1/verify-face`
Verify if a face is detected in an image (lightweight, no embedding extraction)

**Request (multipart/form-data):**
```bash
curl -X POST http://localhost:8000/api/v1/verify-face \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@face.jpg"
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

### `POST /api/v1/extract-embedding`
Extract face embedding from image

**Request (multipart or form data):**
```bash
curl -X POST http://localhost:8000/api/v1/extract-embedding \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@face.jpg"

# Or with base64
curl -X POST http://localhost:8000/api/v1/extract-embedding \
  -H "Authorization: Bearer your-api-key" \
  -F "image_base64=data:image/jpeg;base64,/9j/4AAQ..."
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

### `POST /api/v1/compare-faces`
Compare two face embeddings

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/compare-faces \
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

### `POST /api/v1/batch-extract`
Extract embeddings from multiple images (up to 50)

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

### `POST /api/v1/analyze-face-advanced`
**NEW**: Extract comprehensive facial attributes including age, gender, emotions, quality metrics, symmetry, skin tone, and geometry

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/analyze-face-advanced \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@face.jpg"
```

**Response includes**: embedding, age, gender, emotions, pose, quality scores, symmetry, skin tone, facial geometry ratios

## Local Development

### Prerequisites

- Python 3.11
- UV package manager ([installation guide](https://github.com/astral-sh/uv))
- Docker (recommended for Windows users due to TensorFlow dependencies)

### Setup with Docker (Recommended)

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env and set your API_KEY
nano .env

# 3. Build and run with Docker
docker build -t ai-face-service .
docker run -p 8000:8000 -e API_KEY=your-api-key ai-face-service
```

### Setup with UV (Linux/macOS)

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env and set your API_KEY
nano .env

# 3. Install dependencies with UV
uv sync

# 4. Run development server
uv run uvicorn app.main:app --reload --port 8000
```

Server will start at `http://localhost:8000`

### Test Endpoints

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Interactive API docs (open in browser)
open http://localhost:8000/docs

# Extract embedding
curl -X POST http://localhost:8000/api/v1/extract-embedding \
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
docker run -p 8000:8000 \
  -e API_KEY="your-secure-api-key" \
  -e CORS_ORIGINS="http://localhost:3000" \
  ai-face-service
```

### Test

```bash
# Health check
curl http://localhost:8000/api/v1/health

# API docs
open http://localhost:8000/docs
```

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `API_KEY` | API key for authentication | Yes | - |
| `PORT` | Server port | No | `8000` |
| `DEBUG` | Enable debug mode | No | `False` |
| `CORS_ORIGINS` | Allowed CORS origins (comma-separated) | No | `http://localhost:3000` |
| `MAX_WORKERS` | Process pool workers for ML | No | `2` |
| `FACE_DET_THRESH` | Face detection threshold | No | `0.5` |

See `.env.example` for complete configuration.

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
curl https://your-service.railway.app/api/v1/health
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
ai-service/
├── app/
│   ├── main.py                      # FastAPI app initialization
│   ├── core/                        # Core utilities
│   │   ├── config.py                # Settings & configuration
│   │   ├── security.py              # Authentication
│   │   └── exceptions.py            # Exception handlers
│   ├── api/v1/                      # API version 1
│   │   ├── endpoints/               # Route handlers
│   │   └── router.py                # Router setup
│   ├── models/                      # Pydantic models
│   ├── services/                    # Business logic
│   └── workers/                     # ML worker pool
├── pyproject.toml                   # UV project config & dependencies
├── Dockerfile                       # Multi-stage build with UV
├── MIGRATION.md                     # Migration guide
└── README.md                        # This file
```

### Adding New Endpoints

1. Create Pydantic models in `app/models/`
2. Add business logic in `app/services/`
3. Create endpoint in `app/api/v1/endpoints/`
4. Register router in `app/api/v1/router.py`
5. Test at http://localhost:8000/docs

See existing endpoints for examples.

## Next Steps

1. Deploy service to Railway/Fly.io
2. Get deployment URL
3. Add URL and API key to Next.js `.env.local`
4. Implement Next.js API routes that call this service
5. Test face upload flow end-to-end

## Architecture Highlights

### FastAPI Best Practices
- ✅ Domain-based modular structure
- ✅ Pydantic BaseSettings for configuration
- ✅ Dependency injection for auth & services
- ✅ Process pool for CPU-intensive ML tasks
- ✅ Comprehensive error handling
- ✅ Type hints throughout

### Performance
- **Async Architecture**: Non-blocking API operations
- **Process Pool**: ML inference doesn't block event loop
- **Multi-stage Docker**: Optimized build layers
- **UV Package Manager**: 10-100x faster than pip

## Migration from Flask

This service was migrated from Flask to FastAPI on 2025-11-09. Key changes:
- **Port**: 5000 → 8000
- **Base URL**: `/` → `/api/v1/`
- **Server**: Gunicorn → Uvicorn
- **Architecture**: Monolithic → Modular

See [MIGRATION.md](MIGRATION.md) for complete migration details.

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [InsightFace Documentation](https://github.com/deepinsight/insightface)
- [DeepFace Documentation](https://github.com/serengil/deepface)
- [UV Package Manager](https://github.com/astral-sh/uv)
- [Railway Documentation](https://docs.railway.app/)

---

**Created:** 2025-10-27
**Migrated to FastAPI:** 2025-11-09
**Status:** ✅ Production Ready
