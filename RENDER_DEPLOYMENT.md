# Deploying Python AI Service to Render

Render is an excellent platform for deploying the Python AI microservice with generous free tier and easy setup.

## Why Render?

✅ **Free Tier:** 750 hours/month free (enough for development)
✅ **Easy Deployment:** Git-based automatic deployments
✅ **Built-in SSL:** Free HTTPS certificates
✅ **Auto-scaling:** Scales automatically with traffic
✅ **Good Performance:** Fast cold starts (~2-3 seconds)
✅ **No Credit Card Required:** For free tier

**Cost:**
- **Free Tier:** $0/month (750 hours, then $0.01/hour)
- **Starter Plan:** $7/month (always on)
- **Standard Plan:** $25/month (more CPU/RAM)

## Deployment Options

### Option 1: One-Click Deploy with Blueprint (Easiest)

1. **Push code to GitHub:**
   ```bash
   cd python-ai-service
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/your-username/ai-face-service.git
   git push -u origin main
   ```

2. **Deploy on Render:**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click **"New"** → **"Blueprint"**
   - Connect your GitHub repository
   - Render will detect `render.yaml` and auto-configure
   - Click **"Apply"**

3. **Done!** 🎉
   - Render will build and deploy automatically
   - You'll get a URL like: `https://ai-face-service.onrender.com`
   - API key is auto-generated (find in Environment tab)

### Option 2: Manual Setup (More Control)

1. **Create New Web Service:**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click **"New"** → **"Web Service"**
   - Connect your GitHub repository (or use "Deploy an existing image from a registry" for Docker)

2. **Configure Service:**
   ```
   Name: ai-face-service
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 app:app
   ```

3. **Set Environment Variables:**
   - Go to **Environment** tab
   - Add variables:
     ```
     API_KEY = <click "Generate" for secure random key>
     PYTHON_VERSION = 3.11.0
     PYTHONUNBUFFERED = 1
     ```

4. **Configure Health Check:**
   - Go to **Settings** → **Health Check**
   - Path: `/health`
   - This prevents service from sleeping

5. **Deploy:**
   - Click **"Create Web Service"**
   - Wait 5-10 minutes for initial build
   - Service URL: `https://your-service-name.onrender.com`

### Option 3: Docker Deployment

Render also supports Docker if you prefer:

1. **Use existing Dockerfile:**
   - Already created in the repo

2. **Deploy on Render:**
   - Choose **"Deploy an existing image from a registry"**
   - Or connect GitHub and Render will auto-detect Dockerfile

3. **Configure:**
   - Build & deploy happens automatically
   - Render uses the Dockerfile build instructions

## Getting Your Service URL and API Key

After deployment:

1. **Service URL:**
   - Copy from Render dashboard
   - Format: `https://ai-face-service.onrender.com`

2. **API Key:**
   - Go to **Environment** tab
   - Click on `API_KEY` value to reveal
   - Copy the generated key

## Configure Next.js

Add to `web/.env.local`:

```env
# Render Python AI Service
PYTHON_AI_SERVICE_URL=https://ai-face-service.onrender.com
PYTHON_AI_SERVICE_API_KEY=your-api-key-from-render
```

## Testing Deployment

### 1. Health Check

```bash
curl https://ai-face-service.onrender.com/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model": "insightface",
  "version": "1.0.0"
}
```

### 2. Test Embedding Extraction

```bash
# Get API key from Render dashboard first
API_KEY="your-api-key"

curl -X POST https://ai-face-service.onrender.com/extract-embedding \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@test_face.jpg"
```

**Expected response:**
```json
{
  "face_detected": true,
  "embedding": [0.123, -0.456, ...],
  "bbox": [100, 150, 200, 250],
  "confidence": 0.99
}
```

## Free Tier Behavior

### Important Notes:

1. **Sleep After 15 Minutes:**
   - Free tier services sleep after 15 min of inactivity
   - First request after sleep takes ~30-60 seconds (cold start)
   - Subsequent requests are fast (<1 second)

2. **Prevent Sleeping (Optional):**
   - Upgrade to Starter plan ($7/month) for always-on
   - Or use a cron job to ping every 14 minutes:
     ```bash
     # Add to your system cron or use a service like cron-job.org
     */14 * * * * curl https://ai-face-service.onrender.com/health
     ```

3. **Monthly Limit:**
   - 750 hours/month free
   - Resets on 1st of each month
   - After limit: $0.01/hour

## Logs and Monitoring

### View Logs:
1. Go to Render Dashboard
2. Click your service
3. Go to **Logs** tab
4. Real-time logs displayed

### Monitor Health:
- **Health Check:** Render pings `/health` every 30 seconds
- **Uptime:** View in dashboard
- **Alerts:** Configure email alerts for failures

## Performance Optimization

### 1. Pre-download Model (Faster Cold Starts)

Edit Dockerfile and uncomment this line:

```dockerfile
# Pre-download InsightFace model
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(providers=['CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640, 640))"
```

**Trade-off:**
- ✅ Faster cold starts (30s → 5s)
- ❌ Slower build times (5 min → 10 min)
- ❌ Larger Docker image (+500MB)

### 2. Increase Workers (If Upgraded)

For Starter plan or higher, increase workers:

```yaml
startCommand: gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120 app:app
```

### 3. Enable Keep-Alive

Add to `render.yaml`:

```yaml
healthCheckPath: /health
healthCheckInterval: 60  # Ping every 60 seconds
```

## Automatic Deployments

Render automatically deploys when you push to GitHub:

```bash
# Make changes to code
vim app.py

# Commit and push
git add .
git commit -m "Update face detection logic"
git push origin main

# Render auto-deploys within 2-5 minutes
```

**Monitor deployment:**
- Go to **Events** tab in Render dashboard
- See build logs in real-time

## Environment Management

### Development vs Production

Create separate services for dev and prod:

1. **Dev Service:**
   - Deploy from `dev` branch
   - Use `DEBUG=True`
   - Free tier is fine

2. **Production Service:**
   - Deploy from `main` branch
   - Use `DEBUG=False`
   - Consider Starter plan for always-on

### Managing Secrets

**Generate new API key:**
```bash
# In Render dashboard Environment tab
# Click "Generate" next to API_KEY

# Or generate locally:
openssl rand -base64 32
```

**Update in both places:**
1. Render: Environment tab → API_KEY
2. Next.js: `.env.local` → PYTHON_AI_SERVICE_API_KEY

## Troubleshooting

### Build Fails

**Error: "No module named cv2"**
- **Fix:** Already handled in requirements.txt with `opencv-python-headless`

**Error: "Memory limit exceeded"**
- **Fix:** Upgrade to Starter plan (more RAM)

### Service Not Responding

1. **Check Logs:**
   - Look for errors in Render logs
   - Common: Model loading timeout

2. **Check Health:**
   ```bash
   curl https://your-service.onrender.com/health
   ```

3. **Restart Service:**
   - Go to Render dashboard
   - Click **"Manual Deploy"** → **"Clear build cache & deploy"**

### Slow Cold Starts

**Solutions:**
1. Pre-download model in Dockerfile (see above)
2. Upgrade to Starter plan (always-on)
3. Use health check pinging to keep warm

### "Unauthorized" Errors

1. **Check API Key:**
   - Render dashboard → Environment → API_KEY
   - Copy exact value

2. **Update Next.js:**
   ```bash
   # web/.env.local
   PYTHON_AI_SERVICE_API_KEY=paste-exact-key-here
   ```

3. **Restart Next.js:**
   ```bash
   cd web
   npm run dev  # or restart Vercel deployment
   ```

## Cost Comparison

| Platform | Free Tier | Paid Plan | Cold Start | Setup |
|----------|-----------|-----------|------------|-------|
| **Render** | 750h/mo | $7/mo | ~30s | Easy |
| Railway | $5 credit | $5-10/mo | ~20s | Easy |
| Fly.io | 3GB RAM | $0-5/mo | ~15s | Medium |
| Cloud Run | Pay-per-use | $0-10/mo | ~10s | Hard |

**Recommendation:** Start with Render free tier, upgrade to Starter ($7/mo) if you need always-on.

## Render-Specific Features

### 1. Zero-Downtime Deploys
- Render keeps old version running until new version is healthy
- No service interruption during updates

### 2. Automatic SSL
- Free HTTPS certificates
- Auto-renewal
- Works out of the box

### 3. Custom Domains
- Add your own domain (e.g., `api.yourdomain.com`)
- Free SSL for custom domains

### 4. Preview Environments
- Render can create preview deploys for pull requests
- Test changes before merging

## Quick Start Summary

1. **Deploy to Render** (5 minutes):
   ```bash
   # Push code to GitHub
   git push origin main

   # Go to render.com → New Blueprint
   # Connect GitHub repo
   # Click "Apply"
   ```

2. **Get credentials** (1 minute):
   - Copy service URL: `https://ai-face-service.onrender.com`
   - Copy API key from Environment tab

3. **Configure Next.js** (1 minute):
   ```bash
   # web/.env.local
   PYTHON_AI_SERVICE_URL=https://ai-face-service.onrender.com
   PYTHON_AI_SERVICE_API_KEY=your-api-key
   ```

4. **Test** (1 minute):
   ```bash
   curl https://ai-face-service.onrender.com/health
   ```

5. **Done!** 🎉

## Next Steps

After deploying to Render:

1. ✅ Service is live at `https://your-service.onrender.com`
2. ✅ API key configured in Next.js
3. ✅ Health check passing
4. → Proceed to **Phase 4** - Implement Next.js API routes
5. → Test face upload end-to-end

## Resources

- [Render Documentation](https://render.com/docs)
- [Render Python Guide](https://render.com/docs/deploy-flask)
- [Render Dashboard](https://dashboard.render.com/)
- [Render Status Page](https://status.render.com/)

---

**Updated:** 2025-10-27
**Platform:** Render.com
**Cost:** Free tier available, $7/mo for always-on
