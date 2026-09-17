---
title: DeepScan Inference API
emoji: 🔍
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# DeepScan Inference API

FastAPI backend for the [DeepScan deepfake detector](https://github.com/ShaunT06/AI-Deepfake-Detector).
Serves `POST /predict` (multipart image upload → verdict + Grad-CAM
heatmap) for the Next.js frontend in `../web`.

This directory is deployed as its own standalone unit — see the main
repo's README for how it fits into the rest of the project, and for the
one-time steps to push this folder to a Hugging Face Space.

## Endpoints

- `GET /health` — liveness check
- `POST /predict` — form field `file`: a JPG/PNG/WEBP image, ≤10MB.
  Returns JSON: `is_fake`, `fake_prob`, `real_prob`, `backbone`,
  `labels_verified`, `img_size`, `gradcam_image_base64` (PNG).

## Local run

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 7860
```

## Config

- `ALLOWED_ORIGINS` — comma-separated list of allowed CORS origins
  (e.g. your Vercel frontend URL). Defaults to `*`.
- `DEEPSCAN_MODEL_FILENAME` — override which checkpoint to load (must
  have an entry in `models/manifest.json`).
