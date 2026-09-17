"""FastAPI inference API for DeepScan.

Wraps deepscan.inference.Predictor behind two endpoints:
  GET  /health            liveness/readiness probe
  POST /predict           multipart image upload -> verdict + Grad-CAM

Deployed independently from the Next.js frontend (see ../web) — the
frontend calls this over HTTP. CORS is restricted to the configured
frontend origin(s) via the ALLOWED_ORIGINS env var.
"""

from __future__ import annotations

import base64
import io
import os

from download_model import ensure_model
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from deepscan.inference import Predictor

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB, matches the frontend's advertised limit
MODEL_FILENAME = os.environ.get("DEEPSCAN_MODEL_FILENAME", "deepfake_model.pth")
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="DeepScan Inference API", version="2.0.0")

_allowed_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        model_path = ensure_model(MODEL_FILENAME, dest_dir=BACKEND_DIR)
        _predictor = Predictor(model_path)
    return _predictor


@app.on_event("startup")
def _warm_up() -> None:
    # Download + load the model at startup rather than on the first
    # request, so the first real user isn't the one paying for it.
    get_predictor()


class PredictResponse(BaseModel):
    is_fake: bool
    fake_prob: float
    real_prob: float
    backbone: str
    labels_verified: bool
    img_size: int
    gradcam_image_base64: str  # PNG, base64-encoded


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Upload a JPG, PNG, or WEBP image.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 10MB limit.")

    try:
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Couldn't read that file as an image.") from exc

    predictor = get_predictor()
    pred = predictor.predict(image)

    buf = io.BytesIO()
    pred.gradcam_image.save(buf, format="PNG")
    gradcam_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return PredictResponse(
        is_fake=pred.is_fake,
        fake_prob=pred.fake_prob,
        real_prob=pred.real_prob,
        backbone=pred.backbone,
        labels_verified=pred.labels_verified,
        img_size=predictor.img_size,
        gradcam_image_base64=gradcam_b64,
    )
