"""Vercel Python serverless function: POST /api/predict.

Runs the deepfake classifier + Grad-CAM entirely within Vercel's own
Python runtime, using onnxruntime instead of PyTorch/OpenCV — those
don't fit Vercel's function size limit. See ../../scripts/export_onnx.py
for how the Grad-CAM math got baked into the ONNX graph itself (no
autograd needed here), and model/meta.json for the Real/Fake class
mapping this model was exported with.

Also enforces its own, tighter upload limit: Vercel's own request body
cap is ~4.5MB, well under the 10MB the original app advertised.
"""

import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MAX_UPLOAD_BYTES = 3.5 * 1024 * 1024  # stay safely under Vercel's ~4.5MB request body cap
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _load_meta():
    with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


# Module-level: loaded once per cold start, reused across warm invocations.
_META = _load_meta()
_SESSION = ort.InferenceSession(
    os.path.join(MODEL_DIR, "deepfake_model.onnx"),
    providers=["CPUExecutionProvider"],
)
_JET_LUT = np.load(os.path.join(MODEL_DIR, "jet_lut.npy"))  # (256, 3) uint8, RGB
_IMG_SIZE = _META.get("img_size", 224)


def _preprocess(image: Image.Image):
    """Returns (model_input (1,3,H,W) float32, original_rgb_float (H,W,3) in [0,1])."""
    resized = image.convert("RGB").resize((_IMG_SIZE, _IMG_SIZE), Image.BILINEAR)
    rgb_float = np.asarray(resized, dtype=np.float32) / 255.0  # (H, W, 3)
    normalized = (rgb_float - IMAGENET_MEAN) / IMAGENET_STD
    model_input = np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)  # (1,3,H,W)
    return model_input, rgb_float


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _cam_to_overlay(raw_cam: np.ndarray, original_rgb_float: np.ndarray) -> Image.Image:
    """raw_cam: (h, w) pre-normalize Grad-CAM map (small, e.g. 7x7) for the
    predicted class. Reproduces pytorch_grad_cam's show_cam_on_image()
    formula (min-max normalize -> bilinear upsample -> JET colormap ->
    50/50 blend with the original image -> renormalize) without OpenCV."""
    cam = raw_cam - raw_cam.min()
    cam = cam / (cam.max() + 1e-7)

    cam_img = Image.fromarray(cam.astype(np.float32), mode="F").resize(
        (_IMG_SIZE, _IMG_SIZE), Image.BILINEAR
    )
    cam_resized = np.clip(np.asarray(cam_img), 0.0, 1.0)

    heatmap_idx = np.clip((cam_resized * 255).astype(np.int32), 0, 255)
    heatmap_rgb = _JET_LUT[heatmap_idx].astype(np.float32) / 255.0  # (H, W, 3)

    overlay = 0.5 * heatmap_rgb + 0.5 * original_rgb_float
    overlay = overlay / np.max(overlay)
    overlay_uint8 = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay_uint8, mode="RGB")


def run_predict(raw_bytes: bytes):
    """Returns (status_code, response_dict)."""
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image = ImageOps.exif_transpose(image)
    except UnidentifiedImageError:
        return 400, {"detail": "Couldn't read that file as an image. Please upload a JPG, PNG, or WEBP."}

    model_input, original_rgb_float = _preprocess(image)
    logits, cams = _SESSION.run(["logits", "cams"], {"input": model_input})
    probs = _softmax(logits[0])

    class_to_idx = _META["class_to_idx"]
    fake_idx, real_idx = class_to_idx["Fake"], class_to_idx["Real"]
    fake_prob, real_prob = float(probs[fake_idx]), float(probs[real_idx])
    is_fake = fake_prob > 0.5
    predicted_idx = fake_idx if is_fake else real_idx

    raw_cam = cams[0, predicted_idx]
    gradcam_image = _cam_to_overlay(raw_cam, original_rgb_float)
    buf = io.BytesIO()
    gradcam_image.save(buf, format="PNG")
    gradcam_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return 200, {
        "is_fake": is_fake,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "backbone": _META["backbone"],
        "labels_verified": _META["labels_verified"],
        "img_size": _IMG_SIZE,
        "gradcam_image_base64": gradcam_b64,
    }


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send_json(200, {"status": "ok"})

    def do_POST(self):
        content_type = self.headers.get("Content-Type", "")
        if content_type not in ALLOWED_CONTENT_TYPES:
            self._send_json(400, {
                "detail": "Upload a JPG, PNG, or WEBP image (raw bytes, matching Content-Type).",
            })
            return

        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_UPLOAD_BYTES:
            limit_mb = MAX_UPLOAD_BYTES / 1024 / 1024
            self._send_json(413, {"detail": f"File exceeds the {limit_mb:.1f}MB limit."})
            return

        raw = self.rfile.read(length)

        try:
            status, payload = run_predict(raw)
        except Exception as exc:  # noqa: BLE001 - surface a clean 500 instead of a raw traceback
            self._send_json(500, {"detail": f"Inference failed: {exc}"})
            return

        self._send_json(status, payload)
