# DeepScan — AI Deepfake Detector

A deepfake image classifier with transfer learning and Grad-CAM
explainability. Upload a face image and get a real/fake verdict plus a
heatmap of the regions that drove it.

Three ways to run this project:
- **Next.js on Vercel, all-in-one** (`web/`) — the recommended way to
  deploy this publicly. The model is exported to ONNX with Grad-CAM
  baked into the graph as plain tensor ops (see
  [`scripts/export_onnx.py`](scripts/export_onnx.py) — PyTorch's
  autograd isn't available at inference time, so the Grad-CAM weights
  are derived analytically instead), which lets a tiny `onnxruntime` +
  `numpy` + `pillow` stack run entirely inside a Vercel Python
  serverless function (`web/api/predict.py`). No separate backend host,
  no CORS, one deploy. See [`web/README.md`](web/README.md).
- **Streamlit app** (`app/streamlit_app.py`) — single process, full
  PyTorch/OpenCV, model and UI together. Simplest to run locally.
- **Next.js + separate FastAPI backend** (`web/` + `backend/`) — the
  frontend calls a separately-hosted full-PyTorch inference API instead
  of the bundled ONNX one. Only worth it if you need the full PyTorch
  pipeline in production (e.g. testing a not-yet-exported architecture).
  See [`backend/README.md`](backend/README.md).

> **⚠ Before you trust a verdict from this app:** the currently-deployed
> checkpoint (`resnet18-v1`) is a legacy model whose Real/Fake label order
> was never recorded at training time and could not be recovered after the
> fact (see [`deepscan/labels.py`](src/deepscan/labels.py)). The app shows
> a visible warning banner whenever it's running on an unverified mapping.
> Spot-check it yourself with a couple of known real/fake images before
> relying on it, or retrain with the current pipeline (below), which saves
> this mapping correctly going forward.

## Project layout

```
src/deepscan/
  model.py       # build_model() — single source of truth for the architecture
  data.py        # train/eval transforms and dataloaders
  checkpoint.py  # save/load format that carries backbone + class_to_idx + metrics
  labels.py      # resolves Real/Fake index mapping (embedded > sidecar > fallback)
  inference.py   # Predictor — one forward pass, prediction + Grad-CAM
  train.py       # python -m deepscan.train
  evaluate.py    # python -m deepscan.evaluate
app/
  streamlit_app.py   # Streamlit UI (model + UI in one process)
scripts/
  download_model.py  # fetches the checkpoint from GitHub Releases on first run
  export_onnx.py     # exports deepfake_model.pth -> web/api/model/*.onnx, with Grad-CAM baked in
tests/           # pytest — runs against a tiny untrained fixture, no dataset needed

web/                     # Next.js frontend + bundled ONNX inference — see web/README.md
  api/predict.py         # Vercel Python function: onnxruntime + numpy + pillow only
  api/model/              # exported ONNX model, Grad-CAM colormap LUT, class-mapping metadata
                           # (committed to git — this is a required build artifact, not sample data;
                           # regenerate with scripts/export_onnx.py if the model changes)
backend/                 # optional: separate full-PyTorch FastAPI API — see backend/README.md
                         # (a self-contained copy of the inference-only deepscan modules;
                         # see backend/deepscan/__init__.py for why it's not a shared import)
```

## Setup

Requires Python 3.12+ (pinned `numpy==2.5.3` has no wheels for older versions).

```bash
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The model checkpoint (~45MB) isn't stored in git — it's a
[GitHub Release asset](https://github.com/ShaunT06/AI-Deepfake-Detector/releases/tag/resnet18-v1)
that `scripts/download_model.py` downloads and checksum-verifies
automatically the first time the app runs.

## Training your own model

The original dataset (`Dataset/Train`, `Dataset/Validation`, one
subfolder per class) isn't included in this repo. Point the trainer at
your own:

```bash
pip install -r requirements-dev.txt
python -m deepscan.train --data-dir Dataset --backbone efficientnet_b0 \
    --epochs 8 --fine-tune-epochs 6 --out deepfake_model_v2.pth
```

- `--backbone` supports `resnet18` (the original architecture) and
  `efficientnet_b0` (recommended default — stronger ImageNet features at
  a similar parameter count).
- Training runs two phases: a fast head-only warmup, then fine-tuning
  with the last backbone block unfrozen (`--fine-tune-epochs 0` to skip
  and train only the head).
- The resulting checkpoint embeds its `class_to_idx` mapping, backbone
  name, and validation accuracy — no more guessing at inference time.

Then evaluate on a held-out test set before trusting it:

```bash
python -m deepscan.evaluate --checkpoint deepfake_model_v2.pth --data-dir Dataset/Test
```

This prints accuracy, precision/recall/F1 on the Fake class, ROC-AUC,
and a confusion matrix — none of which the original project computed.

To point the app at your new checkpoint, drop it next to `app/` as
`deepfake_model.pth`, or set `DEEPSCAN_MODEL_FILENAME`.

## Testing

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

Tests build tiny, untrained model instances on the fly, so they run in
seconds with no dataset, GPU, or network access — they check that the
code is correct, not that the model is accurate (that's what `evaluate.py`
is for).

## Known limitations

- **Label order** — see the warning at the top of this file.
- **No face detection/cropping.** The model classifies the full image
  resized to 224×224; a face-detection preprocessing step (e.g. MTCNN)
  would likely improve real-world accuracy.
- **Never evaluated on out-of-distribution generators.** The original
  model was validated only on data from the same source as its training
  set, so its accuracy on deepfakes from generators it never saw is
  unknown. Retrain/evaluate on datasets like FaceForensics++ or Celeb-DF
  before treating this as production-ready.
- **Static confidence threshold.** Verdicts are a hard 50% cutoff with no
  calibration or "uncertain" band.

## Deploying to Vercel

`web/` deploys to Vercel on its own — frontend and inference API
together, no other account or service needed:

```bash
cd web
vercel link      # first time only
vercel --prod
```

Or connect the GitHub repo in the Vercel dashboard and set **Root
Directory** to `web`. Vercel auto-detects `api/predict.py` as a Python
serverless function (installing `web/requirements.txt`) alongside the
Next.js frontend.

Notes:
- The ONNX model (`web/api/model/`) is committed to git specifically so
  a fresh Vercel deploy always has it — see the note in
  [Project layout](#project-layout).
- Uploads are capped at 3.5MB (`web/api/predict.py`'s `MAX_UPLOAD_BYTES`),
  below Vercel's own ~4.5MB request body limit.
- If you retrain the model, regenerate the ONNX export before deploying:
  `python scripts/export_onnx.py --checkpoint deepfake_model_v2.pth`.

### Alternative: separate FastAPI backend

If you'd rather run the full PyTorch pipeline (e.g. `efficientnet_b0`,
which `export_onnx.py` doesn't support yet) behind Vercel instead of the
bundled ONNX function, host `backend/` on a container platform (a
[Hugging Face Space](https://huggingface.co/new-space) with Docker SDK,
Render, Fly.io, ...) and point the frontend at it:

```bash
# backend/ — push to your chosen host, then:
cd web
vercel env add NEXT_PUBLIC_API_URL production   # your backend's URL
```
This requires switching `web/lib/api.ts` back to calling
`NEXT_PUBLIC_API_URL` instead of the bundled `/api/predict` route (see
git history around when `web/api/predict.py` was added) — the two
inference paths aren't wired up simultaneously today.

## Architecture

ResNet-18 (or EfficientNet-B0) → custom classifier head (Linear → ReLU →
Dropout → Linear, 2 classes) → Grad-CAM on the final convolutional block
for visual explanation.
