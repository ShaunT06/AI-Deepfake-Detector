# DeepScan — AI Deepfake Detector

A deepfake image classifier with transfer learning and Grad-CAM
explainability, deployed as a Streamlit app. Upload a face image and get a
real/fake verdict plus a heatmap of the regions that drove it.

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
  streamlit_app.py   # the web UI
scripts/
  download_model.py  # fetches the checkpoint from GitHub Releases on first run
tests/           # pytest — runs against a tiny untrained fixture, no dataset needed
```

## Setup

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

## Architecture

ResNet-18 (or EfficientNet-B0) → custom classifier head (Linear → ReLU →
Dropout → Linear, 2 classes) → Grad-CAM on the final convolutional block
for visual explanation.
