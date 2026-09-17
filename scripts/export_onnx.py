"""Exports the trained ResNet-18 checkpoint to ONNX, with Grad-CAM baked
directly into the graph as plain tensor ops (matmuls, elementwise ReLU) —
no autograd needed at inference time.

Why: the Vercel serverless deployment (see web/api/predict.py) can't
carry full PyTorch/torchvision/OpenCV (deployment size limits), only
onnxruntime + numpy + pillow. onnxruntime is inference-only — it can't
run the backward pass pytorch_grad_cam normally needs. Instead of
tracing an actual autograd.grad() call through the export (fragile —
easy to hit unexportable backward ops), this derives the Grad-CAM
weights analytically for this specific head architecture
(Linear -> ReLU -> Dropout -> Linear, dropout is identity in eval mode)
and computes them as an ordinary forward pass:

  pooled = avgpool(layer4_features)                    # (1, 512)
  h1_pre = pooled @ W1.T + b1;  h1 = relu(h1_pre)       # (1, 256)
  logits = h1 @ W2.T + b2                               # (1, 2)

  For class c, d(logits[c])/d(pooled) is exactly:
      (W2[c, :] * (h1_pre > 0)) @ W1                     # (512,)
  — the standard chain rule through a ReLU MLP, with the ReLU
  derivative as a 0/1 mask. Since avgpool's derivative w.r.t. every
  pixel is a uniform constant (1/(H*W)), Grad-CAM's own
  global-average-pool of the pixel gradients reduces to exactly this
  same per-channel vector (the constant cancels out after the
  ReLU+normalize step Grad-CAM always applies). This is the Grad-CAM
  channel-weight vector for class c, for *both* classes computed
  unconditionally (cheap, and the predicted class isn't known until
  runtime) — so the exported graph outputs logits plus both classes'
  CAMs, and the caller picks the one matching the prediction.

Usage:
    python scripts/export_onnx.py --checkpoint deepfake_model.pth \\
        --out web/api/model/deepfake_model.onnx
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from deepscan.checkpoint import load_checkpoint  # noqa: E402
from deepscan.labels import resolve_class_to_idx  # noqa: E402
from deepscan.model import build_model  # noqa: E402


class GradCAMExportWrapper(nn.Module):
    """Wraps a trained resnet18 DeepScan model. Forward returns
    (logits, cams) where cams has shape (batch, num_classes, 7, 7) —
    the raw (pre-upsample, pre-colormap) Grad-CAM map for every class,
    targeting the model's `layer4` (its final conv block)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        if not hasattr(model, "layer4") or not hasattr(model, "fc"):
            raise ValueError("GradCAMExportWrapper currently only supports the resnet18 architecture")
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            model.layer1, model.layer2, model.layer3, model.layer4,
        )
        self.avgpool = model.avgpool
        # fc = Sequential(Linear(512,256), ReLU, Dropout(0.4), Linear(256,2))
        self.fc1 = model.fc[0]
        self.fc2 = model.fc[3]

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.layer4(x)  # (B, 512, H, W) — H=W=7 for 224x224 input

        pooled = torch.flatten(self.avgpool(feat), 1)  # (B, 512)
        h1_pre = self.fc1(pooled)  # (B, 256)
        h1 = torch.relu(h1_pre)
        logits = self.fc2(h1)  # (B, num_classes)

        mask = (h1_pre > 0).to(h1_pre.dtype)  # (B, 256) — ReLU derivative
        w2 = self.fc2.weight  # (num_classes, 256)
        # (B, num_classes, 256): W2 row per class, masked per-batch-item
        masked_w2 = w2.unsqueeze(0) * mask.unsqueeze(1)
        w1 = self.fc1.weight  # (256, 512)
        weights = torch.matmul(masked_w2, w1)  # (B, num_classes, 512)

        b, c, h, w = feat.shape
        feat_flat = feat.reshape(b, c, h * w)  # (B, 512, H*W)
        cams_flat = torch.bmm(weights, feat_flat)  # (B, num_classes, H*W)
        cams = torch.relu(cams_flat.reshape(b, -1, h, w))  # (B, num_classes, H, W)

        return logits, cams


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="deepfake_model.pth")
    parser.add_argument("--out", default="web/api/model/deepfake_model.onnx")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    if ckpt["backbone"] != "resnet18":
        raise ValueError(f"Only resnet18 is supported by this export script, got {ckpt['backbone']!r}")

    model = build_model("resnet18", pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint)) or "."
    class_to_idx, labels_verified = resolve_class_to_idx(ckpt, checkpoint_dir)

    wrapper = GradCAMExportWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 3, 224, 224)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["logits", "cams"],
        dynamic_axes=None,  # fixed batch size 1, fixed 224x224 — matches the API's preprocessing
        opset_version=args.opset,
    )
    print(f"Exported ONNX model to {args.out}")

    meta = {
        "class_to_idx": class_to_idx,
        "labels_verified": labels_verified,
        "img_size": ckpt.get("img_size", 224),
        "backbone": ckpt["backbone"],
    }
    meta_path = os.path.join(os.path.dirname(args.out), "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {meta_path}: {meta}")


if __name__ == "__main__":
    main()
