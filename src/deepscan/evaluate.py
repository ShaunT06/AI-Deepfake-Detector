"""Evaluate a checkpoint on a held-out test set and report real metrics.

The original project had no evaluation script at all — no accuracy,
precision/recall, or confusion matrix was ever produced, so there was no
way to know how good the model actually was.

Usage:
    python -m deepscan.evaluate --checkpoint deepfake_model_v2.pth --data-dir Dataset/Test
"""

from __future__ import annotations

import argparse
import json

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets

from deepscan.checkpoint import load_checkpoint
from deepscan.data import eval_transform
from deepscan.labels import resolve_class_to_idx
from deepscan.model import build_model


def evaluate(checkpoint_path: str, data_dir: str, batch_size: int = 32) -> dict:
    ckpt = load_checkpoint(checkpoint_path)
    class_to_idx, labels_verified = resolve_class_to_idx(ckpt, ".")
    fake_idx = class_to_idx["Fake"]

    model = build_model(ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dataset = datasets.ImageFolder(data_dir, transform=eval_transform(ckpt.get("img_size", 224)))
    if dataset.class_to_idx != class_to_idx:
        raise ValueError(
            f"Test set class_to_idx {dataset.class_to_idx} does not match "
            f"checkpoint's {class_to_idx}. Fix folder names or the checkpoint's "
            "label mapping before trusting these numbers."
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred, y_fake_prob = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            probs = torch.softmax(model(inputs), dim=1)
            y_true.extend((labels == fake_idx).int().tolist())
            y_fake_prob.extend(probs[:, fake_idx].tolist())
            y_pred.extend((probs[:, fake_idx] > 0.5).int().tolist())

    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics = {
        "n_samples": len(y_true),
        "labels_verified": labels_verified,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, zero_division=0),
        "recall_fake": recall_score(y_true, y_pred, zero_division=0),
        "f1_fake": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_fake_prob) if len(set(y_true)) > 1 else None,
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["Real", "Fake"],
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True,
                         help="ImageFolder-style test set, held out from training")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default=None, help="Optional path to write metrics as JSON")
    args = parser.parse_args()

    metrics = evaluate(args.checkpoint, args.data_dir, args.batch_size)
    print(json.dumps(metrics, indent=2))
    if not metrics["labels_verified"]:
        print("\nWARNING: this checkpoint's Real/Fake label mapping is unverified — "
              "these metrics may be measuring the inverse of what they claim to.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
