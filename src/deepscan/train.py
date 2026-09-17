"""Training entrypoint.

Usage:
    python -m deepscan.train --data-dir Dataset --backbone efficientnet_b0 \\
        --epochs 15 --fine-tune-epochs 10 --out deepfake_model_v2.pth

Two-phase training:
  1. Head-only warmup ('--epochs'): backbone frozen, fast even on CPU.
  2. Fine-tuning ('--fine-tune-epochs'): unfreezes the last backbone block
     at a lower LR. This is what actually lets the model learn
     manipulation-specific artifacts instead of relying purely on frozen
     ImageNet features — skip it (--fine-tune-epochs 0) for a quick
     baseline, but expect a meaningfully weaker model.

Differences from the original script this replaces:
  - argparse instead of hardcoded constants
  - a real validation loop (previously `val_loader` was built but never
    used) with best-checkpoint selection instead of just saving whatever
    the last epoch produced
  - a fixed random seed for reproducibility
  - saves `train_dataset.class_to_idx` into the checkpoint (see
    checkpoint.py / labels.py) instead of losing that mapping
  - a held-out epoch-by-epoch log written to disk, not just stdout
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from deepscan.checkpoint import save_checkpoint
from deepscan.data import build_dataloaders
from deepscan.model import (
    DEFAULT_BACKBONE,
    SUPPORTED_BACKBONES,
    build_model,
    freeze_backbone,
    unfreeze_last_block,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    running_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100.0 * correct / total


def train_phase(model, train_loader, val_loader, device, epochs, lr, phase_name, log):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - start

        line = (f"[{phase_name}] epoch {epoch}/{epochs} | "
                 f"train_loss={train_loss:.3f} train_acc={train_acc:.2f}% | "
                 f"val_loss={val_loss:.3f} val_acc={val_acc:.2f}% | {elapsed:.1f}s")
        print(line)
        log.append({"phase": phase_name, "epoch": epoch, "train_loss": train_loss,
                     "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default="Dataset",
                         help="Folder containing Train/ and Validation/ subfolders")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=SUPPORTED_BACKBONES)
    parser.add_argument("--epochs", type=int, default=8, help="Head-only warmup epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=6,
                         help="Additional epochs with the last backbone block unfrozen (0 to skip)")
    parser.add_argument("--lr", type=float, default=1e-3, help="LR for the head-only warmup phase")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4, help="LR for the fine-tuning phase")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="deepfake_model_v2.pth")
    parser.add_argument("--log-out", default=None, help="Optional path to write the per-epoch JSON log")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} | backbone={args.backbone}")

    train_loader, val_loader, class_to_idx = build_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"class_to_idx: {class_to_idx}")

    model = build_model(args.backbone, pretrained=True).to(device)
    log: list[dict] = []

    freeze_backbone(model, args.backbone)
    best_acc = train_phase(model, train_loader, val_loader, device, args.epochs, args.lr, "warmup", log)

    if args.fine_tune_epochs > 0:
        unfreeze_last_block(model, args.backbone)
        finetune_acc = train_phase(
            model, train_loader, val_loader, device,
            args.fine_tune_epochs, args.fine_tune_lr, "finetune", log,
        )
        best_acc = max(best_acc, finetune_acc)

    save_checkpoint(
        args.out, model, backbone=args.backbone, class_to_idx=class_to_idx,
        metrics={"best_val_acc": best_acc},
        notes=f"Trained with {args}",
    )
    print(f"Saved checkpoint to {args.out} (best val_acc={best_acc:.2f}%)")

    log_path = args.log_out or (args.out + ".trainlog.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"Wrote training log to {log_path}")


if __name__ == "__main__":
    main()
