"""Model construction: single source of truth for the network architecture.

Previously the ResNet-18 + custom head definition was duplicated between
`train.py` and `app.py`, with no guardrail against the two drifting apart.
Both now call `build_model()`.

Supports two backbones:
  - "resnet18": the original architecture, kept so the legacy
    `deepfake_model.pth` checkpoint still loads.
  - "efficientnet_b0": a stronger, more modern backbone (better ImageNet
    accuracy than ResNet-18 at a similar parameter count, and generally a
    better feature extractor for subtle manipulation artifacts). This is
    the recommended default for any *new* training run.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

SUPPORTED_BACKBONES = ("resnet18", "efficientnet_b0")
DEFAULT_BACKBONE = "efficientnet_b0"
NUM_CLASSES = 2
IMG_SIZE = 224


def build_model(backbone: str = DEFAULT_BACKBONE, num_classes: int = NUM_CLASSES,
                 pretrained: bool = True) -> nn.Module:
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        return model

    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        return model

    raise ValueError(f"Unknown backbone {backbone!r}; supported: {SUPPORTED_BACKBONES}")


def get_gradcam_target_layers(model: nn.Module, backbone: str) -> list:
    """Returns the layer(s) Grad-CAM should hook, per architecture."""
    if backbone == "resnet18":
        return [model.layer4[-1]]
    if backbone == "efficientnet_b0":
        return [model.features[-1]]
    raise ValueError(f"Unknown backbone {backbone!r}; supported: {SUPPORTED_BACKBONES}")


def freeze_backbone(model: nn.Module, backbone: str) -> None:
    """Freezes every parameter except the classification head — used for
    fast head-only training on a small dataset / CPU."""
    head_params = set(_head_parameters(model, backbone))
    for param in model.parameters():
        param.requires_grad = param in head_params


def unfreeze_last_block(model: nn.Module, backbone: str) -> None:
    """Unfreezes the head plus the final feature block, for a fine-tuning
    phase once the head has converged. Fine-tuning some backbone layers
    (not just the head) generally improves accuracy over head-only
    training, at the cost of needing a lower learning rate and more
    careful monitoring for overfitting on small datasets."""
    freeze_backbone(model, backbone)
    if backbone == "resnet18":
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif backbone == "efficientnet_b0":
        for param in model.features[-1].parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown backbone {backbone!r}; supported: {SUPPORTED_BACKBONES}")


def _head_parameters(model: nn.Module, backbone: str):
    if backbone == "resnet18":
        return list(model.fc.parameters())
    if backbone == "efficientnet_b0":
        return list(model.classifier.parameters())
    raise ValueError(f"Unknown backbone {backbone!r}; supported: {SUPPORTED_BACKBONES}")
