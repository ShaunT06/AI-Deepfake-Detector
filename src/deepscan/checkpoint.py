"""Checkpoint format that carries its own metadata.

The original `deepfake_model.pth` was a bare `model.state_dict()` with no
record of which backbone produced it or which folder name mapped to which
class index — information that turned out to matter (see
`deepscan.labels`). Every checkpoint saved by this codebase from now on is
a small dict instead, so that question never has to be reverse-engineered
again.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class Checkpoint:
    backbone: str
    class_to_idx: dict
    state_dict: dict
    img_size: int = 224
    metrics: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    notes: str = ""


def save_checkpoint(path: str, model: nn.Module, backbone: str, class_to_idx: dict,
                     img_size: int = 224, metrics: dict | None = None, notes: str = "") -> None:
    payload = {
        "format": "deepscan-checkpoint-v1",
        "backbone": backbone,
        "class_to_idx": class_to_idx,
        "img_size": img_size,
        "metrics": metrics or {},
        "created_at": time.time(),
        "notes": notes,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location="cpu") -> dict[str, Any]:
    """Loads either the new dict-based format, or (for backward
    compatibility) a legacy bare state_dict, which is returned with
    backbone='resnet18' and class_to_idx=None to signal it must come from
    elsewhere (see deepscan.labels.resolve_class_to_idx)."""
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict) and obj.get("format") == "deepscan-checkpoint-v1":
        return obj
    # Legacy bare state_dict (e.g. the original deepfake_model.pth).
    return {
        "format": "legacy-state-dict",
        "backbone": "resnet18",
        "class_to_idx": None,
        "img_size": 224,
        "metrics": {},
        "state_dict": obj,
    }
