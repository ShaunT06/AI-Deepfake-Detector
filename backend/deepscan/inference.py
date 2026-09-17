"""Inference: load a checkpoint once, run prediction + Grad-CAM in a
single forward pass. Used by both the Streamlit app and tests."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from deepscan.checkpoint import load_checkpoint
from deepscan.data import eval_transform
from deepscan.labels import resolve_class_to_idx
from deepscan.model import build_model, get_gradcam_target_layers


@dataclass
class Prediction:
    is_fake: bool
    fake_prob: float
    real_prob: float
    backbone: str
    labels_verified: bool
    gradcam_image: Image.Image


class Predictor:
    """Wraps a loaded model + its resolved class mapping. Construct once
    (e.g. behind st.cache_resource) and reuse across requests."""

    def __init__(self, model_path: str):
        model_dir = os.path.dirname(os.path.abspath(model_path))
        ckpt = load_checkpoint(model_path)

        self.backbone = ckpt["backbone"]
        self.img_size = ckpt.get("img_size", 224)
        self.class_to_idx, self.labels_verified = resolve_class_to_idx(ckpt, model_dir)
        self.fake_idx = self.class_to_idx["Fake"]
        self.real_idx = self.class_to_idx["Real"]

        self.model = build_model(self.backbone, pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.transform = eval_transform(self.img_size)
        self.target_layers = get_gradcam_target_layers(self.model, self.backbone)

    def predict(self, image: Image.Image) -> Prediction:
        img_resized = image.convert("RGB").resize((self.img_size, self.img_size))
        input_tensor = self.transform(img_resized).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]

        fake_prob = float(probs[self.fake_idx])
        real_prob = float(probs[self.real_idx])
        is_fake = fake_prob > 0.5
        predicted_idx = self.fake_idx if is_fake else self.real_idx

        rgb_img = np.array(img_resized) / 255.0
        with GradCAM(model=self.model, target_layers=self.target_layers) as cam:
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(predicted_idx)],
            )[0]
        gradcam_image = Image.fromarray(
            show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True)
        )

        return Prediction(
            is_fake=is_fake,
            fake_prob=fake_prob,
            real_prob=real_prob,
            backbone=self.backbone,
            labels_verified=self.labels_verified,
            gradcam_image=gradcam_image,
        )
