import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_visual_proof(model, input_tensor, original_img_path):
    # Target the last layer of the ResNet for the heatmap
    target_layers = [model.layer4[-1]]
    
    # Load original image for overlay
    rgb_img = cv2.imread(original_img_path)[:, :, ::-1] # BGR to RGB
    rgb_img = np.float32(cv2.resize(rgb_img, (224, 224))) / 255
    
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Generate heatmap for "Fake" class (usually index 1)
    targets = [ClassifierOutputTarget(1)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # Create the X-ray overlay
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization