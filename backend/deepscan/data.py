"""Datasets and transforms.

Train and eval use *different* transforms — the original `train.py` ran
`RandomHorizontalFlip`/`RandomRotation` on the validation set too, which
makes validation accuracy noisy and not comparable across epochs. Only the
training transform should be randomized.
"""

from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from deepscan.model import IMG_SIZE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """No augmentation — used for validation, test, and inference so
    numbers are comparable run to run and match what the app does."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_dataloaders(data_dir: str, batch_size: int = 16, num_workers: int = 2,
                       img_size: int = IMG_SIZE):
    """Expects data_dir/Train/<class>/*.jpg and data_dir/Validation/<class>/*.jpg,
    matching the original repo's `Dataset/` layout."""
    train_dataset = datasets.ImageFolder(f"{data_dir}/Train", transform=train_transform(img_size))
    val_dataset = datasets.ImageFolder(f"{data_dir}/Validation", transform=eval_transform(img_size))

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Train/Validation class_to_idx mismatch: "
            f"{train_dataset.class_to_idx} vs {val_dataset.class_to_idx}. "
            "Check that both splits have the same class subfolder names."
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_dataset.class_to_idx
