import pytest

from deepscan.checkpoint import save_checkpoint
from deepscan.model import build_model


@pytest.fixture(params=["resnet18", "efficientnet_b0"])
def tiny_checkpoint(tmp_path, request):
    """A randomly-initialized (untrained) checkpoint, just to exercise the
    save/load/inference code paths without needing a real dataset, a GPU,
    or network access to download real weights."""
    backbone = request.param
    model = build_model(backbone, pretrained=False)
    path = str(tmp_path / "tiny_model.pth")
    save_checkpoint(
        path, model, backbone=backbone,
        class_to_idx={"Fake": 0, "Real": 1},
        metrics={"best_val_acc": 0.0},
        notes="test fixture, untrained",
    )
    return path
