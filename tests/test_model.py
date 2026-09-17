import pytest
import torch

from deepscan.model import (
    SUPPORTED_BACKBONES,
    build_model,
    freeze_backbone,
    get_gradcam_target_layers,
    unfreeze_last_block,
)


@pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
def test_forward_pass_shape(backbone):
    model = build_model(backbone, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)


@pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
def test_freeze_backbone_only_head_trainable(backbone):
    model = build_model(backbone, pretrained=False)
    freeze_backbone(model, backbone)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert trainable, "expected some trainable (head) parameters"
    assert frozen, "expected some frozen (backbone) parameters"
    head_prefix = "fc" if backbone == "resnet18" else "classifier"
    assert all(n.startswith(head_prefix) for n in trainable)


@pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
def test_unfreeze_last_block_unfreezes_more_than_head(backbone):
    model = build_model(backbone, pretrained=False)
    freeze_backbone(model, backbone)
    n_trainable_head_only = sum(p.requires_grad for p in model.parameters())
    unfreeze_last_block(model, backbone)
    n_trainable_after = sum(p.requires_grad for p in model.parameters())
    assert n_trainable_after > n_trainable_head_only


@pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
def test_gradcam_target_layers_resolve(backbone):
    model = build_model(backbone, pretrained=False)
    layers = get_gradcam_target_layers(model, backbone)
    assert len(layers) == 1


def test_unknown_backbone_raises():
    with pytest.raises(ValueError):
        build_model("resnet152", pretrained=False)
