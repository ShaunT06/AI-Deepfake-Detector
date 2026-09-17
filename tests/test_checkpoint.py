import torch

from deepscan.checkpoint import load_checkpoint, save_checkpoint
from deepscan.model import build_model


def test_round_trip_preserves_class_to_idx_and_weights(tmp_path):
    model = build_model("resnet18", pretrained=False)
    path = str(tmp_path / "ckpt.pth")
    save_checkpoint(path, model, backbone="resnet18",
                     class_to_idx={"Fake": 1, "Real": 0}, metrics={"acc": 0.9})

    loaded = load_checkpoint(path)
    assert loaded["backbone"] == "resnet18"
    assert loaded["class_to_idx"] == {"Fake": 1, "Real": 0}
    assert loaded["metrics"] == {"acc": 0.9}

    reloaded_model = build_model("resnet18", pretrained=False)
    reloaded_model.load_state_dict(loaded["state_dict"])
    for p1, p2 in zip(model.parameters(), reloaded_model.parameters()):
        assert torch.equal(p1, p2)


def test_legacy_bare_state_dict_still_loads(tmp_path):
    model = build_model("resnet18", pretrained=False)
    path = str(tmp_path / "legacy.pth")
    torch.save(model.state_dict(), path)  # old format: no wrapper dict

    loaded = load_checkpoint(path)
    assert loaded["format"] == "legacy-state-dict"
    assert loaded["backbone"] == "resnet18"
    assert loaded["class_to_idx"] is None
