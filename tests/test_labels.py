import json

from deepscan.labels import DEFAULT_CLASS_TO_IDX, resolve_class_to_idx


def test_embedded_class_to_idx_wins(tmp_path):
    ckpt = {"class_to_idx": {"Fake": 1, "Real": 0}}
    mapping, verified = resolve_class_to_idx(ckpt, str(tmp_path))
    assert mapping == {"Fake": 1, "Real": 0}
    assert verified is True


def test_sidecar_override_used_when_no_embedded_mapping(tmp_path):
    sidecar = tmp_path / "deepfake_model.classes.json"
    sidecar.write_text(json.dumps({"Fake": 1, "Real": 0}))

    ckpt = {"class_to_idx": None}
    mapping, verified = resolve_class_to_idx(ckpt, str(tmp_path))
    assert mapping == {"Fake": 1, "Real": 0}
    assert verified is True


def test_falls_back_to_unverified_default(tmp_path):
    ckpt = {"class_to_idx": None}
    mapping, verified = resolve_class_to_idx(ckpt, str(tmp_path))
    assert mapping == DEFAULT_CLASS_TO_IDX
    assert verified is False
