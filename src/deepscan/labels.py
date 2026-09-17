"""Class-label bookkeeping for the deepfake classifier.

`ImageFolder` assigns class index 0/1 by alphabetically sorting the
training subfolder names. The original `train.py` never recorded which
folder name ended up at which index, and the shipped `deepfake_model.pth`
is a bare `state_dict()` with no metadata attached — so there is no way to
recover the *true* mapping after the fact without either the original
`Dataset/` folder or retraining.

`app.py` and `engine.py` previously each guessed a different mapping
(index 0 = Fake vs. index 1 = Fake) with no way to tell which, if either,
was correct. This module makes that assumption a single, explicit,
overridable place instead of two silently-diverging hardcoded guesses.

To verify or correct the mapping for the current `deepfake_model.pth`,
run a few images you know are real and a few you know are fake through
the app. If verdicts come out inverted, create `deepfake_model.classes.json`
next to the model file:

    {"Fake": 0, "Real": 1}

with the indices swapped, and the app will pick it up automatically.
Going forward, `train.py` saves this mapping into the checkpoint itself
(see `checkpoint.py`) so this guessing game only applies to the legacy
weights file.
"""

import json
import os

# Inherited from the original app.py; UNVERIFIED against the real training
# data, since the folder names used to produce the legacy deepfake_model.pth
# are not recorded anywhere. Only used as a last-resort fallback for
# checkpoints that predate the checkpoint.py format and have no sidecar
# override. Treat any verdict derived from this fallback as suspect until
# you've spot-checked it (see module docstring).
DEFAULT_CLASS_TO_IDX = {"Fake": 0, "Real": 1}

SIDECAR_FILENAME = "deepfake_model.classes.json"


def resolve_class_to_idx(checkpoint: dict, model_dir: str,
                          sidecar_filename: str = SIDECAR_FILENAME):
    """Return (class_to_idx, is_verified) for a loaded checkpoint dict
    (see checkpoint.load_checkpoint).

    Preference order:
      1. class_to_idx embedded in the checkpoint itself (train.py saves
         this from `ImageFolder.class_to_idx` — always correct).
      2. a sidecar JSON override next to the model file, for manually
         correcting a legacy checkpoint without retraining it.
      3. the unverified default, as a last resort — is_verified=False.
    """
    embedded = checkpoint.get("class_to_idx")
    if embedded:
        return dict(embedded), True

    sidecar_path = os.path.join(model_dir, sidecar_filename)
    if os.path.exists(sidecar_path):
        with open(sidecar_path, encoding="utf-8") as f:
            return json.load(f), True

    return dict(DEFAULT_CLASS_TO_IDX), False
