"""Downloads a model checkpoint from the main project's GitHub Releases
if it isn't already present locally, verifying its checksum.

Copy of ../../scripts/download_model.py for this standalone backend/
deployable unit (see deepscan/__init__.py) — see that file for the
full history of why the checkpoint isn't committed to git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

REPO = "ShaunT06/AI-Deepfake-Detector"
# This copy lives at the root of the backend/ deployable unit (see
# backend/deepscan/__init__.py) — models/ is a sibling, not a parent dir.
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "manifest.json")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(filename: str = "deepfake_model.pth", dest_dir: str = ".") -> str:
    dest_path = os.path.join(dest_dir, filename)

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    if filename not in manifest:
        raise KeyError(f"No manifest entry for {filename!r} in {MANIFEST_PATH}")
    entry = manifest[filename]

    if os.path.exists(dest_path):
        if _sha256(dest_path) == entry["sha256"]:
            return dest_path
        print(f"[download_model] {dest_path} exists but checksum mismatch — re-downloading.", file=sys.stderr)

    url = f"https://github.com/{REPO}/releases/download/{entry['release_tag']}/{entry['asset_name']}"
    print(f"[download_model] Downloading {url} -> {dest_path}", file=sys.stderr)
    tmp_path = dest_path + ".part"
    urllib.request.urlretrieve(url, tmp_path)

    actual = _sha256(tmp_path)
    if actual != entry["sha256"]:
        os.remove(tmp_path)
        raise ValueError(
            f"Checksum mismatch for {filename}: expected {entry['sha256']}, got {actual}. "
            "Download was not used."
        )
    os.replace(tmp_path, dest_path)
    return dest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filename", default="deepfake_model.pth")
    parser.add_argument("--dest-dir", default=".")
    args = parser.parse_args()
    path = ensure_model(args.filename, args.dest_dir)
    print(path)
