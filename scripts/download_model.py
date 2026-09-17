"""Downloads a model checkpoint from this repo's GitHub Releases if it
isn't already present locally, verifying its checksum.

The 45MB `deepfake_model.pth` used to be committed directly to git despite
`.gitignore` listing `*.pth` (it had been force-added). Every clone of the
repo paid that 45MB regardless of whether they needed the model. It now
lives as a GitHub Release asset instead; this script (called automatically
by the Streamlit app on startup) fetches it on demand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

REPO = "ShaunT06/AI-Deepfake-Detector"
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "manifest.json")


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
