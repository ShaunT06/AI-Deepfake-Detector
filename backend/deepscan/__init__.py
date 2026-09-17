"""DeepScan inference-only package.

This is a deliberate, minimal copy of the relevant modules from
../src/deepscan (model.py, checkpoint.py, labels.py, inference.py),
scoped to what the FastAPI inference API needs. It's duplicated rather
than imported across the repo boundary because this backend/ directory
is deployed as its own standalone unit (e.g. pushed to a Hugging Face
Space as an independent git history) with its own Docker build context
that doesn't have access to ../src.

Training, evaluation, and the Streamlit app use the canonical versions
in src/deepscan/. If you change model.py/checkpoint.py/labels.py there,
mirror the change here.
"""

__version__ = "2.0.0"
