# DeepScan inference API — standalone image for this backend/ directory.
# Built for Hugging Face Spaces (Docker SDK), but runs anywhere that can
# run a container (Render, Fly.io, Cloud Run, ...).
FROM python:3.12-slim

# OpenCV (via opencv-python-headless / pytorch-grad-cam) needs libgl/libglib
# at runtime even in "headless" mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces (Docker SDK) expects the app to listen on 7860 by
# default. Other hosts can override PORT.
ENV PORT=7860
EXPOSE 7860

# Download the model at build time (baked into the image) rather than at
# cold-start, so requests aren't slow behind a first-run download. Safe to
# skip on hosts with no build-time network access — main.py's startup
# hook downloads it lazily if this step is skipped.
RUN python -c "from download_model import ensure_model; ensure_model(dest_dir='/app')" || true

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
