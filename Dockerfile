# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# autmox — Club AutoMixer. Designed to deploy on a free CPU box (e.g. a Hugging
# Face Spaces "Docker" Space, 2 vCPU / 16 GB). Stem separation (demucs, on CPU-only
# torch) is installed by default; build with --build-arg INSTALL_STEMS=false for a
# small, torch-free image (the app then uses 3-band EQ transitions only).
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# System deps: ffmpeg (yt-dlp transcode), libsndfile (soundfile), git (demucs).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps (cached layer: only re-runs when requirements change) ------
COPY requirements.txt requirements-stems.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    # librosa imports pkg_resources at runtime; setuptools >=81 removed it.
    && pip install --no-cache-dir "setuptools<81"

ARG INSTALL_STEMS=true
RUN if [ "$INSTALL_STEMS" = "true" ]; then \
        pip install --no-cache-dir -r requirements-stems.txt \
            --extra-index-url https://download.pytorch.org/whl/cpu && \
        pip install --no-cache-dir --no-deps \
            "git+https://github.com/adefossez/demucs.git" ; \
    fi

# --- App code + non-root runtime user (Hugging Face Spaces runs as uid 1000) -
RUN useradd -m -u 1000 user
COPY --chown=user:user . /app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    PYTHONUNBUFFERED=1

# Pre-generate the royalty-free demo loops into the image (idempotent at runtime).
RUN python -m src.demo_samples data/demo

EXPOSE 7860
ENTRYPOINT ["./docker-entrypoint.sh"]
