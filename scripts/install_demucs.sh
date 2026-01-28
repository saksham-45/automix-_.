#!/usr/bin/env bash
# Install demucs from adefossez/demucs (no lameenc; we use WAV only).
# Run from project root with venv activated: source .venv/bin/activate && ./scripts/install_demucs.sh
set -e
echo "Installing demucs from https://github.com/adefossez/demucs (--no-deps to skip lameenc)..."
pip install --no-deps 'git+https://github.com/adefossez/demucs.git'
echo "Done. Stem separation (htdemucs) is available."
