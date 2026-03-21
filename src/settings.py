"""
Centralized runtime settings.
Values come from environment variables with safe defaults for production.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Core mix defaults
TRANSITION_LENGTH = float(os.getenv("TRANSITION_LENGTH", "30"))
TRANSITION_WINDOW = float(os.getenv("TRANSITION_WINDOW", "64"))
LUFS_TARGET = float(os.getenv("LUFS_TARGET", "-14"))
SR = int(os.getenv("SAMPLE_RATE", "44100"))

# Feature toggles
ENABLE_AGENTIC_DURATION = os.getenv("ENABLE_AGENTIC_DURATION", "false").lower() in ("1", "true", "yes")

# Paths
CACHE_DIR = Path(os.getenv("CACHE_DIR", BASE_DIR / "data" / "cache" / "stream"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", BASE_DIR / "temp_audio"))

# Server
PORT = int(os.getenv("PORT", "5005"))
HOST = os.getenv("HOST", "0.0.0.0")

# YouTube / network
YTDLP_EXTRACTOR_ARGS = os.getenv("YTDLP_EXTRACTOR_ARGS", "youtube:player_client=android")
DOWNLOAD_TIMEOUT_SEC = int(os.getenv("DOWNLOAD_TIMEOUT_SEC", "120"))
