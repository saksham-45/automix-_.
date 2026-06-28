"""
Centralized runtime settings.
Values come from environment variables with safe defaults for production.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_float(name: str, default: float) -> float:
    """Parse a float env var; fall back to default on bad input so a malformed
    value doesn't crash every module that imports settings at import time."""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


# Core mix defaults
TRANSITION_LENGTH = _env_float("TRANSITION_LENGTH", 30)
TRANSITION_WINDOW = _env_float("TRANSITION_WINDOW", 64)
LUFS_TARGET = _env_float("LUFS_TARGET", -14)
SR = _env_int("SAMPLE_RATE", 44100)

# Feature toggles
ENABLE_AGENTIC_DURATION = os.getenv("ENABLE_AGENTIC_DURATION", "false").lower() in ("1", "true", "yes")

# Paths
CACHE_DIR = Path(os.getenv("CACHE_DIR", BASE_DIR / "data" / "cache" / "stream"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", BASE_DIR / "temp_audio"))

# Server
PORT = _env_int("PORT", 5005)
HOST = os.getenv("HOST", "0.0.0.0")

# YouTube / network
YTDLP_EXTRACTOR_ARGS = os.getenv("YTDLP_EXTRACTOR_ARGS", "youtube:player_client=android")
DOWNLOAD_TIMEOUT_SEC = _env_int("DOWNLOAD_TIMEOUT_SEC", 120)
