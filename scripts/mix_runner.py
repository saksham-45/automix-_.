#!/usr/bin/env python3
"""Run YouTube mix: download two URLs and write mixed WAV to a given path."""
import json
import sys
from pathlib import Path

# Run from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from create_mix_from_youtube import download_youtube_audio

TEMP_DIR = PROJECT_ROOT / "temp_audio"
TEMP_DIR.mkdir(exist_ok=True)


def run_youtube_mix(url1: str, url2: str, output_path: Path, duration: int = 60) -> None:
    """
    Download audio from two YouTube URLs, create a mix, and save to output_path.
    Raises on error.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    song_a = TEMP_DIR / "song_a.wav"
    song_b = TEMP_DIR / "song_b.wav"
    for p in (song_a, song_b):
        if p.exists():
            p.unlink()

    try:
        download_youtube_audio(url1, song_a, duration, from_end=True)
        download_youtube_audio(url2, song_b, duration, from_end=False)

        # Mix (same logic as create_mix.py but write to output_path)
        import soundfile as sf
        from src.smart_mixer import SmartMixer

        mixer = SmartMixer()
        ai_data = None
        try:
            with open(PROJECT_ROOT / "youtube_transition.json") as f:
                trans = json.load(f)
                if "curves" in trans:
                    ai_data = trans
        except Exception:
            pass

        mixed_audio = mixer.create_smooth_mix(
            str(song_a),
            str(song_b),
            transition_duration=16.0,
            ai_transition_data=ai_data,
        )
        sf.write(str(output_path), mixed_audio, mixer.sr)
    finally:
        for p in (song_a, song_b):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
