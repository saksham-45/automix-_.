#!/usr/bin/env python3
"""Create a Boiler-Room–style long club blend from two audio files.

Usage:
    python scripts/club_mix.py SONG_A SONG_B [-o out.wav] [--bars 16]

See BOILER_ROOM_MIXING_PLAN.md for the model. A is the outgoing track (mixed out
near its outro), B is the incoming track (mixed in on its intro/breakdown).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.club_mixer import render_club_mix


def main():
    ap = argparse.ArgumentParser(description="Boiler-Room–style club blend of two tracks")
    ap.add_argument("song_a", help="outgoing track")
    ap.add_argument("song_b", help="incoming track")
    ap.add_argument("-o", "--out", default="club_blend.wav", help="output wav")
    ap.add_argument("--bars", type=int, default=16, help="blend length in bars (default 16)")
    ap.add_argument("--sr", type=int, default=44100)
    args = ap.parse_args()

    print(f"Loading A: {args.song_a}")
    ya, _ = librosa.load(args.song_a, sr=args.sr, mono=False)
    print(f"Loading B: {args.song_b}")
    yb, _ = librosa.load(args.song_b, sr=args.sr, mono=False)
    # librosa returns (channels, samples) for stereo -> transpose to (samples, channels)
    if ya.ndim == 2:
        ya = ya.T
    if yb.ndim == 2:
        yb = yb.T

    mixed, info = render_club_mix(ya, yb, sr=args.sr, blend_bars=args.bars)

    print("\n=== CLUB BLEND ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    sf.write(args.out, mixed, args.sr)
    print(f"\nWrote {args.out} ({len(mixed)/args.sr:.1f}s)")


if __name__ == "__main__":
    main()
