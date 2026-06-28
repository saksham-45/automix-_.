#!/usr/bin/env python3
"""
Batch-clean the Astroworld transition WAVs using conservative best practices:
- DC removal
- 30 Hz high-pass to tame sub-rumble
- Short fade-in/out to kill clicks at boundaries
- Headroom trim to -1 dBFS peak (adds ~3 dB safety vs prior renders)

Outputs land in cleaned_astroworld_mix/<file>.wav
"""
from pathlib import Path
import argparse
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt


def highpass(data: np.ndarray, sr: int, cutoff_hz: float = 30.0, order: int = 4) -> np.ndarray:
    b, a = butter(order, cutoff_hz / (sr * 0.5), btype="high", analog=False)
    return filtfilt(b, a, data, axis=0)


def apply_fades(data: np.ndarray, sr: int, fade_in_ms: float = 15.0, fade_out_ms: float = 20.0) -> np.ndarray:
    y = data.copy()
    fade_in_len = int(sr * fade_in_ms / 1000.0)
    fade_out_len = int(sr * fade_out_ms / 1000.0)
    if fade_in_len > 0:
        ramp = np.linspace(0.0, 1.0, fade_in_len)[:, None]
        y[:fade_in_len] *= ramp
    if fade_out_len > 0:
        ramp = np.linspace(1.0, 0.0, fade_out_len)[:, None]
        y[-fade_out_len:] *= ramp
    return y


def trim_headroom(data: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    ceiling_lin = 10 ** (ceiling_db / 20.0)
    peak = np.max(np.abs(data))
    if peak <= 1e-9:
        return data
    if peak > ceiling_lin:
        data = data * (ceiling_lin / peak) * 0.98  # leave a tiny cushion
    return data


def clean_file(src_path: Path, dst_path: Path) -> None:
    audio, sr = sf.read(src_path)
    if audio.ndim == 1:
        audio = audio[:, None]

    # Remove DC per channel
    audio = audio - np.mean(audio, axis=0, keepdims=True)

    # High-pass at 30 Hz to reduce rumble
    audio = highpass(audio, sr, cutoff_hz=30.0, order=4)

    # Short fades to prevent clicks at boundaries
    audio = apply_fades(audio, sr, fade_in_ms=15.0, fade_out_ms=20.0)

    # Trim peak to -1 dBFS
    audio = trim_headroom(audio, ceiling_db=-1.0)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst_path, audio, sr)


def main():
    parser = argparse.ArgumentParser(description="Clean Astroworld WAVs for rumble/click/clipping.")
    parser.add_argument(
        "--input-dir", type=str, default="astroworld_mix", help="Directory with original WAVs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cleaned_astroworld_mix",
        help="Directory to write cleaned WAVs",
    )
    args = parser.parse_args()

    src_dir = Path(args.input_dir)
    dst_dir = Path(args.output_dir)
    files = sorted(src_dir.glob("*.wav"))
    if not files:
        raise SystemExit(f"No WAV files found in {src_dir}")

    for wav in files:
        out_path = dst_dir / wav.name
        print(f"Cleaning {wav.name} -> {out_path}")
        clean_file(wav, out_path)


if __name__ == "__main__":
    main()
