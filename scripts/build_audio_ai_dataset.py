#!/usr/bin/env python3
"""
Build a reusable audio-analysis dataset for downstream AI inspection.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stem_separator import StemSeparator
from src.utils import ensure_dir, save_analysis_json


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo = np.percentile(values, 5)
    hi = np.percentile(values, 95)
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    scaled = (values - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    std = float(np.std(values))
    if std < 1e-9:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def save_matrix_artifacts(matrix: np.ndarray, npz_path: Path, png_path: Path) -> None:
    np.savez_compressed(npz_path, data=matrix.astype(np.float32))
    image_data = robust_normalize(matrix)
    image = np.flipud((image_data * 255).astype(np.uint8))
    Image.fromarray(image, mode="L").save(png_path)


def save_vector_artifacts(values: np.ndarray, npz_path: Path) -> None:
    np.savez_compressed(npz_path, data=np.asarray(values, dtype=np.float32))


def contiguous_segments(score: np.ndarray, times: np.ndarray, threshold: float, label: str):
    segments = []
    active = score >= threshold
    start_idx = None
    for idx, is_active in enumerate(active):
        if is_active and start_idx is None:
            start_idx = idx
        elif not is_active and start_idx is not None:
            end_idx = idx - 1
            segments.append(build_segment(score, times, start_idx, end_idx, label))
            start_idx = None
    if start_idx is not None:
        segments.append(build_segment(score, times, start_idx, len(score) - 1, label))
    return segments


def build_segment(score: np.ndarray, times: np.ndarray, start_idx: int, end_idx: int, label: str):
    end_idx = max(start_idx, end_idx)
    return {
        "label": label,
        "start_sec": float(times[start_idx]),
        "end_sec": float(times[end_idx]),
        "peak_score": float(np.max(score[start_idx:end_idx + 1])),
        "mean_score": float(np.mean(score[start_idx:end_idx + 1])),
    }


def write_event_csv(events, output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "start_sec", "end_sec", "peak_score", "mean_score"],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(event)


def summarize_top_events(events, limit: int = 12):
    return sorted(events, key=lambda item: item["peak_score"], reverse=True)[:limit]


def compute_band_energies(power: np.ndarray, freqs: np.ndarray):
    bands = {
        "sub_bass": (20, 60),
        "bass": (60, 200),
        "low_mid": (200, 500),
        "mid": (500, 2000),
        "upper_mid": (2000, 5000),
        "air": (5000, 16000),
    }
    energies = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            energies[band_name] = np.sum(power[mask], axis=0)
        else:
            energies[band_name] = np.zeros(power.shape[1], dtype=np.float32)
    return energies


def frame_clip_ratio(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    padded = np.pad(np.abs(y), frame_length // 2, mode="reflect")
    frames = librosa.util.frame(padded, frame_length=frame_length, hop_length=hop_length)
    return np.mean(frames >= 0.995, axis=0)


def create_feature_embeddings(feature_matrix: np.ndarray, components: int = 16) -> np.ndarray:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)
    n_components = min(components, scaled.shape[1], scaled.shape[0])
    if n_components < 2:
        return scaled.astype(np.float32)
    reduced = PCA(n_components=n_components, random_state=0).fit_transform(scaled)
    return reduced.astype(np.float32)


def detect_anomalies(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] < 32:
        return np.zeros(embeddings.shape[0], dtype=np.float32)
    detector = IsolationForest(
        n_estimators=200,
        contamination=0.04,
        random_state=0,
    )
    detector.fit(embeddings)
    raw_scores = -detector.score_samples(embeddings)
    return robust_normalize(raw_scores)


def save_stems(stems, sample_rate: int, output_dir: Path):
    stem_summary = {}
    for stem_name, stem_audio in stems.items():
        stem_path = output_dir / f"{stem_name}.wav"
        sf.write(stem_path, stem_audio, sample_rate)
        mono = np.mean(stem_audio, axis=1)
        stem_summary[stem_name] = {
            "path": str(stem_path.resolve()),
            "peak": float(np.max(np.abs(stem_audio))),
            "rms": float(np.sqrt(np.mean(mono ** 2))),
        }
    return stem_summary


def analyze_track(audio_path: Path, track_dir: Path, with_stems: bool) -> dict:
    arrays_dir = ensure_dir(track_dir / "arrays")
    images_dir = ensure_dir(track_dir / "images")
    reports_dir = ensure_dir(track_dir / "reports")
    stems_dir = ensure_dir(track_dir / "stems") if with_stems else None

    stereo, sr = librosa.load(audio_path, sr=None, mono=False)
    if stereo.ndim == 1:
        stereo = np.vstack([stereo, stereo])
    mono = np.mean(stereo, axis=0).astype(np.float32)
    peak = float(np.max(np.abs(mono))) or 1.0
    mono = mono / peak

    n_fft = 2048
    hop_length = 512
    frame_length = n_fft

    stft = librosa.stft(mono, n_fft=n_fft, hop_length=hop_length)
    stft_mag = np.abs(stft)
    stft_db = librosa.amplitude_to_db(stft_mag + 1e-9, ref=np.max)
    mel = librosa.feature.melspectrogram(
        y=mono,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
        fmax=min(16000, sr // 2),
    )
    mel_db = librosa.power_to_db(mel + 1e-9, ref=np.max)
    cqt = np.abs(librosa.cqt(mono, sr=sr, hop_length=hop_length, n_bins=84, bins_per_octave=12))
    cqt_db = librosa.amplitude_to_db(cqt + 1e-9, ref=np.max)

    times = librosa.frames_to_time(np.arange(stft_mag.shape[1]), sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = stft_mag ** 2
    band_energies = compute_band_energies(power, freqs)

    harmonic, percussive = librosa.effects.hpss(mono)
    onset_env = librosa.onset.onset_strength(y=percussive, sr=sr, hop_length=hop_length)
    onset_env = gaussian_filter1d(onset_env.astype(np.float32), sigma=1)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=False,
        units="frames",
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    rms = librosa.feature.rms(S=stft_mag, frame_length=frame_length, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(S=stft_mag, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=stft_mag, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=stft_mag + 1e-9)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft_mag, sr=sr, roll_percent=0.95)[0]
    zcr = librosa.feature.zero_crossing_rate(mono, frame_length=frame_length, hop_length=hop_length)[0]
    chroma = librosa.feature.chroma_stft(S=stft_mag, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=20, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(S=stft_mag, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)

    low_mask = (freqs >= 20) & (freqs < 120)
    mid_mask = (freqs >= 120) & (freqs < 2000)
    high_mask = (freqs >= 2000) & (freqs < 8000)
    low_onset = librosa.onset.onset_strength(S=power[low_mask], sr=sr, hop_length=hop_length)
    mid_onset = librosa.onset.onset_strength(S=power[mid_mask], sr=sr, hop_length=hop_length)
    high_onset = librosa.onset.onset_strength(S=power[high_mask], sr=sr, hop_length=hop_length)

    sub_energy = band_energies["sub_bass"]
    bass_energy = band_energies["bass"]
    air_energy = band_energies["air"]
    total_energy = np.sum(power, axis=0) + 1e-9
    sub_ratio = sub_energy / total_energy
    bass_ratio = bass_energy / total_energy
    air_ratio = air_energy / total_energy
    clip_ratio = frame_clip_ratio(mono, frame_length=frame_length, hop_length=hop_length)
    spectral_flux = np.sqrt(np.sum(np.diff(stft_mag, axis=1, prepend=stft_mag[:, :1]) ** 2, axis=0))

    drum_layer_score = robust_normalize(
        np.maximum(0.0, zscore(low_onset))
        + np.maximum(0.0, zscore(mid_onset))
        + np.maximum(0.0, zscore(high_onset))
        + np.maximum(0.0, zscore(onset_env))
    )
    bass_rumble_score = robust_normalize(
        np.maximum(0.0, zscore(np.log1p(sub_energy)))
        + np.maximum(0.0, zscore(np.log1p(sub_ratio)))
        + np.maximum(0.0, zscore(200.0 - np.minimum(rolloff, 200.0)))
        + np.maximum(0.0, zscore(0.35 - flatness))
    )
    distortion_score = robust_normalize(
        np.maximum(0.0, zscore(clip_ratio))
        + np.maximum(0.0, zscore(flatness))
        + np.maximum(0.0, zscore(air_ratio))
        + np.maximum(0.0, zscore(zcr))
        + np.maximum(0.0, zscore(spectral_flux))
    )

    feature_matrix = np.column_stack(
        [
            rms,
            centroid,
            bandwidth,
            flatness,
            rolloff,
            zcr,
            sub_ratio,
            bass_ratio,
            air_ratio,
            onset_env[: len(times)],
            low_onset[: len(times)],
            mid_onset[: len(times)],
            high_onset[: len(times)],
            spectral_flux,
            mfcc[:13].T,
            chroma.T,
            contrast.T,
            tonnetz.T,
        ]
    )
    embeddings = create_feature_embeddings(feature_matrix, components=16)
    anomaly_score = detect_anomalies(embeddings)
    unwanted_sound_score = robust_normalize(
        0.45 * anomaly_score + 0.35 * distortion_score + 0.20 * bass_rumble_score
    )

    save_matrix_artifacts(mel_db, arrays_dir / "mel_spectrogram.npz", images_dir / "mel_spectrogram.png")
    save_matrix_artifacts(stft_db, arrays_dir / "stft_spectrogram.npz", images_dir / "stft_spectrogram.png")
    save_matrix_artifacts(cqt_db, arrays_dir / "cqt_spectrogram.npz", images_dir / "cqt_spectrogram.png")
    save_vector_artifacts(onset_env, arrays_dir / "onset_envelope.npz")
    save_vector_artifacts(onset_frames.astype(np.float32), arrays_dir / "onset_frames.npz")
    save_vector_artifacts(onset_times.astype(np.float32), arrays_dir / "onset_times.npz")
    save_vector_artifacts(times.astype(np.float32), arrays_dir / "frame_times.npz")
    save_vector_artifacts(drum_layer_score, arrays_dir / "drum_layer_score.npz")
    save_vector_artifacts(bass_rumble_score, arrays_dir / "bass_rumble_score.npz")
    save_vector_artifacts(distortion_score, arrays_dir / "distortion_score.npz")
    save_vector_artifacts(anomaly_score, arrays_dir / "anomaly_score.npz")
    save_vector_artifacts(unwanted_sound_score, arrays_dir / "unwanted_sound_score.npz")
    save_vector_artifacts(embeddings, arrays_dir / "feature_embeddings.npz")

    feature_summary = {
        "tempo_bpm": float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)[0]),
        "duration_sec": float(len(mono) / sr),
        "sample_rate": int(sr),
        "channels": int(stereo.shape[0]),
        "frame_count": int(len(times)),
        "mean_rms": float(np.mean(rms)),
        "mean_centroid_hz": float(np.mean(centroid)),
        "mean_flatness": float(np.mean(flatness)),
        "mean_sub_ratio": float(np.mean(sub_ratio)),
        "mean_bass_ratio": float(np.mean(bass_ratio)),
        "mean_air_ratio": float(np.mean(air_ratio)),
        "peak_bass_rumble_score": float(np.max(bass_rumble_score)),
        "peak_distortion_score": float(np.max(distortion_score)),
        "peak_unwanted_sound_score": float(np.max(unwanted_sound_score)),
    }

    events = []
    events.extend(contiguous_segments(drum_layer_score, times, 0.82, "stacked_drum_activity"))
    events.extend(contiguous_segments(bass_rumble_score, times, 0.80, "bass_rumble"))
    events.extend(contiguous_segments(distortion_score, times, 0.80, "distortion"))
    events.extend(contiguous_segments(unwanted_sound_score, times, 0.78, "unwanted_sound"))
    events = summarize_top_events(events, limit=24)
    write_event_csv(events, reports_dir / "suspicious_events.csv")

    drum_events = contiguous_segments(drum_layer_score, times, 0.84, "stacked_drum_activity")
    write_event_csv(summarize_top_events(drum_events, limit=24), reports_dir / "drum_events.csv")

    stems_summary = None
    if with_stems and stems_dir is not None:
        separator = StemSeparator()
        stems = separator.separate_stems(stereo.T.astype(np.float32), sr=sr)
        stems_summary = save_stems(stems, sr, stems_dir)

    report = {
        "audio_path": str(audio_path.resolve()),
        "features": feature_summary,
        "top_events": events,
        "onset_count": int(len(onset_frames)),
        "stems": stems_summary,
    }
    save_analysis_json(report, reports_dir / "summary.json")
    return report


def collect_audio_files(input_dir: Path):
    patterns = ("*.wav", "*.mp3", "*.flac", "*.aiff", "*.m4a")
    files = []
    for pattern in patterns:
        files.extend(input_dir.glob(pattern))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Build an audio-analysis dataset for AI.")
    parser.add_argument("input_dir", type=str, help="Directory containing audio files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/astroworld_ai_dataset",
        help="Directory where dataset artifacts will be written",
    )
    parser.add_argument(
        "--with-stems",
        action="store_true",
        help="Run Demucs stem separation and save stems for each file",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    audio_files = collect_audio_files(input_dir)
    if not audio_files:
        raise SystemExit(f"No audio files found in {input_dir}")

    manifest = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "track_count": len(audio_files),
        "tracks": [],
        "methods": [
            "mel_spectrogram",
            "stft_spectrogram",
            "cqt_spectrogram",
            "onset_detection",
            "feature_embeddings",
            "anomaly_scoring",
            "bass_rumble_scoring",
            "distortion_scoring",
            "optional_stem_separation",
        ],
    }

    for index, audio_path in enumerate(audio_files, start=1):
        track_name = sanitize_name(audio_path.stem)
        track_dir = ensure_dir(Path(output_dir) / f"{index:02d}_{track_name}")
        print(f"[{index}/{len(audio_files)}] analyzing {audio_path.name}")
        report = analyze_track(audio_path, track_dir, with_stems=args.with_stems)
        manifest["tracks"].append(
            {
                "name": audio_path.name,
                "folder": str(Path(track_dir).resolve()),
                "summary_path": str((Path(track_dir) / "reports" / "summary.json").resolve()),
                "event_path": str((Path(track_dir) / "reports" / "suspicious_events.csv").resolve()),
                "top_unwanted_sound_score": report["features"]["peak_unwanted_sound_score"],
            }
        )

    save_analysis_json(manifest, Path(output_dir) / "dataset_manifest.json")


if __name__ == "__main__":
    main()
