#!/usr/bin/env python3
"""
Render the AI DJ transition as an audio file.

Takes two audio files and the transition prediction, then creates
a mixed audio file you can listen to.
"""
import sys
import json
import argparse
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent.parent))


def apply_volume_curve(audio: np.ndarray, curve: np.ndarray, sr: int) -> np.ndarray:
    """Apply volume automation curve to audio."""
    # Resample curve to match audio length
    audio_length = len(audio)
    curve_indices = np.linspace(0, len(curve) - 1, audio_length)
    audio_curve = np.interp(curve_indices, np.arange(len(curve)), curve)
    
    # Convert from 0-1 to dB, then to linear gain
    # Curve is normalized 0-1, map to -60dB to 0dB
    db_curve = audio_curve * 60 - 60  # 0-1 -> -60 to 0 dB
    linear_gain = 10 ** (db_curve / 20)  # dB to linear
    
    # Apply gain
    return audio * linear_gain[:, np.newaxis] if audio.ndim == 2 else audio * linear_gain


def apply_eq(audio: np.ndarray, bass_gain: float, mid_gain: float, high_gain: float, sr: int) -> np.ndarray:
    """Apply simple 3-band EQ."""
    # Simple IIR filters for EQ bands
    # Bass: ~100 Hz
    # Mid: ~1000 Hz
    # High: ~5000 Hz
    
    bass_db = (bass_gain - 0.5) * 12  # -6dB to +6dB
    mid_db = (mid_gain - 0.5) * 12
    high_db = (high_gain - 0.5) * 12
    
    # Convert dB to linear gain
    bass_gain_lin = 10 ** (bass_db / 20)
    mid_gain_lin = 10 ** (mid_db / 20)
    high_gain_lin = 10 ** (high_db / 20)
    
    # Apply filters (simplified)
    if audio.ndim == 1:
        audio_stereo = np.column_stack([audio, audio])
    else:
        audio_stereo = audio
    
    # Low-pass for bass
    sos_bass = signal.butter(4, 100 / (sr / 2), btype='low')
    bass = signal.sosfilt(sos_bass, audio_stereo)
    
    # Band-pass for mid
    sos_mid = signal.butter(4, [200 / (sr / 2), 3000 / (sr / 2)], btype='band')
    mid = signal.sosfilt(sos_mid, audio_stereo)
    
    # High-pass for high
    sos_high = signal.butter(4, 3000 / (sr / 2), btype='high')
    high = signal.sosfilt(sos_high, audio_stereo)
    
    # Combine with gains
    result = (bass * bass_gain_lin + mid * mid_gain_lin + high * high_gain_lin)
    
    return result if audio.ndim == 2 else result[:, 0]


def apply_filter(audio: np.ndarray, freq_curve: np.ndarray, resonance: float, sr: int) -> np.ndarray:
    """Apply low-pass filter with frequency automation."""
    if audio.ndim == 1:
        audio_stereo = np.column_stack([audio, audio])
    else:
        audio_stereo = audio
    
    # Map curve (0-1) to frequency (20Hz - 20000Hz)
    freq_values = 20 + freq_curve * (20000 - 20)
    
    # Resample freq curve to audio length
    audio_length = len(audio_stereo)
    freq_indices = np.linspace(0, len(freq_curve) - 1, audio_length)
    audio_freq = np.interp(freq_indices, np.arange(len(freq_curve)), freq_values)
    
    # Apply time-varying filter (simplified: use average cutoff)
    avg_freq = np.mean(audio_freq)
    cutoff = avg_freq / (sr / 2)
    cutoff = np.clip(cutoff, 0.001, 0.99)
    
    sos = signal.butter(4, cutoff, btype='low')
    filtered = signal.sosfilt(sos, audio_stereo)
    
    return filtered if audio.ndim == 2 else filtered[:, 0]


def render_transition(audio_a_path: str, 
                     audio_b_path: str,
                     transition_data: dict,
                     output_path: str,
                     transition_start_a: float = None,
                     fade_in_b: float = 0.0):
    """
    Render the transition between two audio files.
    
    Args:
        audio_a_path: Path to outgoing song
        audio_b_path: Path to incoming song
        transition_data: Dictionary with transition prediction and curves
        output_path: Where to save the mixed output
        transition_start_a: Where to start transition in song A (seconds from end)
        fade_in_b: How much of song B to fade in (seconds, from start)
    """
    print("="*60)
    print("RENDERING AI DJ TRANSITION")
    print("="*60)
    
    # Load audio files
    print(f"\nLoading audio files...")
    print(f"  Song A: {Path(audio_a_path).name}")
    audio_a, sr_a = librosa.load(audio_a_path, sr=None, mono=False)
    
    print(f"  Song B: {Path(audio_b_path).name}")
    audio_b, sr_b = librosa.load(audio_b_path, sr=None, mono=False)
    
    # Resample to same sample rate
    target_sr = max(sr_a, sr_b)
    if sr_a != target_sr:
        if audio_a.ndim == 2:
            audio_a = librosa.resample(audio_a[0], orig_sr=sr_a, target_sr=target_sr)
            audio_a = librosa.resample(audio_a[1], orig_sr=sr_a, target_sr=target_sr) if audio_a.ndim == 2 else audio_a
        else:
            audio_a = librosa.resample(audio_a, orig_sr=sr_a, target_sr=target_sr)
    
    if sr_b != target_sr:
        if audio_b.ndim == 2:
            audio_b = librosa.resample(audio_b[0], orig_sr=sr_b, target_sr=target_sr)
            audio_b = librosa.resample(audio_b[1], orig_sr=sr_b, target_sr=target_sr) if audio_b.ndim == 2 else audio_b
        else:
            audio_b = librosa.resample(audio_b, orig_sr=sr_b, target_sr=target_sr)
    
    print(f"  Sample rate: {target_sr} Hz")
    
    # Ensure stereo
    if audio_a.ndim == 1:
        audio_a = np.column_stack([audio_a, audio_a])
    if audio_b.ndim == 1:
        audio_b = np.column_stack([audio_b, audio_b])
    
    # Get transition parameters
    duration_bars = transition_data.get('duration_bars', 4)
    tempo = 128  # Default, could extract from audio
    duration_sec = (duration_bars * 4 * 60) / tempo  # bars * beats/bar * 60s / bpm
    
    if 'curves' in transition_data:
        curves = transition_data['curves']
        time_points = np.array(curves.get('time', []))
        
        if len(time_points) > 0:
            duration_sec = max(time_points)
        
        volume_a_curve = np.array(curves.get('volume_a', []))
        volume_b_curve = np.array(curves.get('volume_b', []))
        bass_a_curve = np.array(curves.get('bass_a', []))
        bass_b_curve = np.array(curves.get('bass_b', []))
        mid_a_curve = np.array(curves.get('mid_a', []))
        mid_b_curve = np.array(curves.get('mid_b', []))
        high_a_curve = np.array(curves.get('high_a', []))
        high_b_curve = np.array(curves.get('high_b', []))
        filter_freq_curve = np.array(curves.get('filter_freq', []))
        filter_res_curve = np.array(curves.get('filter_res', []))
    else:
        # Default linear crossfade
        num_points = int(duration_sec * target_sr / 512)
        volume_a_curve = np.linspace(1.0, 0.0, num_points)
        volume_b_curve = np.linspace(0.0, 1.0, num_points)
        bass_a_curve = np.ones(num_points) * 0.5
        bass_b_curve = np.ones(num_points) * 0.5
        mid_a_curve = np.ones(num_points) * 0.5
        mid_b_curve = np.ones(num_points) * 0.5
        high_a_curve = np.ones(num_points) * 0.5
        high_b_curve = np.ones(num_points) * 0.5
        filter_freq_curve = np.linspace(0.0, 1.0, num_points)
        filter_res_curve = np.ones(num_points) * 0.3
    
    print(f"\nTransition duration: {duration_sec:.2f} seconds ({duration_bars} bars)")
    
    # Determine where to start transition in each song
    transition_samples = int(duration_sec * target_sr)
    
    # Start transition near end of song A
    if transition_start_a is None:
        # Use last 30 seconds of song A, or if song is shorter, use appropriate portion
        transition_start_a_samples = max(len(audio_a) - transition_samples - int(5 * target_sr), 
                                        len(audio_a) // 2)
    else:
        transition_start_a_samples = int(transition_start_a * target_sr)
    
    # Extract transition segments
    audio_a_segment = audio_a[transition_start_a_samples:transition_start_a_samples + transition_samples]
    
    # Start from beginning of song B
    audio_b_segment = audio_b[:transition_samples]
    
    # Pad if necessary
    if len(audio_a_segment) < transition_samples:
        padding = transition_samples - len(audio_a_segment)
        audio_a_segment = np.pad(audio_a_segment, ((0, padding), (0, 0)), mode='constant')
    
    if len(audio_b_segment) < transition_samples:
        padding = transition_samples - len(audio_b_segment)
        audio_b_segment = np.pad(audio_b_segment, ((0, padding), (0, 0)), mode='constant')
    
    # Apply automation curves
    print("\nApplying automation curves...")
    
    # Volume curves
    audio_a_processed = apply_volume_curve(audio_a_segment, volume_a_curve, target_sr)
    audio_b_processed = apply_volume_curve(audio_b_segment, volume_b_curve, target_sr)
    
    # EQ automation (simplified: apply average EQ)
    avg_bass_a = np.mean(bass_a_curve)
    avg_mid_a = np.mean(mid_a_curve)
    avg_high_a = np.mean(high_a_curve)
    audio_a_processed = apply_eq(audio_a_processed, avg_bass_a, avg_mid_a, avg_high_a, target_sr)
    
    avg_bass_b = np.mean(bass_b_curve)
    avg_mid_b = np.mean(mid_b_curve)
    avg_high_b = np.mean(high_b_curve)
    audio_b_processed = apply_eq(audio_b_processed, avg_bass_b, avg_mid_b, avg_high_b, target_sr)
    
    # Filter (simplified)
    avg_res = np.mean(filter_res_curve)
    audio_a_processed = apply_filter(audio_a_processed, filter_freq_curve, avg_res, target_sr)
    audio_b_processed = apply_filter(audio_b_processed, filter_freq_curve, avg_res, target_sr)
    
    # Mix together
    print("Mixing audio...")
    mixed = audio_a_processed + audio_b_processed
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 0.95:
        mixed = mixed * (0.95 / max_val)
    
    # Add some context: include a bit of song A before and song B after
    context_before = int(5 * target_sr)  # 5 seconds before
    context_after = int(5 * target_sr)  # 5 seconds after
    
    start_a = max(0, transition_start_a_samples - context_before)
    end_b = min(len(audio_b), transition_samples + context_after)
    
    final_output = np.concatenate([
        audio_a[start_a:transition_start_a_samples],  # Context before
        mixed,  # Transition
        audio_b[transition_samples:end_b]  # Context after
    ], axis=0)
    
    # Save output
    print(f"\nSaving to: {output_path}")
    sf.write(output_path, final_output, target_sr)
    
    duration_output = len(final_output) / target_sr
    print(f"✓ Rendered transition: {duration_output:.1f} seconds")
    print(f"  Output file: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Render AI DJ transition as audio file')
    parser.add_argument('--song-a', type=str, required=True, help='Path to song A (outgoing)')
    parser.add_argument('--song-b', type=str, required=True, help='Path to song B (incoming)')
    parser.add_argument('--transition-json', type=str, required=True,
                       help='Path to transition prediction JSON')
    parser.add_argument('--output', '-o', type=str, default='transition_mix.wav',
                       help='Output audio file path')
    parser.add_argument('--start-time-a', type=float, default=None,
                       help='Start transition in song A (seconds from end, or absolute)')
    
    args = parser.parse_args()
    
    # Load transition data
    with open(args.transition_json) as f:
        transition_data = json.load(f)
    
    # Render
    render_transition(
        args.song_a,
        args.song_b,
        transition_data,
        args.output,
        transition_start_a=args.start_time_a
    )
    
    print("\n" + "="*60)
    print("✓ Transition mix complete!")
    print(f"  Listen to: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()

