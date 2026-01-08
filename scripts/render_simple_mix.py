#!/usr/bin/env python3
"""
Simple transition renderer - creates a basic crossfade mix.

Uses the transition JSON to apply volume curves and create a mix.
"""
import sys
import json
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

def render_simple_transition(audio_a_path: str, audio_b_path: str, 
                            transition_json: str, output_path: str):
    """Render a simple crossfade transition."""
    print("="*60)
    print("RENDERING TRANSITION MIX")
    print("="*60)
    
    # Load transition data
    with open(transition_json) as f:
        transition = json.load(f)
    
    curves = transition.get('curves', {})
    duration_bars = transition.get('duration_bars', 4)
    
    # Estimate duration (4 bars at ~128 BPM = ~7.5 seconds)
    tempo = 128
    transition_duration = (duration_bars * 4 * 60) / tempo
    
    if curves and 'time' in curves and len(curves['time']) > 0:
        transition_duration = max(curves['time'])
    
    print(f"\nTransition duration: {transition_duration:.1f} seconds")
    
    # Load audio
    print(f"\nLoading audio...")
    print(f"  Song A: {Path(audio_a_path).name}")
    audio_a, sr = librosa.load(audio_a_path, sr=None, mono=False)
    
    print(f"  Song B: {Path(audio_b_path).name}")
    audio_b, sr_b = librosa.load(audio_b_path, sr=sr, mono=False)
    
    # Ensure stereo
    if audio_a.ndim == 1:
        audio_a = np.column_stack([audio_a, audio_a])
    if audio_b.ndim == 1:
        audio_b = np.column_stack([audio_b, audio_b])
    
    transition_samples = int(transition_duration * sr)
    
    # Get segments (end of A, start of B)
    start_a = max(0, len(audio_a) - transition_samples - int(5 * sr))  # 5 sec context
    segment_a = audio_a[start_a:start_a + transition_samples]
    segment_b = audio_b[:transition_samples]
    
    # Pad if needed
    if len(segment_a) < transition_samples:
        pad = transition_samples - len(segment_a)
        segment_a = np.pad(segment_a, ((0, pad), (0, 0)), mode='constant')
    if len(segment_b) < transition_samples:
        pad = transition_samples - len(segment_b)
        segment_b = np.pad(segment_b, ((0, pad), (0, 0)), mode='constant')
    
    # Apply volume curves
    print("Applying volume automation...")
    if 'volume_a' in curves and 'volume_b' in curves:
        vol_a = np.array(curves['volume_a'])
        vol_b = np.array(curves['volume_b'])
        
        # Resample curves to match audio length
        curve_len = len(segment_a)
        indices = np.linspace(0, len(vol_a) - 1, curve_len)
        vol_a_resampled = np.interp(indices, np.arange(len(vol_a)), vol_a)
        vol_b_resampled = np.interp(indices, np.arange(len(vol_b)), vol_b)
        
        # Convert to linear gain (0-1 to -60dB to 0dB)
        gain_a = 10 ** ((vol_a_resampled * 60 - 60) / 20)
        gain_b = 10 ** ((vol_b_resampled * 60 - 60) / 20)
        
        # Apply
        segment_a = segment_a * gain_a[:, np.newaxis]
        segment_b = segment_b * gain_b[:, np.newaxis]
    else:
        # Simple linear crossfade
        t = np.linspace(0, 1, len(segment_a))
        segment_a = segment_a * (1 - t)[:, np.newaxis]
        segment_b = segment_b * t[:, np.newaxis]
    
    # Mix
    mixed = segment_a + segment_b
    
    # Normalize
    max_val = np.max(np.abs(mixed))
    if max_val > 0.95:
        mixed = mixed * (0.95 / max_val)
    
    # Add context
    context_before = audio_a[max(0, start_a - int(5 * sr)):start_a]
    context_after = audio_b[transition_samples:transition_samples + int(5 * sr)]
    
    final = np.concatenate([context_before, mixed, context_after], axis=0)
    
    # Save
    print(f"\nSaving mix to: {output_path}")
    sf.write(output_path, final, sr)
    
    duration = len(final) / sr
    print(f"✓ Rendered {duration:.1f} seconds of audio")
    print(f"  File: {output_path}")
    
    return output_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--song-a', required=True)
    parser.add_argument('--song-b', required=True)
    parser.add_argument('--transition-json', required=True)
    parser.add_argument('--output', default='transition_mix.wav')
    
    args = parser.parse_args()
    render_simple_transition(args.song_a, args.song_b, args.transition_json, args.output)

