#!/usr/bin/env python3
import json
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

print("="*60)
print("RENDERING AI DJ MIX")
print("="*60)

# Load transition
with open('youtube_transition.json') as f:
    trans = json.load(f)

curves = trans.get('curves', {})
print(f"\n✓ Transition: {trans.get('technique', 'unknown').upper()}, {trans.get('duration_bars', 4)} bars")

# Load audio with shorter duration
print("\n✓ Loading audio (30 seconds each)...")
audio_a, sr = librosa.load('temp_audio/song_a.wav', sr=44100, duration=30, mono=False)
audio_b, _ = librosa.load('temp_audio/song_b.wav', sr=44100, duration=30, mono=False)

if audio_a.ndim == 1:
    audio_a = np.column_stack([audio_a, audio_a])
if audio_b.ndim == 1:
    audio_b = np.column_stack([audio_b, audio_b])

# Transition
transition_dur = max(curves.get('time', [7.5])) if curves else 7.5
trans_samples = int(transition_dur * sr)

# Segments
start_a = max(0, len(audio_a) - trans_samples - int(3*sr))
seg_a = audio_a[start_a:start_a + trans_samples]
seg_b = audio_b[:trans_samples]

# Pad
if len(seg_a) < trans_samples:
    seg_a = np.pad(seg_a, ((0, trans_samples-len(seg_a)), (0,0)), 'constant')
if len(seg_b) < trans_samples:
    seg_b = np.pad(seg_b, ((0, trans_samples-len(seg_b)), (0,0)), 'constant')

# Apply volume curves
print("✓ Applying AI automation curves...")
if 'volume_a' in curves:
    vol_a = np.array(curves['volume_a'])
    vol_b = np.array(curves['volume_b'])
    idx = np.linspace(0, len(vol_a)-1, len(seg_a))
    gain_a = 10**((np.interp(idx, np.arange(len(vol_a)), vol_a) * 60 - 60) / 20)
    gain_b = 10**((np.interp(idx, np.arange(len(vol_b)), vol_b) * 60 - 60) / 20)
    seg_a = seg_a * gain_a[:, np.newaxis]
    seg_b = seg_b * gain_b[:, np.newaxis]

# Mix
mixed = seg_a + seg_b
max_val = np.max(np.abs(mixed))
if max_val > 0.95:
    mixed = mixed * (0.95 / max_val)

# Context
ctx_b = audio_a[max(0, start_a - int(3*sr)):start_a]
ctx_a = audio_b[trans_samples:min(len(audio_b), trans_samples + int(3*sr))]
final = np.concatenate([ctx_b, mixed, ctx_a], axis=0)

# Save
print(f"\n✓ Saving mix...")
sf.write('ai_dj_mix.wav', final, sr)

dur = len(final) / sr
size_mb = Path('ai_dj_mix.wav').stat().st_size / 1024 / 1024
print(f"\n{'='*60}")
print("✅ MIX COMPLETE!")
print(f"{'='*60}")
print(f"\n🎵 File: ai_dj_mix.wav")
print(f"   Duration: {dur:.1f} seconds")
print(f"   Size: {size_mb:.1f} MB")
print(f"\n🚀 Play it: open ai_dj_mix.wav")
print(f"{'='*60}")
