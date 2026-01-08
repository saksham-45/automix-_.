#!/usr/bin/env python3
"""Smart mix creator - finds optimal transition points and creates smooth blends"""
import json
import numpy as np
import soundfile as sf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.smart_mixer import SmartMixer

print("="*60)
print("SMART SONG MIXER")
print("="*60)

# Initialize smart mixer
mixer = SmartMixer()

# Load AI transition data if available
ai_data = None
try:
    with open('youtube_transition.json') as f:
        trans = json.load(f)
        if 'curves' in trans:
            ai_data = trans
            print("\n✓ Using AI transition curves from youtube_transition.json")
except:
    print("\n⚠ No AI transition data found, using smooth crossfade")

# Create mix using smart transition finder
print("\nFinding optimal transition points...")
mixed_audio = mixer.create_smooth_mix(
    'temp_audio/song_a.wav',
    'temp_audio/song_b.wav',
    transition_duration=16.0,
    ai_transition_data=ai_data
)

# Save
output_file = 'ai_dj_mix.wav'
print(f"\nSaving mix to: {output_file}")
sf.write(output_file, mixed_audio, mixer.sr)

duration = len(mixed_audio) / mixer.sr
print(f"\n{'='*60}")
print("✅ SMOOTH MIX COMPLETE!")
print(f"{'='*60}")
print(f"\n🎵 File: {output_file}")
print(f"   Duration: {duration:.1f} seconds")
print(f"\n🎧 Features:")
print(f"   ✓ Optimal transition points found automatically")
print(f"   ✓ Beat-matched alignment")
print(f"   ✓ Smooth, gradual volume curves")
print(f"\n🚀 Play it: open {output_file}")
print(f"{'='*60}")
