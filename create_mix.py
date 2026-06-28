#!/usr/bin/env python3
"""Smart mix creator - finds optimal transition points and creates smooth blends"""
import json
import numpy as np
import soundfile as sf
import sys
import time
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
    transition_duration=30.0,
    ai_transition_data=ai_data
)


# Save with timestamp to avoid overwriting
from datetime import datetime
import os

base_name = 'ai_dj_mix'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'{base_name}_{timestamp}.wav'

print(f"\nSaving mix to: {output_file}")
sf.write(output_file, mixed_audio, mixer.sr)

# Move old mix files to data/old_mixes/ (keep only the newest one in working directory)
try:
    cwd = os.path.dirname(os.path.abspath(__file__)) or '.'
    old_mixes_dir = os.path.join(cwd, 'data', 'old_mixes')
    os.makedirs(old_mixes_dir, exist_ok=True)  # Ensure directory exists
    
    mix_files = sorted([f for f in os.listdir(cwd) if f.startswith(base_name) and f.endswith('.wav')], 
                       key=lambda x: os.path.getmtime(os.path.join(cwd, x)), reverse=True)
    moved_count = 0
    for old_file in mix_files[1:]:  # Keep newest 1 in cwd, move rest
        try:
            old_path = os.path.join(cwd, old_file)
            new_path = os.path.join(old_mixes_dir, old_file)
            os.rename(old_path, new_path)
            moved_count += 1
        except Exception as e:
            pass
    if moved_count > 0:
        print(f"📦 Moved {moved_count} old mix file(s) to data/old_mixes/")
except Exception as e:
    pass

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
