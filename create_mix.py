#!/usr/bin/env python3
"""Smart mix creator - finds optimal transition points and creates smooth blends"""
import json
import numpy as np
import soundfile as sf
import sys
import time
from pathlib import Path

#region agent log
import os
log_path = '/Users/saksham/untitled folder 7/.cursor/debug.log'
os.makedirs(os.path.dirname(log_path), exist_ok=True)
try:
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"create_mix.py:11","message":"Script start","data":{},"timestamp":int(time.time()*1000)}) + '\n')
except Exception as e:
    print(f"DEBUG LOG ERROR: {e}")
#endregion

sys.path.insert(0, str(Path(__file__).parent))

#region agent log
import_start = time.time()
try:
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"create_mix.py:17","message":"Before SmartMixer import","data":{},"timestamp":int(time.time()*1000)}) + '\n')
except: pass
#endregion

from src.smart_mixer import SmartMixer

#region agent log
import_time = time.time() - import_start
try:
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"create_mix.py:23","message":"SmartMixer imported","data":{"time_sec":import_time},"timestamp":int(time.time()*1000)}) + '\n')
except: pass
#endregion

print("="*60)
print("SMART SONG MIXER")
print("="*60)

# Initialize smart mixer
#region agent log
init_start = time.time()
try:
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"create_mix.py:33","message":"Before SmartMixer() init","data":{},"timestamp":int(time.time()*1000)}) + '\n')
except: pass
#endregion

mixer = SmartMixer()

#region agent log
init_time = time.time() - init_start
try:
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"create_mix.py:37","message":"SmartMixer initialized","data":{"time_sec":init_time},"timestamp":int(time.time()*1000)}) + '\n')
except: pass
#endregion

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

#region agent log
mix_start = time.time()
try:
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"create_mix.py:50","message":"Before create_smooth_mix call","data":{},"timestamp":int(time.time()*1000)}) + '\n')
except: pass
#endregion

mixed_audio = mixer.create_smooth_mix(
    'temp_audio/song_a.wav',
    'temp_audio/song_b.wav',
    transition_duration=16.0,
    ai_transition_data=ai_data
)

#region agent log
mix_time = time.time() - mix_start
try:
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"create_mix.py:60","message":"create_smooth_mix complete","data":{"time_sec":mix_time},"timestamp":int(time.time()*1000)}) + '\n')
except: pass
#endregion

# Save with timestamp to avoid overwriting
from datetime import datetime
import os

base_name = 'ai_dj_mix'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'{base_name}_{timestamp}.wav'

print(f"\nSaving mix to: {output_file}")
sf.write(output_file, mixed_audio, mixer.sr)

# Delete old mix files AFTER saving new one (keep only the newest 3)
try:
    cwd = os.path.dirname(os.path.abspath(__file__)) or '.'
    mix_files = sorted([f for f in os.listdir(cwd) if f.startswith(base_name) and f.endswith('.wav')], 
                       key=lambda x: os.path.getmtime(os.path.join(cwd, x)), reverse=True)
    deleted_count = 0
    for old_file in mix_files[3:]:  # Keep newest 3, delete rest
        try:
            old_path = os.path.join(cwd, old_file)
            os.remove(old_path)
            deleted_count += 1
        except Exception as e:
            pass
    if deleted_count > 0:
        print(f"🗑️  Deleted {deleted_count} old mix file(s)")
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
