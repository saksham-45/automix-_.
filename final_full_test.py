#!/usr/bin/env python3.13
import sys
from pathlib import Path
import os
import json
import numpy as np
import soundfile as sf
import librosa
import shutil

# Add project root to path
sys.path.append(os.getcwd())

from api.batch_processor import BatchProcessor

def create_full_production_mix(url_a, url_b, output_name):
    # Setup fresh env
    CACHE_DIR = Path('temp_audio/cache/final_prod_test')
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True)

    bp = BatchProcessor(cache_dir=str(CACHE_DIR))

    songs = [
        {"id": "song_a", "url": url_a, "title": "Track A"},
        {"id": "song_b", "url": url_b, "title": "Track B"}
    ]

    print(f"\n🚀 STARTING FULL PRODUCTION MIX")
    print(f"Track A: {url_a}")
    print(f"Track B: {url_b}")
    print("-" * 60)

    # 1. Download
    print("\n[1/4] Downloading...")
    audio_paths = []
    for song in songs:
        path = bp.download_full_song(song)
        if not path:
            print(f"❌ Failed to download {song['id']}")
            return
        audio_paths.append(path)

    # 2. Analyze
    print("\n[2/4] Analyzing (Full song profiling)...")
    analyses = []
    for i, path in enumerate(audio_paths):
        # We need a fresh analysis because structure depends on full song now
        # SmartMixer is used inside analyze_song
        analysis = bp.analyze_song(songs[i], path)
        if not analysis:
            print(f"❌ Failed to analyze {songs[i]['id']}")
            return
        analyses.append(analysis)

    # 3. Create Transition
    print("\n[3/4] Finding optimal transition...")
    transition_path = bp.create_transition(
        songs[0], songs[1],
        audio_paths[0], audio_paths[1],
        analyses[0], analyses[1],
        0 # Index
    )

    if not transition_path:
        print("❌ Transition creation failed.")
        return

    # 4. Stitching Full Mix
    print("\n[4/4] Stitching full mix together...")
    
    # Load all audio
    y_a, sr = librosa.load(str(audio_paths[0]), sr=44100)
    y_b, _ = librosa.load(str(audio_paths[1]), sr=44100)
    
    trans_audio, sr_t = sf.read(str(transition_path))
    if sr_t != 44100:
        trans_audio = librosa.resample(trans_audio, orig_sr=sr_t, target_sr=44100)
    
    # Load metadata
    with open(transition_path.with_suffix('.json')) as f:
        meta = json.load(f)
    
    # Logic from generate_demo_mix
    if meta and 'start_time_from_end' in meta:
        points_a_from_end = float(meta['start_time_from_end'])
        transition_start_sample = int(len(y_a) - (points_a_from_end * 44100))
    else:
        transition_start_sample = int(len(y_a) - (16.0 * 44100))
        
    print(f"\n[4/4] Stitching full mix together...")
    
    # Load all audio
    y_a, sr = librosa.load(str(audio_paths[0]), sr=44100)
    y_b, _ = librosa.load(str(audio_paths[1]), sr=44100)
    
    trans_audio, sr_t = sf.read(str(transition_path))
    if sr_t != 44100:
        trans_audio = librosa.resample(trans_audio, orig_sr=sr_t, target_sr=44100)
    
    # Load metadata
    with open(transition_path.with_suffix('.json')) as f:
        meta = json.load(f)
    
    duration_a = len(y_a) / 44100
    duration_b = len(y_b) / 44100
    
    # Updated stitching logic using transition_start_a for precision
    if meta and 'transition_start_a' in meta:
        transition_start_sec = float(meta['transition_start_a'])
    elif meta and 'start_time_from_end' in meta:
        transition_start_sec = float(duration_a - meta['start_time_from_end'])
    else:
        transition_start_sec = float(duration_a - 16.0)
    
    # Ensure transition_start_sec is valid
    transition_start_sec = max(0.0, min(duration_a - 1.0, transition_start_sec))
    transition_start_sample = int(transition_start_sec * 44100)
    
    # Duration of the transition itself
    trans_duration_sec = len(trans_audio) / 44100
    
    # LOGGING FOR DEBUG
    print(f"DEBUG: Song A Duration: {duration_a:.2f}s")
    print(f"DEBUG: Transition Start A: {transition_start_sec:.2f}s")
    print(f"DEBUG: Transition Duration: {trans_duration_sec:.2f}s")
    
    # Micro fade for splice (prevent clicks)
    MICRO_FADE_MS = 25
    mf_samples = int((MICRO_FADE_MS / 1000.0) * 44100)
    
    song_pre_cut = y_a[:transition_start_sample]
    fade_start = max(0, len(song_pre_cut) - mf_samples)
    
    # Components
    a_body = song_pre_cut[:fade_start]
    a_fade = song_pre_cut[fade_start:]
    
    # Match length of t_fade_in to a_fade
    t_fade_in = trans_audio[:len(a_fade)]
    t_body = trans_audio[len(a_fade):]
    
    # Ensure stereo
    def to_stereo(x):
        if x is None or len(x) == 0: return np.empty((0, 2))
        if x.ndim == 1: return np.column_stack([x, x])
        return x
    
    a_body = to_stereo(a_body)
    a_fade = to_stereo(a_fade)
    t_fade_in = to_stereo(t_fade_in)
    t_body = to_stereo(t_body)
    y_b = to_stereo(y_b)

    # Crossfade
    if len(a_fade) > 0:
        t_curve = np.linspace(0, np.pi/2, len(a_fade))
        fo = np.sqrt(np.cos(t_curve))[:, np.newaxis] # Equal power
        fi = np.sqrt(np.sin(t_curve))[:, np.newaxis]
        splice = a_fade * fo + t_fade_in * fi
    else:
        splice = np.empty((0, 2))
    
    # B Tail Logic
    # point_b is where song B STARTS in the transition global time (aligned_b)
    point_b = float(meta.get('transition_point_b', meta.get('transition_start_b', 0.0)))
    trans_duration_meta = float(meta.get('transition_duration', 16.0))
    
    # The transition core is exactly 'trans_duration_meta' long.
    # It covers Song B from [point_b] to [point_b + trans_duration_meta].
    # So we continue Song B from [point_b + trans_duration_meta].
    start_offset_b_sec = point_b + trans_duration_meta
    start_offset_b_sample = int(start_offset_b_sec * 44100)
    
    print(f"DEBUG: Song B Point B: {point_b:.2f}s")
    print(f"DEBUG: Song B Continue From: {start_offset_b_sec:.2f}s")
    
    b_tail = y_b[start_offset_b_sample:] if start_offset_b_sample < len(y_b) else np.empty((0, 2))
    
    # Combine
    full_mix = np.concatenate([a_body, splice, t_body, b_tail])
    
    # Save to Desktop
    final_path = Path.home() / "Desktop" / output_name
    sf.write(str(final_path), full_mix, 44100)
    
    print(f"\n✅ SUCCESS! Full mix saved to: {final_path}")
    print(f"Total Duration: {len(full_mix)/44100:.2f}s")
    print(f"Points Used: A @ {meta['transition_point_a']:.2f}s, B @ {meta['transition_point_b']:.2f}s")
    print(f"Technique: {meta['technique']}")
    print(f"\nRun: open ~/Desktop/{output_name}")

if __name__ == "__main__":
    url1 = "https://youtu.be/U8F5G5wR1mk?si=cNG13E1lG69s8KFL"
    url2 = "https://youtu.be/KhhN6m7Lfyw?si=jGYoDu3tQ5n1Pp3q"
    create_full_production_mix(url1, url2, "weeknd_21savage_mix.wav")
