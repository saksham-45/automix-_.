import os
import json
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.getcwd())

from api.batch_processor import BatchProcessor

def generate_demo(idx=0):
    print(f"initializing demo for transition {idx}...")
    mix_id = "da54b08a_final"
    bp = BatchProcessor(cache_dir=f'temp_audio/cache/{mix_id}')
    
    # Restore cache to verify we have everything
    print("Restoring cache...")
    # Load playlist
    with open("data/playlists/da54b08a.json") as f:
        playlist_data = json.load(f)
    playlist = playlist_data['songs']
    
    bp.restore_from_cache(mix_id, playlist)
    
    print("Loading content...")
    # Load Song A
    song_a_path = bp.completed_songs.get(idx)
    if not song_a_path:
        print(f"Song {idx} not found in cache")
        return
    song_a, sr = sf.read(str(song_a_path))
    print(f"Song A ({idx}) loaded: {len(song_a)/sr:.2f}s")
    
    # Load Song B
    song_b_path = bp.completed_songs.get(idx + 1)
    if not song_b_path:
        print(f"Song {idx+1} not found in cache")
        return
    song_b, sr_b = sf.read(str(song_b_path))
    
    # Resample if needed
    if sr != 44100:
        song_a = librosa.resample(song_a, orig_sr=sr, target_sr=44100)
        sr = 44100
    if sr_b != 44100:
        song_b = librosa.resample(song_b, orig_sr=sr_b, target_sr=44100)
    
    print(f"Song B ({idx+1}) loaded: {len(song_b)/44100:.2f}s")
        
    # Load Transition
    trans_data = bp.get_transition(idx)
    if isinstance(trans_data, tuple):
        trans_audio, trans_meta = trans_data
    else:
        print(f"Transition {idx} not found or invalid format")
        return

    print(f"Transition {idx} loaded: {len(trans_audio)/sr:.2f}s")
    print(f"Metadata: {json.dumps(trans_meta, indent=2)}")
    
    # LOGIC 1: Switch/Cut Song A
    print(f"Processing Song {idx}...")
    
    if trans_meta and 'start_time_from_end' in trans_meta:
        start_time_from_end = float(trans_meta['start_time_from_end'])
        # Clamp
        start_time_from_end = max(5.0, min(45.0, start_time_from_end))
        transition_start_sec = max(0, (len(song_a)/sr) - start_time_from_end)
    else:
        transition_start_sec = max(0, (len(song_a)/sr) - 16.0)
        
    transition_start_sample = int(transition_start_sec * sr)
    
    # Cut Song A at transition point + Micro Fade
    MICRO_FADE_MS = 10
    micro_fade_samples = int((MICRO_FADE_MS / 1000.0) * sr)
    
    cut_point = transition_start_sample
    song_pre_cut = song_a[:cut_point]
    
    # Blend Splice
    fade_start = len(song_pre_cut) - micro_fade_samples
    song_body = song_pre_cut[:fade_start]
    song_fade_out = song_pre_cut[fade_start:]
    trans_fade_in = trans_audio[:micro_fade_samples]
    trans_body = trans_audio[micro_fade_samples:]
    
    # Match dims for micro fade
    if song_fade_out.ndim == 1: song_fade_out = song_fade_out[:, np.newaxis]
    if trans_fade_in.ndim == 1: trans_fade_in = trans_fade_in[:, np.newaxis]
    
    # Stereo match
    if trans_fade_in.shape[1] > song_fade_out.shape[1]:
        song_fade_out = np.column_stack([song_fade_out, song_fade_out])
        if song_body.ndim == 1: song_body = np.column_stack([song_body, song_body])
    elif song_fade_out.shape[1] > trans_fade_in.shape[1]:
        trans_fade_in = np.column_stack([trans_fade_in, trans_fade_in])
        if trans_body.ndim == 1: trans_body = np.column_stack([trans_body, trans_body])
        
    # Fade
    t = np.linspace(0, np.pi/2, micro_fade_samples)
    fo = np.cos(t)[:, np.newaxis]
    fi = np.sin(t)[:, np.newaxis]
    
    mix_splice = song_fade_out * fo + trans_fade_in * fi
    
    # LOGIC 2: Trim Song B
    print(f"Processing Song {idx+1}...")
    point_b = float(trans_meta['transition_point_b'])
    duration = float(trans_meta['transition_duration'])
    start_offset_sec = point_b + duration
    start_trim_samples = int(start_offset_sec * sr)
    
    song_b_trimmed = song_b[start_trim_samples:]
    
    # Match dims for song B
    if song_b_trimmed.ndim == 1 and trans_body.ndim > 1:
        song_b_trimmed = np.column_stack([song_b_trimmed, song_b_trimmed])
    elif song_b_trimmed.ndim > 1 and trans_body.ndim == 1:
        trans_body = np.column_stack([trans_body, trans_body])
        mix_splice = np.column_stack([mix_splice, mix_splice])
        if song_body.ndim == 1: song_body = np.column_stack([song_body, song_body])
        
    # CONCATENATE
    print("Stitching...")
    if song_body.ndim == 1 and trans_body.ndim > 1: song_body = np.column_stack([song_body, song_body])
    
    full_mix = np.concatenate([song_body, mix_splice, trans_body, song_b_trimmed])
    
    output_path = f"output_demo_mix_{idx}.wav"
    sf.write(output_path, full_mix, sr)
    print(f"SUCCESS: Saved to {output_path}")
    print(f"Total Duration: {len(full_mix)/sr:.2f}s")
    return output_path

if __name__ == "__main__":
    # Get index from args or default to 1
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    generate_demo(idx)
