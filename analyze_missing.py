import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from src.smart_mixer import SmartMixer
import librosa
import json
import os

# Use the actual cache
MIX_ID = 'da54b08a_final'
song_ids = ['aca3a2df-eaba-4e', 'adb84f26-4252-48']
mixer = SmartMixer(sr=44100)

for song_id in song_ids:
    audio_path = f'temp_audio/cache/{MIX_ID}/songs/{song_id}.wav'
    json_path = f'temp_audio/cache/{MIX_ID}/songs/{song_id}.json'
    
    if os.path.exists(json_path):
        print(f"Skipping {song_id}, already exists.")
        continue

    print(f"Analyzing {audio_path}...")
    y, sr = librosa.load(audio_path, sr=44100)
    analysis = mixer._analyze_song_fast(y)

    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"✅ Analysis saved to {json_path}")
