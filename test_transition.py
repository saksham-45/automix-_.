import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from api.batch_processor import BatchProcessor
import json
import os

# Use the actual cache
MIX_ID = 'da54b08a_final'
bp = BatchProcessor(cache_dir=f'temp_audio/cache/{MIX_ID}')

with open("data/playlists/da54b08a.json") as f:
    playlist = json.load(f)['songs']

# Force restoration of any existing work
bp.restore_from_cache(MIX_ID, playlist)

# Transition 4 is between index 4 and 5
idx_a = 4
idx_b = 5
song_a = playlist[idx_a]
song_b = playlist[idx_b]

print(f"\n--- Testing Transition {idx_a} -> {idx_b} ---")
print(f"Song A: {song_a['title']}")
print(f"Song B: {song_b['title']}")

# Ensure songs and analysis are loaded
audio_a_path = Path(f'temp_audio/cache/{MIX_ID}/songs/{song_a["id"]}.wav')
audio_b_path = Path(f'temp_audio/cache/{MIX_ID}/songs/{song_b["id"]}.wav')
analysis_a_path = Path(f'temp_audio/cache/{MIX_ID}/songs/{song_a["id"]}.json')
analysis_b_path = Path(f'temp_audio/cache/{MIX_ID}/songs/{song_b["id"]}.json')

if not audio_a_path.exists() or not audio_b_path.exists():
    print(f"ERROR: Audio files not found at {audio_a_path} or {audio_b_path}")
    sys.exit(1)

with open(analysis_a_path) as f:
    analysis_a = json.load(f)
with open(analysis_b_path) as f:
    analysis_b = json.load(f)

# Trigger transition creation
result = bp.create_transition(
    song_a,
    song_b,
    audio_a_path,
    audio_b_path,
    analysis_a,
    analysis_b,
    idx_a
)

if result:
    print(f"\n✅ SUCCESS: Transition 4 created at {result}")
else:
    print("\n❌ FAILED: Transition 4 creation failed")
