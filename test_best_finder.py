import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from api.batch_processor import BatchProcessor
import json
import os
import shutil

# Setup fresh env
CACHE_DIR = Path('temp_audio/cache/fresh_test')
if CACHE_DIR.exists():
    shutil.rmtree(CACHE_DIR)
CACHE_DIR.mkdir(parents=True)

bp = BatchProcessor(cache_dir=str(CACHE_DIR))

# Define two random/popular songs
test_songs = [
    {
        "id": "weeknd-blinding",
        "url": "https://www.youtube.com/watch?v=4NRXx6U8ABQ",
        "title": "The Weeknd - Blinding Lights"
    },
    {
        "id": "weeknd-starboy",
        "url": "https://www.youtube.com/watch?v=34Na4j8AVgA",
        "title": "The Weeknd - Starboy"
    }
]

print("\n🚀 STARTING FRESH TRANSITION TEST")
print("Target: Find optimal transition point in 2nd half of A and into good section of B")
print("-" * 60)

# 1. Download
print("\n[1/3] Downloading songs...")
audio_paths = []
for song in test_songs:
    path = bp.download_full_song(song)
    if not path:
        print(f"❌ Failed to download {song['title']}")
        sys.exit(1)
    audio_paths.append(path)

# 2. Analyze
print("\n[2/3] Analyzing songs (Full song profiling)...")
analyses = []
for i, path in enumerate(audio_paths):
    analysis = bp.analyze_song(test_songs[i], path)
    if not analysis:
        print(f"❌ Failed to analyze {test_songs[i]['title']}")
        sys.exit(1)
    analyses.append(analysis)

# 3. Create Transition
print("\n[3/3] Finding optimal points and creating mix...")
transition_path = bp.create_transition(
    test_songs[0], test_songs[1],
    audio_paths[0], audio_paths[1],
    analyses[0], analyses[1],
    0 # Index
)

if transition_path:
    # Get metadata from JSON
    metadata_path = transition_path.with_suffix('.json')
    with open(metadata_path) as f:
        meta = json.load(f)
    
    print("\n✅ TRANSITION CREATED!")
    print(f"File: {transition_path}")
    print("-" * 20)
    print(f"Song A Point: {meta['transition_point_a']:.2f}s (Should be in 2nd half)")
    print(f"Song B Point: {meta['transition_point_b']:.2f}s")
    print(f"Technique: {meta['technique']}")
    print("-" * 20)
    
    # Generate a demo snippet for playback
    from generate_demo_mix import DemoMixer
    demo = DemoMixer(MIX_ID="fresh_test")
    # We need to hack the DemoMixer a bit since it's designed for batch, 
    # but we'll just stitch these two
    output_name = "fresh_optimal_test.wav"
    
    # Save a final wav to Desktop
    final_output = Path.home() / "Desktop" / output_name
    
    # For simplicity, let's just copy the transition core for a quick listen
    shutil.copy(transition_path, final_output)
    print(f"\n📂 Final mix snippet saved to Desktop: {final_output}")
    print("Run: open ~/Desktop/fresh_optimal_test.wav")
else:
    print("\n❌ Transition creation failed.")
