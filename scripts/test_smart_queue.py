import sys
import os
sys.path.insert(0, os.getcwd())
from src.smart_queue import SmartQueue
from pathlib import Path

def test_smart_queue():
    cache_dir = "cache/test_previews"
    sq = SmartQueue(cache_dir)
    
    # Test with a known NCS song (different URL)
    url = "https://www.youtube.com/watch?v=k4DyBUG242c"
    # Wait, let's use a different one
    url = "https://www.youtube.com/watch?v=7uB6uG_S9Xg" 
    track_id = "test_song_1"
    
    print(f"Testing metadata extraction for {url}...")
    meta = sq.get_preview_metadata(url, track_id)
    print(f"Resulting Metadata: {meta}")
    
    # Test compatibility scoring
    meta_a = {"bpm": 128, "key": "8A", "energy": 0.7}
    meta_b = {"bpm": 126, "key": "9A", "energy": 0.65}
    score = sq.score_compatibility(meta_a, meta_b)
    print(f"Compatibility Score (128/8A vs 126/9A): {score:.2f}")

if __name__ == "__main__":
    test_smart_queue()
