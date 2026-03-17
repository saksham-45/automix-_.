import os
import subprocess
import librosa
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

class SmartQueue:
    """
    Intelligently reorders a playlist based on musical compatibility (BPM, Key)
    using fast 30-second audio previews.
    """
    
    def __init__(self, cache_dir: str, sr: int = 22050):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sr = sr
        self.metadata_cache = {} # id -> {bpm, key, energy}

    def get_preview_metadata(self, track_url: str, track_id: str) -> Dict:
        """Downloads 30s preview and extracts musical metadata."""
        if track_id in self.metadata_cache:
            return self.metadata_cache[track_id]
            
        preview_path = self.cache_dir / f"preview_{track_id}.mp3"
        
        # 1. Faster download: only first 30 seconds
        if not preview_path.exists():
            print(f"DEBUG: Downloading 30s preview for {track_id}...")
            cmd = [
                "yt-dlp",
                "--quiet",
                "--no-warnings",
                "--extract-audio",
                "--audio-format", "mp3",
                "--download-sections", "*0-30",
                "-o", str(preview_path),
                track_url
            ]
            try:
                subprocess.run(cmd, check=True, timeout=60)
            except Exception as e:
                print(f"ERROR: Preview download failed for {track_id}: {e}")
                return {"bpm": 120, "key": "8A", "energy": 0.5, "error": True}

        # 2. Fast Analysis
        if not preview_path.exists():
            return {"bpm": 120, "key": "8A", "energy": 0.5, "error": True}

        try:
            y, _ = librosa.load(str(preview_path), sr=self.sr, duration=30)
            if len(y) == 0:
                raise ValueError("Empty audio loaded")
            
            # BPM
            tempo, _ = librosa.beat.beat_track(y=y, sr=self.sr)
            # Handle both array and scalar tempo from different librosa versions
            bpm = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
            
            # Key (Chromagram based)
            chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
            if chroma.size == 0:
                raise ValueError("Empty chromagram")
            
            mean_chroma = np.mean(chroma, axis=1)
            # Simplistic key detection (12 bins)
            key_idx = np.argmax(mean_chroma)
            # Convert to Camelot (simplistic mapping for demo)
            camelot_map = {
                0: '1B', 1: '8B', 2: '3B', 3: '10B', 4: '5B', 5: '12B',
                6: '7B', 7: '2B', 8: '9B', 9: '4B', 10: '11B', 11: '6B'
            }
            key = camelot_map.get(key_idx, '8A')
            
            # Energy (RMS)
            energy = float(np.sqrt(np.mean(y**2)))
            # Normalize energy loosely
            energy = min(1.0, energy * 5.0) 

            meta = {"bpm": bpm, "key": key, "energy": energy}
            self.metadata_cache[track_id] = meta
            return meta
            
        except Exception as e:
            print(f"ERROR: Analysis failed for {track_id}: {e}")
            return {"bpm": 120, "key": "8A", "energy": 0.5}

    def score_compatibility(self, meta_a: Dict, meta_b: Dict) -> float:
        """Scores compatibility between two tracks (0 to 1)."""
        # 1. BPM Score (closer is better, within 8% range)
        bpm_diff = abs(meta_a['bpm'] - meta_b['bpm'])
        bpm_score = max(0, 1.0 - (bpm_diff / (meta_a['bpm'] * 0.08)))
        
        # 2. Key Score (Camelot Wheel logic)
        key_a = meta_a['key']
        key_b = meta_b['key']
        
        num_a = int(key_a[:-1])
        mode_a = key_a[-1]
        num_b = int(key_b[:-1])
        mode_b = key_b[-1]
        
        # Harmonic keys: same, +/- 1, or relative major/minor
        key_score = 0.0
        if key_a == key_b:
            key_score = 1.0
        elif mode_a == mode_b and abs(num_a - num_b) % 12 == 1:
            key_score = 0.8
        elif num_a == num_b and mode_a != mode_b:
            key_score = 0.6
        else:
            key_score = 0.2
            
        # 3. Energy Flow (prefer similar or slightly increasing)
        energy_score = 1.0 - abs(meta_a['energy'] - meta_b['energy'])
        
        return (bpm_score * 0.4) + (key_score * 0.4) + (energy_score * 0.2)

    def select_best_next(self, current_meta: Dict, candidates: List[Dict]) -> int:
        """
        Given current track meta and a list of candidate items (with ids and urls),
        returns the index of the best next track.
        """
        best_score = -1
        best_idx = 0
        
        # Limit candidates for speed
        candidates = candidates[:5] 
        
        for i, item in enumerate(candidates):
            meta_b = self.get_preview_metadata(item['url'], item['id'])
            score = self.score_compatibility(current_meta, meta_b)
            
            # Add bias for original order to prevent wild shuffling
            score += (1.0 - (i / len(candidates))) * 0.1
            
            print(f"DEBUG: Candidate {item.get('title')} Score: {score:.2f} (BPM: {meta_b['bpm']:.1f}, Key: {meta_b['key']})")
            
            if score > best_score:
                best_score = score
                best_idx = i
                
        return best_idx
