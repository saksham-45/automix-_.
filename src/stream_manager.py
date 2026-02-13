import threading
import queue
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional
import librosa
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

from src.smart_mixer import SmartMixer
from src.youtube_downloader import download_youtube_audio
from scripts.mix_playlist import fetch_playlist_video_ids

# Constants
TRANSITION_WINDOW = 45.0  # Seconds of audio to grab from ends for mixing
TRANSITION_LENGTH = 16.0  # Target overlap duration
SR = 44100
CACHE_DIR = Path("data/cache/stream")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class StreamSession:
    def __init__(self, playlist_url: str):
        self.id = str(uuid.uuid4())
        self.playlist_url = playlist_url
        self.status = "initializing"  # initializing, playing, finished, error
        self.tracks: List[Dict] = []  # {id, title, duration, path}
        self.playlist_items: List[Dict] = []
        
        # We produce a stream of "Chunks"
        # Each chunk: { "id": str, "type": "main"|"transition", "path": str, "duration": float, "metadata": {} }
        self.chunks_queue = queue.Queue()
        self.processed_chunks = [] # Keep history
        
        self._stop_flag = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        self.mixer = SmartMixer(sr=SR)
        
    def _worker_loop(self):
        try:
            self.status = "fetching_playlist"
            try:
                items = fetch_playlist_video_ids(self.playlist_url)
            except Exception as e:
                print(f"WARN: Playlist fetch failed ({e}). Using fallback demo playlist.")
                items = [
                    {"id": "n8X9_MgEdCg", "title": "NCS: Cradles - Sub Urban"},
                    {"id": "K4DyBUG242c", "title": "NCS: Cartoon - On & On"},
                    {"id": "p7ZsBPK656s", "title": "NCS: Aurora - Lost"}
                ]
            
            self.playlist_items = items
            
            if not items:
                self.status = "error"
                return

            self.status = "processing"
            
            # Initial download of first 2 songs
            # We process pairwise: (Current, Next)
            # 1. Download A
            # 2. Download B
            # 3. Create Chunk: A_Body (0 to End-Window)
            # 4. Create Chunk: Transition (Mix A_Tail + B_Head)
            # 5. Move window: Current = B
            
            # Pre-download first song
            current_item = items[0]
            current_path = self._download_track(current_item)
            if not current_path:
                return # Error handled in download
                
            current_y, _ = librosa.load(current_path, sr=SR)
            current_dur = len(current_y) / SR
            # Start playing first song from the very beginning.
            # For subsequent songs, this start index will be updated based on
            # how much of the head was consumed by the previous transition.
            current_body_start_idx = 0
            window_samples = int(TRANSITION_WINDOW * SR)
            
            # Loop through rest
            for i in range(1, len(items)):
                if self._stop_flag.is_set():
                    break
                    
                
                print(f"DEBUG: Processing next item: {items[i]['title']}", flush=True)
                next_item = items[i]
                print(f"Processing transition {i}: {current_item['title']} -> {next_item['title']}", flush=True)
                # Candidate tail snippet from current song for transition building
                tail_start_idx = int(max(0, len(current_y) - window_samples))
                snippet_a = current_y[tail_start_idx:]  # Last window (or shorter if song ends)

                # Download and prepare next song
                print(f"DEBUG: Downloading {next_item['title']}...", flush=True)
                next_path = self._download_track(next_item)
                if not next_path:
                    # Skip this song if download fails, try next
                    print(f"DEBUG: Download failed for {next_item['title']}, skipping", flush=True)
                    continue
                    
                print(f"DEBUG: Loading audio for {next_item['title']}...", flush=True)
                next_y, _ = librosa.load(next_path, sr=SR)
                next_dur = len(next_y) / SR
                print(f"DEBUG: Loaded {next_item['title']} duration: {next_dur}s", flush=True)

                # Snippet from next song head used for transition
                snippet_b = next_y[:window_samples]
                
                # --- Produce Transition ---
                print(f"DEBUG: Creating transition {current_item['title']} -> {next_item['title']}", flush=True)
                
                # Default markers:
                # - transition starts at start of snippet_a
                # - Song B resumes from start
                a_transition_start_in_snippet = 0
                b_resume_offset_samples = 0
                mixed = None

                # Mix
                try:
                    # Save temp snippets for mixer
                    p_a = CACHE_DIR / "temp_a.wav"
                    p_b = CACHE_DIR / "temp_b.wav"
                    sf.write(p_a, snippet_a, SR)
                    sf.write(p_b, snippet_b, SR)
                    
                    print("DEBUG: Calling create_superhuman_mix...", flush=True)
                    mixed, mix_meta = self.mixer.create_superhuman_mix(
                        str(p_a), str(p_b), 
                        transition_duration=TRANSITION_LENGTH,
                        optimize_quality=False, # Faster
                        return_metadata=True,
                    )
                    print("DEBUG: create_superhuman_mix returned", flush=True)

                    # Where transition starts in Song A snippet
                    try:
                        a_transition_start_in_snippet = int(mix_meta.get("a_transition_start_samples", 0))
                    except Exception:
                        a_transition_start_in_snippet = 0
                    a_transition_start_in_snippet = max(0, min(len(snippet_a), a_transition_start_in_snippet))

                    # Where should Song B resume after the transition?
                    try:
                        b_resume_offset_samples = int(mix_meta.get("b_resume_offset_samples", 0))
                    except Exception:
                        b_resume_offset_samples = 0
                    b_resume_offset_samples = max(0, min(len(next_y), b_resume_offset_samples))
                except Exception as e:
                    print(f"Mixing failed: {e}")
                    # Fallback: play full remaining current song and hard-cut to next song start.
                    a_transition_start_in_snippet = len(snippet_a)
                    b_resume_offset_samples = 0

                # --- Produce "Main Body" of Current Song ---
                # End body exactly where transition content starts in Song A.
                # This avoids duplicate/gap between body and transition.
                body_start_idx = max(0, min(len(current_y), current_body_start_idx))
                body_end_idx = max(0, min(len(current_y), tail_start_idx + a_transition_start_in_snippet))

                print(
                    f"DEBUG: Creating body chunk for {current_item['title']} "
                    f"(start={body_start_idx}, end={body_end_idx}, "
                    f"tail_start={tail_start_idx}, trans_start_in_tail={a_transition_start_in_snippet})",
                    flush=True
                )
                if body_end_idx > body_start_idx:
                    main_body_audio = current_y[body_start_idx:body_end_idx]
                    chunk_path = CACHE_DIR / f"{self.id}_song_{i-1}_body.wav"
                    sf.write(chunk_path, main_body_audio, SR)
                    self.chunks_queue.put({
                        "id": f"{self.id}_{i-1}_body",
                        "type": "main",
                        "path": str(chunk_path),
                        "duration": len(main_body_audio)/SR,
                        "title": current_item['title'] + " (Body)"
                    })
                    print(f"DEBUG: Body chunk created for {current_item['title']}", flush=True)
                else:
                    print(f"DEBUG: Skipping body chunk for {current_item['title']} (start >= end)", flush=True)

                # Transition must be queued AFTER body for correct playback order.
                if mixed is not None:
                    trans_path = CACHE_DIR / f"{self.id}_trans_{i-1}_{i}.wav"
                    sf.write(trans_path, mixed, SR)
                    self.chunks_queue.put({
                        "id": f"{self.id}_trans_{i-1}_{i}",
                        "type": "transition",
                        "path": str(trans_path),
                        "duration": len(mixed)/SR,
                        "title": f"Mix: {current_item['title']} → {next_item['title']}"
                    })

                # Shift
                current_item = next_item
                current_y = next_y
                current_dur = next_dur
                current_path = next_path
                # Update start index for the new "current" song.
                # We resume exactly where the transition finished for Song B.
                current_body_start_idx = min(len(current_y), max(0, b_resume_offset_samples))
                
            # End of playlist: Play rest of last song
            # Start wherever the last transition left Song N.
            start_idx = current_body_start_idx
            if start_idx < len(current_y):
                remain = current_y[start_idx:]
                p = CACHE_DIR / f"{self.id}_final.wav"
                sf.write(p, remain, SR)
                self.chunks_queue.put({
                    "id": f"{self.id}_final",
                    "type": "main",
                    "path": str(p),
                    "duration": len(remain)/SR,
                    "title": current_item['title'] + " (Outro)"
                })
                
            self.chunks_queue.put(None) # Sentinel for "Done"
            self.status = "finished"
            
        except Exception as e:
            print(f"Worker Error: {e}")
            import traceback
            traceback.print_exc()
            self.status = "error"
            
    def _download_track(self, item) -> Optional[str]:
        vid = item['id']
        url = f"https://www.youtube.com/watch?v={vid}"
        path = CACHE_DIR / f"{vid}.wav"
        if path.exists():
            return str(path)
        try:
            download_youtube_audio(url, path, max_duration=0) # 0 = Full video
            return str(path)
        except Exception as e:
            print(f"Download failed for {url}: {e}")
            return None

class StreamManager:
    def __init__(self):
        self.sessions: Dict[str, StreamSession] = {}
        
    def start_session(self, playlist_url: str) -> str:
        s = StreamSession(playlist_url)
        self.sessions[s.id] = s
        return s.id
        
    def get_session(self, sid: str) -> Optional[StreamSession]:
        return self.sessions.get(sid)

# Global Instance
manager = StreamManager()
