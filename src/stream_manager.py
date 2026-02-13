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
# Keep strict playlist order for streaming stability.
ENABLE_SMART_QUEUE = False

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

    def _build_fallback_transition(self, snippet_a: np.ndarray, snippet_b: np.ndarray):
        """
        Fast fallback transition when AI transition generation fails.
        Returns:
            mixed_fallback, a_transition_start_in_snippet, b_resume_offset_samples
        """
        if len(snippet_a) == 0 or len(snippet_b) == 0:
            return None, len(snippet_a), 0

        overlap_samples = int(TRANSITION_LENGTH * SR)
        overlap_samples = max(1, min(overlap_samples, len(snippet_a), len(snippet_b)))

        # Transition starts this far into snippet_a; body should end exactly here.
        a_transition_start = len(snippet_a) - overlap_samples
        tail_a = snippet_a[a_transition_start:]
        head_b = snippet_b[:overlap_samples]

        fade_out = np.sqrt(np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32))
        fade_in = np.sqrt(np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32))
        mixed = (tail_a * fade_out) + (head_b * fade_in)

        # Song B should resume after the consumed overlap head.
        b_resume_offset = overlap_samples
        return mixed.astype(np.float32), a_transition_start, b_resume_offset
        
    def _worker_loop(self):
        try:
            self.status = "fetching_playlist"
            try:
                items = fetch_playlist_video_ids(self.playlist_url)
            except Exception as e:
                print(f"WARN: Playlist fetch failed ({e}). Using fallback demo playlist.")
                items = [
                    {"id": "n8X9_MgEdCg", "title": "NCS: Cradles - Sub Urban", "url": "https://www.youtube.com/watch?v=n8X9_MgEdCg"},
                    {"id": "K4DyBUG242c", "title": "NCS: Cartoon - On & On", "url": "https://www.youtube.com/watch?v=K4DyBUG242c"},
                    {"id": "p7ZsBPK656s", "title": "NCS: Aurora - Lost", "url": "https://www.youtube.com/watch?v=p7ZsBPK656s"}
                ]
            
            self.playlist_items = items
            
            if not items:
                self.status = "error"
                return

            # Ensure every item has a canonical watch URL (needed by SmartQueue).
            for item in items:
                vid = item.get("id", "")
                if vid and "url" not in item:
                    item["url"] = f"https://www.youtube.com/watch?v={vid}"

            self.status = "processing"
            
            # Initial download of first 2 songs
            # We process pairwise: (Current, Next)
            # 1. Download A
            # 2. Download B
            # 3. Create Chunk: A_Body (0 to End-Window)
            # 4. Create Chunk: Transition (Mix A_Tail + B_Head)
            # 5. Move window: Current = B
            
            # Find first playable song (skip unavailable/deleted/private entries).
            current_item = None
            current_path = None
            start_index = None
            for idx, candidate in enumerate(items):
                current_path = self._download_track(candidate)
                if current_path:
                    current_item = candidate
                    start_index = idx
                    break
                print(f"WARN: Skipping unavailable opening track: {candidate.get('title', candidate.get('id', 'unknown'))}", flush=True)

            if not current_item or not current_path or start_index is None:
                print("ERROR: No playable tracks found in playlist.", flush=True)
                self.status = "error"
                return
                
            current_y, _ = librosa.load(current_path, sr=SR)
            current_dur = len(current_y) / SR
            # Start playing first song from the very beginning.
            # For subsequent songs, this start index will be updated based on
            # how much of the head was consumed by the previous transition.
            current_body_start_idx = 0
            window_samples = int(TRANSITION_WINDOW * SR)
            
            # Optional smart queueing (disabled by default for strict playlist order).
            smart_q = None
            current_meta = None
            if ENABLE_SMART_QUEUE:
                from src.smart_queue import SmartQueue
                smart_q = SmartQueue(CACHE_DIR)
                current_meta = smart_q.get_preview_metadata(current_item['url'], current_item['id'])


            # Loop through rest
            for i in range(start_index + 1, len(items)):
                if self._stop_flag.is_set():
                    break
                # Use Smart Queueing to pick the best next track from the remaining pool
                remaining_pool = items[i:]
                if ENABLE_SMART_QUEUE and smart_q is not None and current_meta is not None and len(remaining_pool) > 1:
                    print(f"DEBUG: SmartQueue selecting best next track from {len(remaining_pool)} candidates...", flush=True)
                    best_idx_in_pool = smart_q.select_best_next(current_meta, remaining_pool)
                    if best_idx_in_pool > 0: # Best is not the next one in original order
                        best_idx_in_all = i + best_idx_in_pool
                        print(f"DEBUG: Reordering playlist! Moving {items[best_idx_in_all]['title']} to next position.", flush=True)
                        # Move best to current position i
                        item = items.pop(best_idx_in_all)
                        items.insert(i, item)

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
                if len(next_y) == 0:
                    print(f"DEBUG: Empty audio loaded for {next_item['title']}, skipping", flush=True)
                    continue
                
                # --- FIX: LCD (Loudness Consistency enforcement) ---
                # Normalize entire track to -14 LUFS to match transition processing
                # This prevents the "blink of eye" volume jump between Body and Transition
                try:
                    from src.psychoacoustics import PsychoacousticAnalyzer
                    analyzer = PsychoacousticAnalyzer(sr=SR)
                    analysis = analyzer.analyze_loudness_lufs(next_y)
                    lufs = analysis.get('integrated_lufs', -23.0)
                    target_lufs = -14.0
                    gain_db = target_lufs - lufs
                    if ENABLE_SMART_QUEUE and smart_q is not None:
                        next_meta = smart_q.get_preview_metadata(next_item['url'], next_item['id'])
                        next_meta['lufs'] = lufs

                    # Clamp gain to avoid noise floor explosion
                    gain_db = max(-10.0, min(10.0, gain_db))
                    gain_lin = 10 ** (gain_db / 20.0)
                    next_y = next_y * gain_lin
                    # Soft clip limit
                    if len(next_y) > 0 and np.max(np.abs(next_y)) > 0.98:
                        next_y = np.tanh(next_y)
                    print(f"DEBUG: Normalized {next_item['title']} to -14 LUFS (Gain: {gain_db:+.2f} dB)", flush=True)
                except Exception as e:
                    print(f"WARN: Failed to normalize audio: {e}")

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
                    # Fallback: deterministic equal-power crossfade so stitching still works.
                    mixed, a_transition_start_in_snippet, b_resume_offset_samples = self._build_fallback_transition(
                        snippet_a, snippet_b
                    )

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
                if ENABLE_SMART_QUEUE and smart_q is not None:
                    current_meta = smart_q.get_preview_metadata(current_item['url'], current_item['id'])

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
            # Optimization: Only download first 180s initially. 
            # This is enough for the transition engine for most tracks.
            # We use max_duration=0 for now because the player needs the full body,
            # but we can fetch it in a separate thread if needed. 
            # Actually, let's stick to max_duration=0 but ensure it doesn't block forever.
            download_youtube_audio(url, path, max_duration=0)
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
