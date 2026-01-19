#!/usr/bin/env python3
"""
Real-Time Streaming Mixer for web-based continuous mixing.

ALIGNMENT WITH PROJECT GOALS:
- Uses triple buffering for seamless transitions
- Supports shuffle (AI) and sequential (playlist) modes
- Generates transitions in real-time while songs play
- Integrates with NextSongSelector for intelligent song selection
- Uses existing SmartMixer for quality transitions
"""
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
import threading

from src.streaming_buffer import StreamingBuffer
from src.next_song_selector import NextSongSelector
from src.smart_mixer import SmartMixer
from src.fast_segment_analyzer import FastSegmentAnalyzer
from src.youtube_streaming import download_segment
from src.database import MusicDatabase
from api.batch_processor import BatchProcessor


class RealTimeStreamingMixer:
    """
    Real-time streaming mixer with triple buffering.
    
    ALIGNMENT CHECK:
    ✓ Uses triple buffering (current/transition/next)
    ✓ Supports shuffle (AI) and sequential (playlist) modes
    ✓ Generates transitions in background while playing
    ✓ Uses existing SmartMixer for quality
    ✓ Fast segment analysis for real-time
    """
    
    def __init__(self,
                 mode: str = 'sequential',
                 transition_duration: float = 16.0,
                 db_path: Optional[str] = None,
                 batch_size: int = 3,
                 cache_dir: str = 'temp_audio/cache'):
        """
        Initialize hybrid streaming mixer with batch processing.
        
        Args:
            mode: 'sequential' or 'shuffle'
            transition_duration: Transition duration (seconds)
            db_path: Path to music database (for caching)
            batch_size: Songs per batch for processing (default: 3)
            cache_dir: Directory for cached songs and transitions
        """
        self.mode = mode
        self.transition_duration = transition_duration
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.sr = 44100
        self.buffer = StreamingBuffer(sr=self.sr)
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            batch_size=batch_size,
            cache_dir=str(cache_dir),
            db_path=db_path,
            sr=self.sr,
            transition_duration=transition_duration
        )
        
        self.next_selector = NextSongSelector() if mode == 'shuffle' else None
        
        # Database for analysis caching
        self.db = MusicDatabase(db_path) if db_path and Path(db_path).exists() else None
        
        # Playlist management
        self.playlist: List[Dict] = []
        self.current_index: int = 0
        self.current_chunk_index: int = 0
        self.played_songs: Set[str] = set()
        self.recent_songs: List[Dict] = []
        
        # Pre-computed transitions and songs (loaded from batch processor)
        self.transitions: Dict[int, np.ndarray] = {}  # transition_index -> audio array
        self.songs: Dict[int, np.ndarray] = {}  # song_index -> full song audio array
        
        # Status tracking
        self.is_playing: bool = False
        self.processing_started: bool = False
        self.first_batch_ready: bool = False
        
        self.lock = threading.Lock()
        
        # Status
        self.status = {
            'mode': mode,
            'current_song': None,
            'next_song': None,
            'total_songs': 0,
            'songs_played': 0,
            'chunks_ready': 0,
            'is_playing': False,
            'processing_status': 'not_started'
        }
    
    def start_playlist(self, songs: List[Dict], on_first_batch_ready: Optional[Callable] = None):
        """
        Start playlist processing with batch processor.
        
        ALIGNMENT: 
        - Processes first batch immediately (to start playback ASAP)
        - Continues processing remaining batches in background
        - Uses pre-computed transitions for fast streaming
        """
        self.playlist = songs
        self.current_index = 0
        self.played_songs = set()
        self.recent_songs = []
        self.transitions = {}
        self.songs = {}
        self.current_chunk_index = 0
        
        self.status['total_songs'] = len(songs)
        self.status['songs_played'] = 0
        self.status['chunks_ready'] = 0
        self.status['processing_status'] = 'processing'
        
        if not songs:
            return
        
        # Update status
        if len(songs) > 0:
            self.status['current_song'] = songs[0].get('title', songs[0].get('url', 'Unknown'))
        if len(songs) > 1:
            self.status['next_song'] = songs[1].get('title', songs[1].get('url', 'Unknown'))
        
        # Start batch processing in background
        def first_batch_callback(batch_result):
            """Called when first batch is ready."""
            with self.lock:
                self.first_batch_ready = True
                self.status['processing_status'] = 'first_batch_ready'
                
                # Load first batch transitions into memory
                # Use get_transition() to ensure 16-second core extraction from old 36s transitions
                for chunk_idx, transition_path in batch_result.get('transitions', {}).items():
                    try:
                        transition_idx = int(chunk_idx)
                        # Use batch_processor.get_transition() which has the extraction fix
                        audio = self.batch_processor.get_transition(transition_idx)
                        if audio is not None:
                            self.transitions[transition_idx] = audio
                    except Exception as e:
                        print(f"  ⚠ Error loading transition {chunk_idx}: {e}")
                
                # Load first batch songs into memory
                # Songs from first batch (indices 0 to batch_size-1)
                batch_size = self.batch_processor.batch_size
                for song_idx in range(min(batch_size, len(self.playlist))):
                    song_audio = self.batch_processor.get_full_song(song_idx)
                    #region agent log
                    import time, json
                    log_path = '/Users/saksham/untitled folder 7/.cursor/debug.log'
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"LOAD","location":"streaming_mixer.py:163","message":"Loading song for first batch","data":{"song_idx":song_idx,"found":song_audio is not None,"completed_songs_count":len(self.batch_processor.completed_songs) if hasattr(self.batch_processor, 'completed_songs') else 0},"timestamp":int(time.time()*1000)}) + '\n')
                    #endregion
                    if song_audio is not None:
                        self.songs[song_idx] = song_audio
                    else:
                        print(f"  ⚠ Song {song_idx} not loaded (not in batch_processor.completed_songs)")
                
                ready_count = self._count_ready_chunks()
                self.status['chunks_ready'] = ready_count
                print(f"  ✓ First batch loaded: {len(self.transitions)} transitions, {len(self.songs)} songs ({ready_count} total chunks ready)")
                
                if on_first_batch_ready:
                    on_first_batch_ready()
        
        # CRITICAL FIX: Restore songs/transitions from cache if they exist
        # Extract mix_id from cache_dir path (format: temp_audio/cache/{mix_id})
        cache_dir_str = str(self.cache_dir)
        mix_id = None
        
        if '/cache/' in cache_dir_str:
            # Extract mix_id from path like "temp_audio/cache/26eb2541"
            parts = cache_dir_str.split('/cache/')
            if len(parts) > 1:
                mix_id = parts[-1].strip('/')
        elif self.cache_dir.name and len(self.cache_dir.name) == 8:
            # Cache dir itself is the mix_id
            mix_id = self.cache_dir.name
        
        if mix_id:
            #region agent log - H3: Cache restoration start
            import time, json
            log_path = '/Users/saksham/untitled folder 7/.cursor/debug.log'
            restore_start = time.time()
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H3","location":"streaming_mixer.py:198","message":"Cache restoration starting","data":{"mix_id":mix_id,"playlist_length":len(songs)},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            print(f"  🔄 Restoring cache for mix {mix_id}...")
            self.batch_processor.restore_from_cache(mix_id, songs)
            
            #region agent log - H3: Cache restoration complete
            restore_time = (time.time() - restore_start) * 1000
            songs_restored = len(self.batch_processor.completed_songs) if hasattr(self.batch_processor, 'completed_songs') else 0
            transitions_restored = len(self.batch_processor.completed_transitions) if hasattr(self.batch_processor, 'completed_transitions') else 0
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H3","location":"streaming_mixer.py:206","message":"Cache restoration complete","data":{"mix_id":mix_id,"restore_time_ms":restore_time,"songs_restored":songs_restored,"transitions_restored":transitions_restored},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            print(f"  ✅ Cache restored: {songs_restored} songs, {transitions_restored} transitions")
            
            # Immediately load available songs into memory (in a separate thread to not block)
            import threading
            def load_songs_thread():
                #region agent log - H5: Background loading thread start
                thread_start = time.time()
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H5","location":"streaming_mixer.py:215","message":"Background loading thread started","data":{"songs_to_load":songs_restored},"timestamp":int(time.time()*1000)}) + '\n')
                #endregion
                
                # CRITICAL FIX: Load songs in priority order (first batch first, then rest)
                # This ensures playback can start immediately
                songs_loaded = 0
                transitions_loaded = 0
                
                # Priority 1: Load first batch (songs 0-2) immediately
                first_batch_size = min(3, len(songs))
                #region agent log - H2, H5: Starting first batch load
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H2,H5","location":"streaming_mixer.py:234","message":"Starting first batch load","data":{"first_batch_size":first_batch_size,"total_songs":len(songs)},"timestamp":int(time.time()*1000)}) + '\n')
                #endregion
                
                for song_idx in range(first_batch_size):
                    #region agent log - H2, H5: Checking song in cache
                    with self.lock:
                        in_cache = song_idx in self.batch_processor.completed_songs
                        already_loaded = song_idx in self.songs
                        completed_songs_keys = list(self.batch_processor.completed_songs.keys())
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H2,H5","location":"streaming_mixer.py:242","message":"Checking song in cache","data":{"song_idx":song_idx,"in_cache":in_cache,"already_loaded":already_loaded,"completed_songs_keys":completed_songs_keys},"timestamp":int(time.time()*1000)}) + '\n')
                    #endregion
                    
                    if in_cache and not already_loaded:
                        load_start = time.time()
                        #region agent log - H2, H5: Calling get_full_song
                        with open(log_path, 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H2,H5","location":"streaming_mixer.py:248","message":"Calling get_full_song","data":{"song_idx":song_idx},"timestamp":int(time.time()*1000)}) + '\n')
                        #endregion
                        song_audio = self.batch_processor.get_full_song(song_idx)
                        if song_audio is not None:
                            with self.lock:
                                self.songs[song_idx] = song_audio
                            songs_loaded += 1
                            #region agent log - H5: Priority song loaded
                            load_time = (time.time() - load_start) * 1000
                            with open(log_path, 'a') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H5","location":"streaming_mixer.py:256","message":"Priority song loaded","data":{"song_idx":song_idx,"duration_sec":len(song_audio)/self.sr,"load_time_ms":load_time,"priority":"first_batch"},"timestamp":int(time.time()*1000)}) + '\n')
                            #endregion
                            print(f"    ✓ [PRIORITY] Loaded song {song_idx} into memory ({len(song_audio)/self.sr:.1f}s)")
                        else:
                            #region agent log - H2, H5: get_full_song returned None
                            with open(log_path, 'a') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H2,H5","location":"streaming_mixer.py:262","message":"get_full_song returned None","data":{"song_idx":song_idx},"timestamp":int(time.time()*1000)}) + '\n')
                            #endregion
                    elif not in_cache:
                        #region agent log - H2, H5: Song not in cache
                        with open(log_path, 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H2,H5","location":"streaming_mixer.py:266","message":"Song not in completed_songs","data":{"song_idx":song_idx},"timestamp":int(time.time()*1000)}) + '\n')
                        #endregion
                    elif already_loaded:
                        #region agent log - H2, H5: Song already loaded
                        with open(log_path, 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H2,H5","location":"streaming_mixer.py:270","message":"Song already in memory","data":{"song_idx":song_idx},"timestamp":int(time.time()*1000)}) + '\n')
                        #endregion
                
                # Priority 2: Load first batch transitions
                for trans_idx in range(first_batch_size - 1):
                    with self.lock:
                        in_cache = trans_idx in self.batch_processor.completed_transitions
                    if in_cache and trans_idx not in self.transitions:
                        trans_data = self.batch_processor.get_transition(trans_idx)
                        if isinstance(trans_data, tuple):
                            trans_audio, trans_meta = trans_data
                        else:
                            trans_audio = trans_data
                            trans_meta = None
                            
                        if trans_audio is not None:
                            with self.lock:
                                self.transitions[trans_idx] = (trans_audio, trans_meta)
                            transitions_loaded += 1
                            print(f"    ✓ [PRIORITY] Loaded transition {trans_idx} into memory ({len(trans_audio)/self.sr:.1f}s)")
                
                # Priority 3: Load remaining songs in background (non-blocking)
                for song_idx in range(first_batch_size, len(songs)):
                    with self.lock:
                        in_cache = song_idx in self.batch_processor.completed_songs
                        already_loaded = song_idx in self.songs
                    if in_cache and not already_loaded:
                        load_start = time.time()
                        song_audio = self.batch_processor.get_full_song(song_idx)
                        if song_audio is not None:
                            with self.lock:
                                self.songs[song_idx] = song_audio
                            songs_loaded += 1
                            #region agent log - H5: Background song loaded
                            load_time = (time.time() - load_start) * 1000
                            with open(log_path, 'a') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H5","location":"streaming_mixer.py:257","message":"Background song loaded","data":{"song_idx":song_idx,"duration_sec":len(song_audio)/self.sr,"load_time_ms":load_time,"songs_loaded_so_far":songs_loaded},"timestamp":int(time.time()*1000)}) + '\n')
                            #endregion
                            print(f"    ✓ Loaded song {song_idx} into memory ({len(song_audio)/self.sr:.1f}s)")
                
                # Load remaining transitions
                for trans_idx in range(first_batch_size - 1, len(songs) - 1):
                    with self.lock:
                        in_cache = trans_idx in self.batch_processor.completed_transitions
                        already_loaded = trans_idx in self.transitions
                    if in_cache and not already_loaded:
                        trans_data = self.batch_processor.get_transition(trans_idx)
                        if isinstance(trans_data, tuple):
                            trans_audio, trans_meta = trans_data
                        else:
                            trans_audio = trans_data
                            trans_meta = None
                            
                        if trans_audio is not None:
                            with self.lock:
                                self.transitions[trans_idx] = (trans_audio, trans_meta)
                            transitions_loaded += 1
                            print(f"    ✓ Loaded transition {trans_idx} into memory ({len(trans_audio)/self.sr:.1f}s)")
                
                # Update ready count
                with self.lock:
                    ready_count = self._count_ready_chunks()
                    self.status['chunks_ready'] = ready_count
                
                #region agent log - H5: Background loading complete
                thread_time = (time.time() - thread_start) * 1000
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H5","location":"streaming_mixer.py:280","message":"Background loading thread complete","data":{"songs_loaded":songs_loaded,"transitions_loaded":transitions_loaded,"ready_chunks":ready_count,"total_time_ms":thread_time},"timestamp":int(time.time()*1000)}) + '\n')
                #endregion
                print(f"  ✅ Ready chunks: {ready_count}/{len(songs)}")
            
            # Start loading in background
            load_thread = threading.Thread(target=load_songs_thread, daemon=True)
            load_thread.start()
            #region agent log - H5: Background thread started
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H5","location":"streaming_mixer.py:288","message":"Background loading thread launched","data":{"thread_name":load_thread.name},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
        
        # Start processing in background thread
        import threading
        processing_thread = threading.Thread(
            target=self._process_playlist_background,
            args=(songs, first_batch_callback),
            daemon=True
        )
        processing_thread.start()
        self.processing_started = True
        
        self.is_playing = True
        self.status['is_playing'] = True
    
    def _process_playlist_background(self, playlist: List[Dict], first_batch_callback: Callable):
        """Process playlist in background (runs in separate thread)."""
        try:
            # Process playlist with batch processor
            results = self.batch_processor.process_playlist(
                playlist,
                on_first_batch_ready=lambda batch_result: first_batch_callback(batch_result)
            )
            
            # Monitor and load songs and transitions as they're created
            import time
            total_songs = len(playlist)
            total_transitions_needed = total_songs - 1
            total_chunks_needed = total_songs  # FIXED: Each chunk is a full song with embedded transition
            
            while True:
                # Load any new songs
                for song_idx in range(total_songs):
                    if song_idx not in self.songs:
                        song_audio = self.batch_processor.get_full_song(song_idx)
                        if song_audio is not None:
                            with self.lock:
                                self.songs[song_idx] = song_audio
                
                # Load any new transitions
                for transition_idx in range(total_transitions_needed):
                    if transition_idx not in self.transitions:
                        transition_audio = self.batch_processor.get_transition(transition_idx)
                        if transition_audio is not None:
                            with self.lock:
                                self.transitions[transition_idx] = transition_audio
                
                # Update ready count
                with self.lock:
                    ready_count = self._count_ready_chunks()
                    self.status['chunks_ready'] = ready_count
                
                # Check if all done
                songs_ready = self.batch_processor.get_ready_songs_count()
                transitions_ready = self.batch_processor.get_ready_transitions_count()
                
                if songs_ready >= total_songs and transitions_ready >= total_transitions_needed:
                    self.status['processing_status'] = 'complete'
                    print(f"  ✓ All content ready: {songs_ready}/{total_songs} songs, {transitions_ready}/{total_transitions_needed} transitions ({ready_count}/{total_chunks_needed} total chunks)")
                    break
                
                time.sleep(2)  # Check every 2 seconds
                
        except Exception as e:
            print(f"  ✗ Background processing error: {e}")
            import traceback
            traceback.print_exc()
            self.status['processing_status'] = 'error'
    
    # Legacy method - not used in hybrid batch processing
    def _create_transition_legacy_unused(self, song_a: Dict, song_b: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Create transition between two songs.
        
        ALIGNMENT: Uses existing SmartMixer for quality transitions.
        """
        try:
            # Download segments (parallel)
            seg_a_path = self.temp_dir / f"seg_a_{self.current_index}.wav"
            seg_b_path = self.temp_dir / f"seg_b_{self.current_index}.wav"
            
            with ThreadPoolExecutor(max_workers=2) as dl_executor:
                # Download outgoing segment (last 60s of song A)
                future_a = dl_executor.submit(
                    download_segment,
                    song_a['url'], seg_a_path,
                    start_time=None,
                    duration=self.segment_duration,
                    from_end=True
                )
                
                # Download incoming segment (first 60s of song B)
                future_b = dl_executor.submit(
                    download_segment,
                    song_b['url'], seg_b_path,
                    start_time=0,
                    duration=self.segment_duration,
                    from_end=False
                )
                
                seg_a_path = future_a.result()
                seg_b_path = future_b.result()
            
            # Fast analyze segments
            analysis_a = self.fast_analyzer.analyze_segment_file(str(seg_a_path))
            analysis_b = self.fast_analyzer.analyze_segment_file(str(seg_b_path))
            
            # Create transition using SmartMixer (QUALITY ASSURED)
            mixed_transition = self.smart_mixer.create_smooth_mix(
                str(seg_a_path),
                str(seg_b_path),
                transition_duration=self.transition_duration,
                song_a_analysis=analysis_a,
                song_b_analysis=analysis_b
            )
            
            # Store current song analysis for next selection (shuffle mode)
            self.current_song_analysis = analysis_b
            
            metadata = {
                'song_a': song_a.get('title', song_a.get('url', 'Unknown')),
                'song_b': song_b.get('title', song_b.get('url', 'Unknown')),
                'transition_duration': self.transition_duration,
                'created_at': datetime.now().isoformat()
            }
            
            # Cleanup temp files
            try:
                seg_a_path.unlink()
                seg_b_path.unlink()
            except:
                pass
            
            return mixed_transition, metadata
            
        except Exception as e:
            print(f"  ⚠ Transition creation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # Legacy method - no longer used (chunks now directly map to songs)
    # Kept for reference but not called
    def _chunk_index_to_content(self, chunk_index: int) -> Tuple[str, int]:
        """
        DEPRECATED: Chunks now directly map to songs with embedded transitions.
        This method is kept for backwards compatibility but should not be used.
        """
        # Direct mapping: chunk_index = song_index
        return ('song', chunk_index)
    
    def get_audio_chunk(self, chunk_index: int) -> Optional[bytes]:
        """
        Get audio chunk for streaming.
        
        NEW STRUCTURE (Full songs with embedded transitions):
        - Each chunk is a full song with transition embedded in second half
        - Chunk 0: Full Song 1 (with transition to Song 2 in second half)
        - Chunk 1: Full Song 2 (with transition to Song 3 in second half)
        - ... continues for entire playlist
        
        Returns WAV bytes for continuous playback.
        
        CRITICAL FIX: Lock is held only for state access, not I/O operations.
        This allows concurrent playback requests and background loading.
        """
        #region agent log
        import time, json
        log_path = '/Users/saksham/untitled folder 7/.cursor/debug.log'
        entry_time = time.time()
        #endregion
        
        # NEW: chunk_index directly maps to song_index (no separate transition chunks)
        song_index = chunk_index
        
        #region agent log - H1, H5: Check memory state (quick lock)
        lock_check_start = time.time()
        with self.lock:
            songs_in_memory = len(self.songs)
            transitions_in_memory = len(self.transitions)
            in_memory = song_index in self.songs
            has_transition = song_index in self.transitions
            cache_songs_count = len(self.batch_processor.completed_songs) if hasattr(self.batch_processor, 'completed_songs') else 0
        lock_check_time = (time.time() - lock_check_start) * 1000
        
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1,H5","location":"streaming_mixer.py:424","message":"get_audio_chunk entry","data":{"chunk_index":chunk_index,"playlist_length":len(self.playlist),"songs_in_memory":songs_in_memory,"transitions_in_memory":transitions_in_memory,"song_in_memory":in_memory,"has_transition":has_transition,"cache_songs_count":cache_songs_count,"lock_check_ms":lock_check_time},"timestamp":int(entry_time*1000)}) + '\n')
        #endregion
        
        # Load song (OUTSIDE lock to avoid blocking)
        #region agent log - H1, H4: Song lookup path
        lookup_start = time.time()
        #endregion
        
        song_audio = None
        if in_memory:
            # Quick memory access (minimal lock time)
            with self.lock:
                song_audio = self.songs.get(song_index)
            #region agent log - H1: Found in memory
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1","location":"streaming_mixer.py:445","message":"Song found in memory","data":{"song_index":song_index,"lookup_time_ms":(time.time()-lookup_start)*1000},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
        else:
            # Try batch processor (may involve file I/O, but no lock held)
            #region agent log - H1: Not in memory, trying batch_processor
            bp_start = time.time()
            #endregion
            song_audio = self.batch_processor.get_full_song(song_index)
            #region agent log - H1: batch_processor result
            bp_time = (time.time() - bp_start) * 1000
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1","location":"streaming_mixer.py:456","message":"batch_processor.get_full_song result","data":{"song_index":song_index,"found":song_audio is not None,"time_ms":bp_time},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            # If found, cache it in memory (quick lock)
            if song_audio is not None:
                with self.lock:
                    self.songs[song_index] = song_audio
            else:
                # Try direct cache lookup as fallback (file I/O, no lock)
                with self.lock:
                    in_cache_dict = song_index in self.batch_processor.completed_songs if hasattr(self.batch_processor, 'completed_songs') else False
                    song_path = self.batch_processor.completed_songs.get(song_index) if in_cache_dict else None
                
                if song_path:
                    #region agent log - H4: Direct cache fallback
                    cache_fallback_start = time.time()
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H4","location":"streaming_mixer.py:471","message":"Trying direct cache fallback","data":{"song_index":song_index,"song_path":str(song_path),"path_exists":song_path.exists() if hasattr(song_path,'exists') else False},"timestamp":int(time.time()*1000)}) + '\n')
                    #endregion
                    try:
                        song_audio, sr = sf.read(str(song_path))
                        # Resample if needed
                        if sr != self.sr:
                            import librosa
                            song_audio = librosa.resample(song_audio, orig_sr=sr, target_sr=self.sr)
                        # Cache in memory (quick lock)
                        with self.lock:
                            self.songs[song_index] = song_audio
                        #region agent log - H4: Cache fallback success
                        cache_time = (time.time() - cache_fallback_start) * 1000
                        with open(log_path, 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H4","location":"streaming_mixer.py:485","message":"Direct cache load success","data":{"song_index":song_index,"duration_sec":len(song_audio)/sr,"load_time_ms":cache_time},"timestamp":int(time.time()*1000)}) + '\n')
                        #endregion
                        print(f"  ✓ Loaded song {song_index} from cache directly")
                    except Exception as e:
                        #region agent log - H4: Cache fallback failed
                        with open(log_path, 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H4","location":"streaming_mixer.py:490","message":"Direct cache load failed","data":{"song_index":song_index,"error":str(e)},"timestamp":int(time.time()*1000)}) + '\n')
                        #endregion
                        print(f"  ⚠ Error loading song {song_index} from cache: {e}")
        
        if song_audio is None:
            #region agent log - H1, H4: Song not found anywhere
            total_lookup_time = (time.time() - lookup_start) * 1000
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1,H4","location":"streaming_mixer.py:497","message":"Song not available","data":{"song_index":song_index,"total_lookup_time_ms":total_lookup_time,"in_memory":in_memory},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            print(f"  ❌ Song {song_index} not available (not in memory, batch_processor, or cache)")
            return None
        
        #region agent log
        song_duration = len(song_audio) / self.sr if song_audio is not None else 0
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"streaming_mixer.py:505","message":"Song loaded","data":{"song_index":song_index,"duration_sec":song_duration,"has_transition":has_transition},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion

        # PHASE 3: Trimming Start of Song (Continuity Fix)
        # We must trim the intro of this song because it was already played 
        # as part of the previous transition (at the end of the previous chunk).
        if song_index > 0:
            start_trim_samples = 0
            prev_trans_idx = song_index - 1
            
            # Try to get previous transition metadata
            prev_meta = None
            
            # Check memory
            with self.lock:
                prev_data_mem = self.transitions.get(prev_trans_idx)
            
            if prev_data_mem and isinstance(prev_data_mem, tuple):
                _, prev_meta = prev_data_mem
            
            # Check batch processor if not in memory
            if not prev_meta:
                prev_data_bp = self.batch_processor.get_transition(prev_trans_idx)
                if isinstance(prev_data_bp, tuple):
                    _, prev_meta = prev_data_bp
            
            if prev_meta and 'transition_point_b' in prev_meta:
                try:
                    point_b = float(prev_meta['transition_point_b'])
                    duration = float(prev_meta['transition_duration'])
                    
                    # The transition covers Song B from 'point_b' to 'point_b + duration'
                    # So we must start playing Song B relative to the original file from:
                    # point_b + duration
                    
                    # Note: point_b is the placement relative to the start of the 60s analysis clip?
                    # No, smart_mixer returns 'transition_point_b' which is Time in Seconds relative to song start?
                    # Let's verify smart_mixer.py:
                    # 'transition_point_b': float(transition_pair.song_b_point.time_sec)
                    # Use StructureAnalyzer time_sec. Typically absolute time in song.
                    # Wait, StructureAnalyzer extracts features.
                    # BUT `smart_mixer` operates on 60s clips!
                    # "song_b_point" was found in the 60s clip.
                    # So 'point_b' is relative to the START of the 60s clip?
                    # BatchProcessor Load: first 60s of Song B.
                    # So Top-Start of Song B = 0.
                    # So point_b is correct absolute time.
                    
                    start_offset_sec = point_b + duration
                    
                    # Apply trim
                    start_trim_samples = int(start_offset_sec * self.sr)
                    
                    if start_trim_samples < len(song_audio):
                        original_len = len(song_audio)
                        song_audio = song_audio[start_trim_samples:]
                        
                        #region agent log
                        with open(log_path, 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"streaming_mixer.py:Trim","message":"Trimmed song start","data":{"song_index":song_index,"point_b":point_b,"duration":duration,"start_offset_sec":start_offset_sec,"trimmed_samples":start_trim_samples,"original_len":original_len},"timestamp":int(time.time()*1000)}) + '\n')
                        #endregion
                    else:
                        print(f"  ⚠ Warning: Trim amount {start_offset_sec}s exceeds song length")
                except Exception as e:
                    print(f"  ⚠ Error calculating trim for song {song_index}: {e}")
            else:
                 # Fallback if no metadata (Legacy/Default)
                 # If we assume 16s transition at start of Song B being 0:00...
                 # We simply trim the transition duration.
                 # Only if we know it happened at 0.
                 # Safe conservative trimming: Default 16s.
                 pass
        
        # Get transition (OUTSIDE lock)
        transition_audio = None
        transition_metadata = None
        
        if self.mode == 'sequential':
            if has_transition:
                with self.lock:
                    trans_data = self.transitions.get(chunk_index)
                
                if isinstance(trans_data, tuple):
                    transition_audio, transition_metadata = trans_data
                else:
                    transition_audio = trans_data
                    transition_metadata = None
            else:
                # Try batch processor (returns tuple now)
                transition_data = self.batch_processor.get_transition(chunk_index)
                if isinstance(transition_data, tuple):
                    transition_audio, transition_metadata = transition_data
                else:
                    transition_audio = transition_data
                    transition_metadata = None
                    
                if transition_audio is not None:
                    with self.lock:
                        self.transitions[chunk_index] = (transition_audio, transition_metadata)
        
        # region agent log - D: Embedding step
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"streaming_mixer.py:525","message":"Embedding transition in song","data":{"song_index":song_index,"song_duration":len(song_audio)/self.sr,"song_samples":len(song_audio),"transition_duration":len(transition_audio)/self.sr if transition_audio is not None else 0,"has_metadata":transition_metadata is not None},"timestamp":int(time.time()*1000)}) + '\n')
        # endregion
        
        # Apply transition embedding if applicable
        audio_data = song_audio
        if transition_audio is not None:
            audio_data = self._embed_transition_in_song(song_audio, transition_audio, song_index, metadata=transition_metadata)
        
        # Convert to WAV bytes (CPU-bound, no lock needed)
        #region agent log - H2: Before WAV conversion
        wav_start = time.time()
        #endregion
        import io
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, audio_data, self.sr, format='WAV')
        wav_bytes.seek(0)
        wav_data = wav_bytes.read()
        #region agent log - H2: After WAV conversion
        wav_time = (time.time() - wav_start) * 1000
        total_time = (time.time() - entry_time) * 1000
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H2","location":"streaming_mixer.py:567","message":"WAV conversion complete","data":{"chunk_index":chunk_index,"wav_size_bytes":len(wav_data),"wav_time_ms":wav_time,"total_time_ms":total_time},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Update status (quick lock)
        with self.lock:
            if chunk_index > self.current_chunk_index:
                self.current_chunk_index = chunk_index
                self.status['songs_played'] = song_index
                if song_index < len(self.playlist):
                    self.status['current_song'] = self.playlist[song_index].get('title', 'Unknown')
                if song_index + 1 < len(self.playlist):
                    self.status['next_song'] = self.playlist[song_index + 1].get('title', 'Unknown')
            
            # Update chunks ready count
            total_chunks = self._get_total_chunks()
            ready_chunks = self._count_ready_chunks()
            self.status['chunks_ready'] = ready_chunks
        
        #region agent log - H2: Complete
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H2","location":"streaming_mixer.py:588","message":"get_audio_chunk complete","data":{"chunk_index":chunk_index,"total_time_ms":total_time,"chunk_size_bytes":len(wav_data)},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        return wav_data
    
    def _embed_transition_in_song(self, song_audio: np.ndarray, transition_audio: np.ndarray, song_index: int, metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Embed transition into the END of a song using professional crossfading.
        """
        try:
            #region agent log
            import time, json
            log_path = '/Users/saksham/untitled folder 7/.cursor/debug.log'
            song_duration_sec = len(song_audio) / self.sr
            transition_duration_sec = len(transition_audio) / self.sr
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"streaming_mixer.py:702","message":"_embed_transition_in_song entry (FIXED VERSION)","data":{"song_index":song_index,"song_duration_sec":song_duration_sec,"transition_duration_sec":transition_duration_sec,"song_samples":len(song_audio),"transition_samples":len(transition_audio),"has_metadata":metadata is not None},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            # PHASE 2: Structure-Aware Placement
            # If metadata (from batch_processor -> smart_mixer) tells us exactly where to start
            # relative to the end, use that!
            if metadata and 'start_time_from_end' in metadata:
                start_time_from_end = float(metadata['start_time_from_end'])
                # Ensure it's reasonable (between 5s and 45s from end)
                start_time_from_end = max(5.0, min(45.0, start_time_from_end))
                transition_start_sec = max(0, song_duration_sec - start_time_from_end)
                placement_source = "metadata"
            else:
                # Fallback: Start 16 seconds before end
                transition_start_sec = max(0, song_duration_sec - transition_duration_sec)
                placement_source = "fallback"
                
            transition_start_sample = int(transition_start_sec * self.sr)
            
            #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"streaming_mixer.py:720","message":"Transition placement (FIXED)","data":{"transition_start_sec":transition_start_sec,"transition_start_sample":transition_start_sample,"placement_source":placement_source,"song_duration":song_duration_sec},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            # Part 1: Song up to the cut point
            # We will CUT from the original song to the transition audio.
            # To avoid clicks, we use a very short micro-crossfade (10ms).
            MICRO_FADE_MS = 10
            micro_fade_samples = int((MICRO_FADE_MS / 1000.0) * self.sr)
            
            # Ensure proper slice indices
            cut_point = transition_start_sample
            
            # Slices for the micro-fade
            # Song fades out just before cut point
            song_pre_cut = song_audio[:cut_point]
            
            # Since we want to crossfade, we need a small overlap.
            # We'll take micro_fade_samples from BEFORE cut_point for fading out
            if len(song_pre_cut) > micro_fade_samples:
                fade_start = len(song_pre_cut) - micro_fade_samples
                song_body = song_pre_cut[:fade_start]
                song_fade_out_part = song_pre_cut[fade_start:]
                
                # Transition fades in at its start
                transition_fade_in_part = transition_audio[:micro_fade_samples]
                transition_body = transition_audio[micro_fade_samples:]
                
                # Ensure dimensions match for the fade
                # (transition_audio is usually stereo 2 channels)
                if len(transition_fade_in_part) == len(song_fade_out_part):
                    # Create equal-power micro-fade
                    t = np.linspace(0, np.pi/2, micro_fade_samples)
                    fade_out = np.cos(t)
                    fade_in = np.sin(t)
                    
                    if song_fade_out_part.ndim > 1:
                        fade_out = fade_out[:, np.newaxis]
                        fade_in = fade_in[:, np.newaxis]
                    
                    # Handle stereo/mono mismatch
                    if transition_fade_in_part.ndim > 1 and song_fade_out_part.ndim == 1:
                        transition_fade_in_part = np.mean(transition_fade_in_part, axis=1)
                    elif transition_fade_in_part.ndim == 1 and song_fade_out_part.ndim > 1:
                        transition_fade_in_part = np.column_stack([transition_fade_in_part, transition_fade_in_part])
                        
                    # Apply fade
                    micro_fade_mix = song_fade_out_part * fade_out + transition_fade_in_part * fade_in
                    
                    # Dimension checks for reconstruction
                    if transition_body.ndim == 1 and song_body.ndim > 1:
                        transition_body = np.column_stack([transition_body, transition_body])
                    if micro_fade_mix.ndim == 1 and song_body.ndim > 1:
                         micro_fade_mix = np.column_stack([micro_fade_mix, micro_fade_mix])

                    # Concatenate: [Song Body] + [Micro Fade] + [Rest of Transition]
                    result = np.concatenate([song_body, micro_fade_mix, transition_body])
                else:
                     # Fallback if overlap mismatch
                     result = np.concatenate([song_pre_cut, transition_audio])
            else:
                 # Fallback if song too short
                 result = np.concatenate([song_pre_cut, transition_audio])
            
            #region agent log
            result_duration_sec = len(result) / self.sr
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"streaming_mixer.py:785","message":"_embed_transition_in_song exit (SWITCH MODE)","data":{"song_index":song_index,"original_duration_sec":song_duration_sec,"result_duration_sec":result_duration_sec,"switch_point_sec":cut_point/self.sr},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            return result
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            try:
                with open('/Users/saksham/untitled folder 7/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ERROR","location":"streaming_mixer.py:800","message":"_embed_transition_in_song CRASHED","data":{"error":str(e),"traceback":error_details},"timestamp":int(time.time()*1000)}) + '\n')
            except:
                pass
            print(f"CRITICAL ERROR in _embed_transition_in_song: {e}")
            # Fallback: return original song to ensure playback continues
            return song_audio
    
    def _get_total_chunks(self) -> int:
        """Get total number of chunks needed for full playlist."""
        # NEW: Each chunk is a full song (with embedded transition), so N chunks for N songs
        N = len(self.playlist)
        return N if N > 0 else 0
    
    def _count_ready_chunks(self) -> int:
        """Count how many chunks are ready."""
        total_chunks = self._get_total_chunks()
        ready = 0
        
        for chunk_idx in range(total_chunks):
            song_index = chunk_idx  # Direct mapping: chunk = song
            
            # Check if song is ready
            song_ready = (song_index in self.songs) or self.batch_processor.is_song_ready(song_index)
            
            # Check if transition is ready (needed for all except last song)
            transition_ready = True
            if song_index + 1 < len(self.playlist):
                transition_ready = (song_index in self.transitions) or self.batch_processor.is_transition_ready(song_index)
            
            if song_ready and transition_ready:
                ready += 1
        
        return ready
    
    # Legacy methods removed - using batch processor instead
    
    def _has_more_songs(self) -> bool:
        """Check if more songs are available."""
        if self.mode == 'sequential':
            return self.current_index + 1 < len(self.playlist)
        else:  # shuffle
            return len(self.played_songs) < len(self.playlist)
    
    def get_next_song(self) -> Optional[Dict]:
        """Get next song info (for shuffle mode)."""
        if self.mode != 'shuffle':
            return None
        
        if not self.current_song_analysis:
            return None
        
        candidates = [s for s in self.playlist if s.get('id') not in self.played_songs]
        
        if not candidates:
            return None
        
        # TODO: Get analysis for candidates
        # For now, return first candidate
        return candidates[0]
    
    def get_next_chunk_info(self) -> Dict:
        """Get info about next chunk (for pre-buffering)."""
        return {
            'chunk_index': self.current_chunk_index + 1,
            'has_more': self._has_more_songs(),
                'next_chunk_url': f'/api/stream/audio?chunk={self.current_chunk_index + 1}'
        }
    
    def get_status(self) -> Dict:
        """Get current mix status."""
        with self.lock:
            total_chunks = self._get_total_chunks()
            ready_chunks = self._count_ready_chunks()
            
            return {
                **self.status,
                'has_more_songs': self._has_more_songs(),
                'chunks_ready': ready_chunks,
                'total_chunks': total_chunks,
                'songs_ready': len(self.songs),
                'transitions_ready': len(self.transitions),
                'current_chunk': self.current_chunk_index,
                'first_batch_ready': self.first_batch_ready,
                'total_transitions_needed': max(0, len(self.playlist) - 1),
                'total_songs_in_playlist': len(self.playlist)
            }
    
    def stop(self):
        """Stop mixing and cleanup."""
        self.is_playing = False
        self.status['is_playing'] = False
        
        self.buffer.clear()
        self.buffer.shutdown()
        
        # Note: batch processor continues in background
        # Cleanup can be done later if needed
