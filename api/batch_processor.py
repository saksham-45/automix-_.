#!/usr/bin/env python3
"""
Batch Processor for hybrid two-phase mixing system.

Processes songs in batches of 3:
1. Downloads full songs in parallel
2. Analyzes songs (full song analysis for better quality)
3. Pre-computes transitions between songs
4. Stores transitions in cache for fast streaming

ALIGNMENT:
- Processes first batch immediately (to start playback ASAP)
- Continues processing remaining batches in background
- Pre-computes all transitions for highest quality
- Uses full song analysis (better than segment-only)
"""
import json
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import hashlib

from src.song_analyzer import SongAnalyzer
from src.smart_mixer import SmartMixer
from src.database import MusicDatabase


class BatchProcessor:
    """
    Processes songs in batches of 3 for hybrid streaming.
    
    Flow:
    1. User uploads playlist
    2. Process first batch (3 songs) → 2 transitions ready
    3. Start playback immediately
    4. Continue processing remaining batches in background
    5. Transitions are pre-computed and cached
    """
    
    def __init__(self,
                 batch_size: int = 3,
                 cache_dir: str = 'temp_audio/cache',
                 db_path: Optional[str] = None,
                 sr: int = 44100,
                 transition_duration: float = 16.0):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of songs to process per batch (default: 3)
            cache_dir: Directory to cache downloaded songs and transitions
            db_path: Path to music database for analysis caching
            sr: Sample rate
            transition_duration: Transition duration in seconds
        """
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.songs_cache_dir = self.cache_dir / 'songs'
        self.transitions_cache_dir = self.cache_dir / 'transitions'
        self.songs_cache_dir.mkdir(parents=True, exist_ok=True)
        self.transitions_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.sr = sr
        self.transition_duration = transition_duration
        
        # Initialize components
        self.song_analyzer = SongAnalyzer(sample_rate=sr)
        self.smart_mixer = SmartMixer(sr=sr)
        self.db = MusicDatabase(db_path) if db_path and Path(db_path).exists() else None
        
        # Processing state
        self.processing_status: Dict[str, Dict] = {}
        self.completed_transitions: Dict[int, Path] = {}  # chunk_index -> transition_path
        self.completed_songs: Dict[int, Path] = {}  # song_index -> song_path (CRITICAL: was missing!)
        self.lock = threading.Lock()
        
        # Progress callbacks
        self.progress_callbacks: List[Callable] = []
    
    def restore_from_cache(self, mix_id: str, playlist: List[Dict]):
        """
        Restore completed_songs and completed_transitions from cache directory.
        
        This is called when server restarts and songs/transitions already exist in cache.
        """
        #region agent log - H1, H3: restore_from_cache entry
        import time, json
        log_path = '/Users/saksham/untitled folder 7/.cursor/debug.log'
        restore_start = time.time()
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1,H3","location":"batch_processor.py:87","message":"restore_from_cache entry","data":{"mix_id":mix_id,"playlist_length":len(playlist),"cache_dir":str(self.cache_dir)},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Handle both formats: cache_dir might be temp_audio/cache or temp_audio/cache/{mix_id}
        if str(mix_id) in str(self.cache_dir):
            mix_cache_dir = Path(self.cache_dir)
        else:
            mix_cache_dir = self.cache_dir / mix_id if self.cache_dir.name != mix_id else self.cache_dir
        
        #region agent log - H1: Cache directory resolution
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1","location":"batch_processor.py:100","message":"Cache directory resolved","data":{"mix_id":mix_id,"mix_cache_dir":str(mix_cache_dir),"exists":mix_cache_dir.exists() if hasattr(mix_cache_dir,'exists') else False},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        songs_dir = mix_cache_dir / 'songs' if (mix_cache_dir / 'songs').exists() else None
        if not songs_dir:
            # Try parent directory
            songs_dir = mix_cache_dir.parent / 'songs' if (mix_cache_dir.parent / 'songs').exists() else None
        
        transitions_dir = mix_cache_dir / 'transitions' if (mix_cache_dir / 'transitions').exists() else None
        if not transitions_dir:
            transitions_dir = mix_cache_dir.parent / 'transitions' if (mix_cache_dir.parent / 'transitions').exists() else None
        
        # Restore songs - match by song ID from playlist
        if songs_dir and songs_dir.exists():
            print(f"  🔍 Checking cache for {len(playlist)} songs...")
            
            songs_restored = 0
            for idx, song in enumerate(playlist):
                song_id = song.get('id')
                if not song_id: continue
                
                # Look for file starting with song_id
                matches = list(songs_dir.glob(f"{song_id}*.wav"))
                if matches:
                    song_file = matches[0]
                    with self.lock:
                        self.completed_songs[idx] = song_file
                    songs_restored += 1
                    
                    #region agent log
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1","location":"batch_processor.py:130","message":"Song matched and restored","data":{"idx":idx,"song_id":song_id,"file":song_file.name},"timestamp":int(time.time()*1000)}) + '\n')
                    #endregion
                    print(f"    ✓ Song {idx} ({song.get('title', 'Unknown')[:30]}): Found")
                
            print(f"  ✅ RESTORED: {songs_restored} songs")
                
        # Restore transitions by index
        # We only restore transitions if the songs adjacent to them were found
        if transitions_dir and transitions_dir.exists():
            print(f"  🔍 Checking for {len(playlist)-1} transitions...")
            transitions_restored = 0
            for idx in range(len(playlist) - 1):
                # Verify that both song A and song B for this transition were found in cache
                if idx in self.completed_songs and (idx + 1) in self.completed_songs:
                    trans_file = transitions_dir / f"transition_{idx}.wav"
                    if trans_file.exists():
                        with self.lock:
                            self.completed_transitions[idx] = trans_file
                        transitions_restored += 1
                        #region agent log
                        with open(log_path, 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1","location":"batch_processor.py:150","message":"Transition restored","data":{"idx":idx},"timestamp":int(time.time()*1000)}) + '\n')
                        #endregion
                        print(f"    ✓ Transition {idx}: Found")
                
            print(f"  ✅ RESTORED: {transitions_restored} transitions")

        #region agent log - H1, H3: restore_from_cache complete
        restore_time = (time.time() - restore_start) * 1000
        final_songs_count = len(self.completed_songs)
        final_transitions_count = len(self.completed_transitions)
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1,H3","location":"batch_processor.py:163","message":"restore_from_cache complete","data":{"mix_id":mix_id,"songs_restored":final_songs_count,"transitions_restored":final_transitions_count,"restore_time_ms":restore_time},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        print(f"  ✅ Cache restored: {final_songs_count} songs, {final_transitions_count} transitions")
    
    def add_progress_callback(self, callback: Callable):
        """Add callback for progress updates."""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, batch_num: int, total_batches: int, status: str, details: Dict = None):
        """Notify progress callbacks."""
        for callback in self.progress_callbacks:
            try:
                callback(batch_num, total_batches, status, details or {})
            except:
                pass
    
    def download_full_song(self, song: Dict, max_retries: int = 3) -> Optional[Path]:
        """
        Download full song from YouTube.
        
        Returns:
            Path to downloaded audio file, or None if failed
        """
        song_id = song.get('id', self._hash_url(song['url']))
        output_path = self.songs_cache_dir / f"{song_id}.wav"
        
        # Check if already cached
        if output_path.exists():
            print(f"    ✓ Song already cached: {song['title'][:50]}")
            return output_path
        
        url = song['url']
        print(f"    Downloading: {song['title'][:50]}")
        
        # Try multiple download strategies
        download_attempts = [
            # Strategy 1: Best audio format
            [
                'yt-dlp',
                '-f', 'bestaudio[ext=m4a]/bestaudio/best',
                '--extract-audio',
                '--audio-format', 'wav',
                '--audio-quality', '0',
                '--no-playlist',
                '--no-warnings',
                '-o', str(output_path).replace('.wav', '.%(ext)s'),
                url
            ],
            # Strategy 2: Any audio
            [
                'yt-dlp',
                '-f', 'bestaudio',
                '--extract-audio',
                '--audio-format', 'wav',
                '--no-playlist',
                '--no-warnings',
                '-o', str(output_path).replace('.wav', '.%(ext)s'),
                url
            ],
            # Strategy 3: Best video (extract audio)
            [
                'yt-dlp',
                '-f', 'best[height<=720]',
                '--extract-audio',
                '--audio-format', 'wav',
                '--no-playlist',
                '--no-warnings',
                '-o', str(output_path).replace('.wav', '.%(ext)s'),
                url
            ]
        ]
        
        for attempt_num, cmd in enumerate(download_attempts, 1):
            try:
                print(f"      Attempt {attempt_num}/{len(download_attempts)}...")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False
                )
                
                # Check if file was created (may have different extension)
                if output_path.exists():
                    return output_path
                
                # Try to find file with different extension
                for ext in ['.m4a', '.opus', '.mp3', '.webm', '.mkv']:
                    candidate = output_path.parent / f"{output_path.stem}{ext}"
                    if candidate.exists():
                        # Convert to WAV
                        convert_cmd = [
                            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                            '-i', str(candidate),
                            '-acodec', 'pcm_s16le',
                            '-ar', str(self.sr),
                            str(output_path)
                        ]
                        subprocess.run(convert_cmd, capture_output=True, check=True)
                        candidate.unlink()  # Cleanup
                        if output_path.exists():
                            return output_path
                
            except subprocess.TimeoutExpired:
                print(f"      ⚠ Timeout on attempt {attempt_num}")
            except Exception as e:
                print(f"      ⚠ Error on attempt {attempt_num}: {e}")
        
        print(f"    ✗ Failed to download: {song['title'][:50]}")
        return None
    
    def analyze_song(self, song: Dict, audio_path: Path) -> Optional[Dict]:
        """
        Analyze a song (full song analysis for better quality).
        
        Returns:
            Analysis dict, or None if failed
        """
        song_id = song.get('id', self._hash_url(song['url']))
        
        # Check database first
        if self.db:
            analysis = self.db.get_song_analysis(song_id)
            if analysis:
                print(f"    ✓ Analysis from DB: {song['title'][:50]}")
                return analysis
        
        try:
            print(f"    Analyzing: {song['title'][:50]}")
            #region agent log
            import time
            log_path = '/Users/saksham/untitled folder 7/.cursor/debug.log'
            analyze_start = time.time()
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"batch_processor.py:211","message":"Starting song analysis","data":{"song_id":song_id,"audio_path":str(audio_path)},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            analysis = self.song_analyzer.analyze(str(audio_path))
            
            #region agent log
            analyze_time = time.time() - analyze_start
            analysis_keys = list(analysis.keys()) if analysis else []
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"batch_processor.py:215","message":"Song analysis complete","data":{"time_sec":analyze_time,"analysis_keys":analysis_keys},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            # Save to database
            if self.db:
                #region agent log
                save_start = time.time()
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"batch_processor.py:217","message":"Saving to database","data":{},"timestamp":int(time.time()*1000)}) + '\n')
                #endregion
                
                analysis_json_path = self.cache_dir / 'analyses' / f"{song_id}.json"
                analysis_json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(analysis_json_path, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
                
                #region agent log
                try:
                    self.db.save_song_analysis(song_id, analysis, str(analysis_json_path))
                    save_time = time.time() - save_start
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"batch_processor.py:223","message":"Database save SUCCESS","data":{"time_sec":save_time},"timestamp":int(time.time()*1000)}) + '\n')
                except Exception as db_err:
                    save_time = time.time() - save_start
                    with open(log_path, 'a') as f:
                        import traceback
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"batch_processor.py:223","message":"Database save FAILED","data":{"time_sec":save_time,"error":str(db_err),"error_type":type(db_err).__name__,"traceback":traceback.format_exc()[:300]},"timestamp":int(time.time()*1000)}) + '\n')
                    # Continue anyway - analysis is still valid
                #endregion
            
            #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"batch_processor.py:232","message":"analyze_song returning analysis","data":{"has_analysis":analysis is not None},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            return analysis
            
        except Exception as e:
            print(f"    ✗ Analysis failed: {e}")
            #region agent log
            import traceback
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"batch_processor.py:238","message":"analyze_song EXCEPTION","data":{"error":str(e),"error_type":type(e).__name__,"traceback":traceback.format_exc()[:500]},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            return None
    
    def create_transition(self,
                         song_a: Dict,
                         song_b: Dict,
                         audio_a_path: Path,
                         audio_b_path: Path,
                         analysis_a: Dict,
                         analysis_b: Dict,
                         transition_index: int) -> Optional[Path]:
        """
        Create and save transition between two songs.
        
        Returns:
            Path to saved transition file, or None if failed
        """
        try:
            print(f"    Creating transition: {song_a['title'][:30]} → {song_b['title'][:30]}")
            
            # Load audio segments (last 60s of song A, first 60s of song B)
            import librosa
            
            # Load full audio for both songs to find optimal transition points globally
            y_a, sr_a = librosa.load(str(audio_a_path), sr=self.sr)
            y_b, sr_b = librosa.load(str(audio_b_path), sr=self.sr)
            
            # Save temporary segments
            seg_a_path = self.cache_dir / f"seg_a_temp_{transition_index}.wav"
            seg_b_path = self.cache_dir / f"seg_b_temp_{transition_index}.wav"
            
            sf.write(str(seg_a_path), y_a, self.sr)
            sf.write(str(seg_b_path), y_b, self.sr)
            
            #region agent log
            import time, json
            log_path = '/Users/saksham/untitled folder 7/.cursor/debug.log'
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"batch_processor.py:265","message":"Before create_smooth_mix","data":{"analysis_a_keys":list(analysis_a.keys()) if analysis_a else [],"analysis_b_keys":list(analysis_b.keys()) if analysis_b else [],"analysis_a_key":analysis_a.get('key') if analysis_a else None,"analysis_a_tempo":analysis_a.get('tempo') if analysis_a else None},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            # Create transition using SmartMixer
            #region agent log
            mix_start = time.time()
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"batch_processor.py:266","message":"Calling create_smooth_mix","data":{"seg_a_path":str(seg_a_path),"seg_b_path":str(seg_b_path)},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            try:
                mixed_transition, metadata = self.smart_mixer.create_smooth_mix(
                    str(seg_a_path),
                    str(seg_b_path),
                    transition_duration=self.transition_duration,
                    song_a_analysis=analysis_a,
                    song_b_analysis=analysis_b
                )
            #region agent log
            except Exception as e:
                with open(log_path, 'a') as f:
                    import traceback
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"batch_processor.py:274","message":"create_smooth_mix FAILED","data":{"error":str(e),"error_type":type(e).__name__,"traceback":traceback.format_exc()[:500]},"timestamp":int(time.time()*1000)}) + '\n')
                raise
            #endregion
            
            #region agent log
            mix_time = time.time() - mix_start
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"batch_processor.py:283","message":"create_smooth_mix SUCCESS","data":{"time_sec":mix_time,"mixed_len":len(mixed_transition) if mixed_transition is not None else 0,"mixed_duration_sec":len(mixed_transition)/self.sr if mixed_transition is not None else 0},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            # CRITICAL FIX: create_smooth_mix returns [10s context A] + [mixed] + [10s context B]
            # We need to extract ONLY the transition core (middle section)
            context_duration = 10.0
            context_samples = int(context_duration * self.sr)
            transition_samples = int(self.transition_duration * self.sr)
            
            # The 'mixed' part returned by executor might be slightly different than transition_duration 
            # if beat matching adjusted it, but SmartMixer uses len(mixed).
            actual_mixed_len = len(mixed_transition) - (context_samples * 2)
            
            if actual_mixed_len > 0:
                transition_core = mixed_transition[context_samples:context_samples + actual_mixed_len]
                mixed_transition = transition_core
            
            # Calculate transition placement logic for full songs
            # point_a is the time in song A where the transition ENDS
            point_a = metadata.get('transition_point_a', 0.0)
            point_b = metadata.get('transition_point_b', 0.0)
            trans_duration = metadata.get('transition_duration', self.transition_duration)
            
            song_a_duration = len(y_a) / self.sr
            
            # transition_start_a is when Song A begins to fade out/transition
            transition_start_a = point_a - trans_duration
            start_time_from_end = song_a_duration - transition_start_a
            
            # Save refined metadata
            metadata_path = self.transitions_cache_dir / f"transition_{transition_index}.json"
            metadata['start_time_from_end'] = float(start_time_from_end)
            metadata['transition_start_a'] = float(transition_start_a)
            metadata['transition_start_b'] = float(point_b)
            metadata['transition_index'] = transition_index
            metadata['song_a_duration'] = float(song_a_duration)
            
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                print(f"    ⚠ Failed to save transition metadata: {e}")
            
            # Save transition (now just the 16-second core)
            transition_path = self.transitions_cache_dir / f"transition_{transition_index}.wav"
            sf.write(str(transition_path), mixed_transition, self.sr)
            
            # Cleanup temp files
            seg_a_path.unlink(missing_ok=True)
            seg_b_path.unlink(missing_ok=True)
            
            print(f"    ✓ Transition created: {transition_path.name}")
            return transition_path
            
        except Exception as e:
            print(f"    ✗ Transition creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_batch(self,
                     batch_songs: List[Dict],
                     batch_num: int,
                     start_index: int,
                     playlist_start_index: int = 0) -> Dict:
        """
        Process a batch of songs.
        
        Args:
            batch_songs: List of song dicts to process
            start_index: Starting index in playlist (for transition numbering)
            playlist_start_index: Starting index in full playlist (for song numbering)
        
        Returns:
            Dict with batch results
        """
        print(f"\n{'='*60}")
        print(f"Processing Batch {batch_num} ({len(batch_songs)} songs)")
        print(f"{'='*60}")
        
        results = {
            'batch_num': batch_num,
            'songs_processed': 0,
            'transitions_created': 0,
            'failed_songs': [],
            'transitions': {}  # transition_index -> transition_path
        }
        
        # Step 1: Download all songs in parallel
        print(f"\n[1/3] Downloading {len(batch_songs)} songs...")
        downloaded_songs = []
        song_indices = []  # Track playlist indices
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.download_full_song, song): (idx, song)
                for idx, song in enumerate(batch_songs)
            }
            
            for future in as_completed(futures):
                batch_idx, song = futures[future]
                try:
                    audio_path = future.result()
                    if audio_path:
                        playlist_song_idx = playlist_start_index + batch_idx
                        downloaded_songs.append((playlist_song_idx, song, audio_path))
                        song_indices.append(playlist_song_idx)
                        results['songs_processed'] += 1
                        # Store song path for full playback (by playlist index)
                        with self.lock:
                            self.completed_songs[playlist_song_idx] = audio_path
                except Exception as e:
                    print(f"    ✗ Download failed for {song['title']}: {e}")
                    results['failed_songs'].append(song['title'])
        
        if len(downloaded_songs) < 2:
            print(f"  ⚠ Not enough songs downloaded for transitions (need at least 2)")
            return results
        
        # Step 2: Analyze songs in parallel
        print(f"\n[2/3] Analyzing {len(downloaded_songs)} songs...")
        analyzed_songs = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.analyze_song, song, audio_path): (playlist_idx, song, audio_path)
                for playlist_idx, song, audio_path in downloaded_songs
            }
            
            for future in as_completed(futures):
                playlist_idx, song, audio_path = futures[future]
                try:
                    analysis = future.result()
                    if analysis:
                        analyzed_songs.append((playlist_idx, song, audio_path, analysis))
                except Exception as e:
                    print(f"    ✗ Analysis failed for {song['title']}: {e}")
        
        if len(analyzed_songs) < 2:
            print(f"  ⚠ Not enough songs analyzed for transitions")
            return results
        
        # Sort analyzed songs by playlist index to ensure correct transition order
        # ALIGNMENT: Analysis order is random (as_completed), we must restore playlist order.
        analyzed_songs.sort(key=lambda x: x[0])
        
        # Step 3: Create transitions (sequential for now, could parallelize)
        print(f"\n[3/3] Creating {len(analyzed_songs) - 1} transitions...")
        for i in range(len(analyzed_songs) - 1):
            playlist_idx_a, song_a, audio_a, analysis_a = analyzed_songs[i]
            playlist_idx_b, song_b, audio_b, analysis_b = analyzed_songs[i + 1]
            
            # Skip if already completed (cached)
            if playlist_idx_a in self.completed_transitions:
                print(f"    ✓ Transition {playlist_idx_a} already cached.")
                continue
            
            # Transition index is the index of the first song (transitions connect i to i+1)
            transition_index = playlist_idx_a
            transition_path = self.create_transition(
                song_a, song_b,
                audio_a, audio_b,
                analysis_a, analysis_b,
                transition_index
            )
            
            if transition_path:
                results['transitions'][transition_index] = str(transition_path)
                results['transitions_created'] += 1
                
                # Update completed transitions
                with self.lock:
                    self.completed_transitions[transition_index] = transition_path
        
        print(f"\n✓ Batch {batch_num} complete: {results['transitions_created']} transitions created")
        return results
    
    def process_playlist(self,
                        playlist: List[Dict],
                        on_first_batch_ready: Optional[Callable] = None) -> Dict:
        """
        Process entire playlist in batches.
        
        Args:
            playlist: List of song dicts
            on_first_batch_ready: Callback when first batch is ready (for immediate playback)
        
        Returns:
            Dict with processing results
        """
        total_songs = len(playlist)
        total_batches = (total_songs + self.batch_size - 1) // self.batch_size
        
        print(f"\n{'='*70}")
        print(f"PROCESSING PLAYLIST")
        print(f"{'='*70}")
        print(f"Total songs: {total_songs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Total batches: {total_batches}")
        print(f"{'='*70}\n")
        
        all_results = {
            'total_songs': total_songs,
            'total_batches': total_batches,
            'batches': [],
            'first_batch_ready': False
        }
        
        # Process first batch immediately
        first_batch = playlist[:self.batch_size]
        print(f"⚡ Processing FIRST BATCH (to start playback ASAP)...")
        first_batch_result = self.process_batch(first_batch, batch_num=1, start_index=0, playlist_start_index=0)
        all_results['batches'].append(first_batch_result)
        
        # Notify first batch ready
        if on_first_batch_ready and first_batch_result['transitions_created'] > 0:
            all_results['first_batch_ready'] = True
            print(f"\n✓✓✓ FIRST BATCH READY - Playback can start! ✓✓✓")
            on_first_batch_ready(first_batch_result)
        
        # Process remaining batches in background
        if total_batches > 1:
            print(f"\n📦 Processing remaining {total_batches - 1} batches in background...")
            
            for batch_num in range(2, total_batches + 1):
                # BRIDGE FIX: Each batch starts from the last song of the previous batch
                # to ensure the transition between batches is created.
                start_idx = (batch_num - 1) * self.batch_size - 1
                end_idx = min(start_idx + self.batch_size + 1, total_songs)
                batch_songs = playlist[start_idx:end_idx]
                
                batch_result = self.process_batch(
                    batch_songs,
                    batch_num=batch_num,
                    start_index=start_idx,
                    playlist_start_index=start_idx
                )
                all_results['batches'].append(batch_result)
                
                # Notify progress
                self._notify_progress(
                    batch_num=batch_num,
                    total_batches=total_batches,
                    status='completed',
                    details=batch_result
                )
        
        print(f"\n{'='*70}")
        print(f"✓ PLAYLIST PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total transitions created: {sum(b['transitions_created'] for b in all_results['batches'])}")
        
        return all_results
    
    def get_transition(self, chunk_index: int) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Get pre-computed transition by chunk index.
        
        Returns:
            Tuple of (Audio array, Metadata dict)
            Audio: 16-second transition core, or None if not ready
            Metadata: Transition placement info, or None
        """
        with self.lock:
            if chunk_index in self.completed_transitions:
                transition_path = self.completed_transitions[chunk_index]
                try:
                    audio, sr = sf.read(str(transition_path))
                    
                    # Try to load metadata
                    metadata = None
                    try:
                        metadata_path = transition_path.with_suffix('.json')
                        if metadata_path.exists():
                            import json
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                    except Exception as e:
                        print(f"  ⚠ Error loading transition metadata {chunk_index}: {e}")
                    
                    # CRITICAL FIX: Extract 16-second core if transition is 36 seconds (old format)
                    expected_duration = self.transition_duration  # 16 seconds
                    actual_duration = len(audio) / sr
                    
                    if abs(actual_duration - 36.0) < 1.0:  # Old format (36 seconds)
                        # Extract 16-second core from middle
                        context_samples = int(10.0 * sr)  # Skip first 10 seconds
                        core_samples = int(expected_duration * sr)  # Take 16 seconds
                        
                        if len(audio) > context_samples + core_samples:
                            audio = audio[context_samples:context_samples + core_samples]
                    
                    return audio, metadata
                except Exception as e:
                    print(f"  ⚠ Error loading transition {chunk_index}: {e}")
                    return None, None
            return None, None
    
    def is_transition_ready(self, chunk_index: int) -> bool:
        """Check if transition is ready."""
        with self.lock:
            return chunk_index in self.completed_transitions
    
    def get_ready_transitions_count(self) -> int:
        """Get count of ready transitions."""
        with self.lock:
            return len(self.completed_transitions)
    
    def get_full_song(self, song_index: int) -> Optional[np.ndarray]:
        """
        Get full song by index in playlist.
        
        Returns:
            Audio array, or None if not ready yet
        """
        #region agent log - H1: get_full_song entry
        import time, json
        log_path = '/Users/saksham/untitled folder 7/.cursor/debug.log'
        bp_entry_time = time.time()
        #endregion
        
        bp_lock_start = time.time()
        with self.lock:
            bp_lock_time = (time.time() - bp_lock_start) * 1000
            #region agent log - H1, H2: Inside batch_processor lock
            in_dict = song_index in self.completed_songs
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1,H2","location":"batch_processor.py:675","message":"get_full_song lock acquired","data":{"song_index":song_index,"in_completed_songs":in_dict,"lock_wait_ms":bp_lock_time,"completed_songs_keys":list(self.completed_songs.keys())[:10]},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            if song_index in self.completed_songs:
                song_path = self.completed_songs[song_index]
                #region agent log - H1: File path found, reading
                read_start = time.time()
                path_exists = song_path.exists() if hasattr(song_path, 'exists') else False
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1","location":"batch_processor.py:682","message":"Reading song file","data":{"song_index":song_index,"song_path":str(song_path),"path_exists":path_exists},"timestamp":int(time.time()*1000)}) + '\n')
                #endregion
                try:
                    audio, sr = sf.read(str(song_path))
                    #region agent log - H1: File read success
                    read_time = (time.time() - read_start) * 1000
                    total_time = (time.time() - bp_entry_time) * 1000
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1","location":"batch_processor.py:687","message":"Song file read success","data":{"song_index":song_index,"duration_sec":len(audio)/sr,"read_time_ms":read_time,"total_time_ms":total_time},"timestamp":int(time.time()*1000)}) + '\n')
                    #endregion
                    return audio
                except Exception as e:
                    #region agent log - H1: File read failed
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1","location":"batch_processor.py:692","message":"Song file read failed","data":{"song_index":song_index,"error":str(e)},"timestamp":int(time.time()*1000)}) + '\n')
                    #endregion
                    print(f"  ⚠ Error loading song {song_index}: {e}")
                    return None
            #region agent log - H1: Song not in completed_songs
            else:
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H1","location":"batch_processor.py:698","message":"Song not in completed_songs","data":{"song_index":song_index,"completed_count":len(self.completed_songs)},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            return None
    
    def is_song_ready(self, song_index: int) -> bool:
        """Check if song is ready."""
        with self.lock:
            return song_index in self.completed_songs
    
    def get_ready_songs_count(self) -> int:
        """Get count of ready songs."""
        with self.lock:
            return len(self.completed_songs)
    
    def _hash_url(self, url: str) -> str:
        """Generate hash from URL."""
        return hashlib.md5(url.encode()).hexdigest()[:16]
    
    def cleanup_cache(self, keep_recent: int = 10):
        """Cleanup old cached files (keep most recent N transitions)."""
        # TODO: Implement cleanup logic
        pass
