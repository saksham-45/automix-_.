#!/usr/bin/env python3
"""
Continuous mixer for processing queues of songs.
Creates seamless transitions: A+B, B+C, C+D, etc.
"""
import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.queue_manager import QueueManager
from src.smart_mixer import SmartMixer
from src.youtube_downloader import download_youtube_audio_parallel, download_youtube_audio
from src.database import MusicDatabase, compute_song_id


class ContinuousMixer:
    """
    Creates continuous mixes from a queue of songs.
    Optimized for Apple Music-level quality with caching.
    """
    
    def __init__(self, 
                 cache_dir: str = "data/cache",
                 db_path: str = "data/music_analysis.db",
                 temp_dir: str = "temp_audio"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        
        # Initialize mixer with cache enabled
        self.mixer = SmartMixer()
        if self.mixer.stem_separation_enabled and self.mixer.stem_separator:
            # Enable stem caching
            self.mixer.stem_separator.cache_dir = str(self.cache_dir / "stems")
        
        # Database for analysis cache
        self.db = MusicDatabase(db_path) if db_path else None
    
    def process_queue(self,
                     queue_path: str = "data/queue.json",
                     output_path: str = None,
                     segment_duration: int = 60,
                     transition_duration: float = 30.0,
                     max_workers: int = 4) -> Path:
        """
        Process entire queue into continuous mix.
        
        Args:
            queue_path: Path to queue JSON file
            output_path: Output path for mix (None = auto-generate)
            segment_duration: Duration to download from each song (seconds)
            transition_duration: Transition duration (seconds)
            max_workers: Parallel download workers
            
        Returns:
            Path to created mix
        """
        # Load queue
        queue_manager = QueueManager(queue_path)
        songs = queue_manager.list_queue()
        
        if len(songs) < 2:
            raise ValueError("Need at least 2 songs for mixing")
        
        print("\n" + "="*60)
        print(f"CONTINUOUS MIX: {len(songs)} songs")
        print("="*60)
        
        # Step 1: Download all songs in parallel
        print(f"\n[Step 1/{len(songs) + 2}] Downloading {len(songs)} songs...")
        audio_paths = self._download_songs(songs, segment_duration, max_workers)
        
        # Step 2: Analyze all songs (check cache, parallel if possible)
        print(f"\n[Step 2/{len(songs) + 2}] Analyzing songs...")
        analyses = self._analyze_songs(audio_paths)
        
        # Step 3: Mix sequentially
        print(f"\n[Step 3/{len(songs) + 2}] Creating continuous mix...")
        mixed_audio = self._create_continuous_mix(
            audio_paths, analyses, transition_duration
        )
        
        # Step 4: Final processing (normalization, smoothing)
        print(f"\n[Step 4/{len(songs) + 2}] Final processing...")
        final_audio = self._finalize_mix(mixed_audio)
        
        # Step 5: Save output
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"continuous_mix_{timestamp}.wav"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[Step 5/{len(songs) + 2}] Saving mix...")
        sf.write(output_path, final_audio, self.mixer.sr)
        
        print(f"\n{'='*60}")
        print(f"✓ CONTINUOUS MIX COMPLETE")
        print(f"{'='*60}")
        print(f"Output: {output_path}")
        print(f"Duration: {len(final_audio) / self.mixer.sr / 60:.1f} minutes")
        print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        return output_path
    
    def _download_songs(self, songs: List[Dict], 
                       segment_duration: int,
                       max_workers: int) -> Dict[int, Path]:
        """Download all songs in parallel."""
        audio_paths = {}
        
        # Prepare download tasks
        download_tasks = []
        for i, song in enumerate(songs):
            song_id = song['id']
            url = song['url']
            source = song['source']
            
            if source == 'youtube':
                # Determine if we need last or first segment
                # First song: last N seconds (outgoing)
                # Last song: first N seconds (incoming)
                # Middle songs: last N seconds for outgoing, first N seconds for incoming
                # For now, download last N for outgoing, first N for incoming
                # We'll download both segments per song in the future for optimization
                # For now, just download what we need per position
                from_end = (i == 0)  # First song: last segment, others: first segment
                
                output_path = self.temp_dir / f"song_{song_id}.wav"
                download_tasks.append((url, output_path, segment_duration, from_end))
                audio_paths[song_id] = output_path
            else:
                # Local file
                audio_paths[song_id] = Path(url)
        
        # Download in parallel
        if download_tasks:
            results = download_youtube_audio_parallel(download_tasks, max_workers)
            # Update paths from results - map back to song_ids
            for i, song in enumerate(songs):
                if song['source'] == 'youtube':
                    url = song['url']
                    song_id = song['id']
                    result_path = results.get(url)
                    if result_path:
                        audio_paths[song_id] = result_path
        
        return audio_paths
    
    def _analyze_songs(self, audio_paths: Dict[int, Path]) -> Dict[int, Dict]:
        """Analyze all songs, using cache when available."""
        analyses = {}
        
        for song_id, audio_path in audio_paths.items():
            if not audio_path.exists():
                continue
            
            # Compute song_id from audio
            song_id_hash = compute_song_id(str(audio_path))
            
            # Check database cache
            cached_analysis = None
            if self.db:
                cached_analysis = self.db.get_song_analysis(song_id_hash)
            
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=self.mixer.sr)
            
            # Analyze (will use cache if available)
            analysis = self.mixer._analyze_song_fast(
                y, 
                existing_analysis=cached_analysis,
                song_id=song_id_hash,
                db_path=str(self.db_path) if self.db else None
            )
            
            analyses[song_id] = analysis
        
        return analyses
    
    def _create_continuous_mix(self,
                              audio_paths: Dict[int, Path],
                              analyses: Dict[int, Dict],
                              transition_duration: float) -> np.ndarray:
        """Create continuous mix by chaining transitions."""
        song_ids = sorted(audio_paths.keys())
        mixed_segments = []
        
        # Cache for reused stems (when song B becomes song A)
        stem_cache = {}
        
        for i in range(len(song_ids) - 1):
            song_a_id = song_ids[i]
            song_b_id = song_ids[i + 1]
            
            print(f"  Mixing {i+1}/{len(song_ids)-1}: Song {song_a_id} → Song {song_b_id}")
            
            song_a_path = audio_paths[song_a_id]
            song_b_path = audio_paths[song_b_id]
            
            if not song_a_path.exists() or not song_b_path.exists():
                print(f"    ⚠ Skipping: files not found")
                continue
            
            analysis_a = analyses.get(song_a_id, {})
            analysis_b = analyses.get(song_b_id, {})
            
            # Mix A+B
            try:
                mixed = self.mixer.create_smooth_mix(
                    str(song_a_path),
                    str(song_b_path),
                    transition_duration=transition_duration,
                    song_a_analysis=analysis_a,
                    song_b_analysis=analysis_b
                )
                
                # Extract transition segment
                # For first transition: include beginning of song A
                # For subsequent transitions: just the transition
                if i == 0:
                    # First transition: include beginning of first song
                    # Take last (transition_duration + segment_duration) seconds
                    segment_start = len(mixed) - int((transition_duration + 60) * self.mixer.sr)
                    segment_start = max(0, segment_start)
                    mixed_segments.append(mixed[segment_start:])
                else:
                    # Subsequent transitions: just the transition part
                    transition_start = len(mixed) - int(transition_duration * self.mixer.sr)
                    transition_start = max(0, transition_start)
                    mixed_segments.append(mixed[transition_start:])
                    
            except Exception as e:
                print(f"    ✗ Mixing failed: {e}")
                import traceback
                traceback.print_exc()
                # Add silence as fallback
                silence_duration = int(transition_duration * self.mixer.sr)
                if mixed_segments:
                    silence = np.zeros((silence_duration, 2))
                    mixed_segments.append(silence)
                continue
        
        # Concatenate all segments
        if mixed_segments:
            continuous_mix = np.concatenate(mixed_segments, axis=0)
            return continuous_mix
        else:
            raise ValueError("No segments to concatenate")
    
    def process_queue_streaming(self,
                               queue_path: str = "data/queue.json",
                               output_path: str = None,
                               segment_duration: int = 60,
                               transition_duration: float = 30.0,
                               buffer_seconds: int = 60) -> Path:
        """
        Process queue with streaming: download segments on-demand.
        
        Key difference from process_queue():
        - Downloads only segments needed (60s outgoing + 60s incoming)
        - Processes transitions while "current" song would be playing
        - Stitches segments together progressively
        
        This enables "real-time" feeling where transitions happen
        while songs are playing (like a live DJ).
        
        Args:
            queue_path: Path to queue JSON file
            output_path: Output path for mix (None = auto-generate)
            segment_duration: Duration to download from each song (seconds)
            transition_duration: Transition duration (seconds)
            buffer_seconds: Buffer size for streaming (not used yet, for future)
            
        Returns:
            Path to created mix
        """
        from src.youtube_streaming import download_segment
        from src.fast_segment_analyzer import FastSegmentAnalyzer
        
        # Load queue
        queue_manager = QueueManager(queue_path)
        songs = queue_manager.list_queue()
        
        if len(songs) < 2:
            raise ValueError("Need at least 2 songs for mixing")
        
        print("\n" + "="*60)
        print(f"STREAMING CONTINUOUS MIX: {len(songs)} songs")
        print("="*60)
        print("Mode: Segment-based (download only what's needed)")
        
        # Initialize fast analyzer for segments
        fast_analyzer = FastSegmentAnalyzer(sample_rate=self.mixer.sr)
        
        all_segments = []
        
        # Process each transition
        for i in range(len(songs) - 1):
            song_a = songs[i]
            song_b = songs[i + 1]
            
            print(f"\n[Transition {i+1}/{len(songs)-1}]")
            print(f"  Song A: {song_a['url'][:50]}...")
            print(f"  Song B: {song_b['url'][:50]}...")
            
            # Download segments (not full songs)
            print("  Downloading segments...")
            seg_a_path = self.temp_dir / f"seg_a_{i}.wav"
            seg_b_path = self.temp_dir / f"seg_b_{i}.wav"
            
            # Download in parallel
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Download outgoing segment (last 60s of song A)
                future_a = executor.submit(
                    download_segment,
                    song_a['url'], seg_a_path,
                    start_time=None,
                    duration=segment_duration,
                    from_end=True
                )
                
                # Download incoming segment (first 60s of song B)
                future_b = executor.submit(
                    download_segment,
                    song_b['url'], seg_b_path,
                    start_time=0,
                    duration=segment_duration,
                    from_end=False
                )
                
                # Wait for both
                seg_a_path = future_a.result()
                seg_b_path = future_b.result()
            
            # Fast analyze segments
            print("  Analyzing segments...")
            analysis_a = fast_analyzer.analyze_segment_file(str(seg_a_path))
            analysis_b = fast_analyzer.analyze_segment_file(str(seg_b_path))
            
            # Create transition using existing SmartMixer
            print("  Creating transition...")
            mixed_transition = self.mixer.create_smooth_mix(
                str(seg_a_path),
                str(seg_b_path),
                transition_duration=transition_duration,
                song_a_analysis=analysis_a,
                song_b_analysis=analysis_b
            )
            
            # Extract transition portion (last N seconds of mixed result)
            if i == 0:
                # First transition: include beginning of first song
                segment_start = len(mixed_transition) - int((transition_duration + segment_duration) * self.mixer.sr)
                segment_start = max(0, segment_start)
                all_segments.append(mixed_transition[segment_start:])
            else:
                # Subsequent transitions: just transition part
                transition_start = len(mixed_transition) - int(transition_duration * self.mixer.sr)
                transition_start = max(0, transition_start)
                all_segments.append(mixed_transition[transition_start:])
        
        # Concatenate all segments
        if all_segments:
            continuous_mix = np.concatenate(all_segments, axis=0)
        else:
            raise ValueError("No segments to concatenate")
        
        # Finalize
        final_audio = self._finalize_mix(continuous_mix)
        
        # Save
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"streaming_mix_{timestamp}.wav"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[Step 5/{len(songs) + 2}] Saving mix...")
        sf.write(output_path, final_audio, self.mixer.sr)
        
        print(f"\n{'='*60}")
        print(f"✓ STREAMING MIX COMPLETE")
        print(f"{'='*60}")
        print(f"Output: {output_path}")
        print(f"Duration: {len(final_audio) / self.mixer.sr / 60:.1f} minutes")
        print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        return output_path
    
    def _finalize_mix(self, mixed_audio: np.ndarray) -> np.ndarray:
        """Final processing: normalization, energy smoothing."""
        # Normalize to -0.3dB peak (avoid clipping)
        peak = np.max(np.abs(mixed_audio))
        if peak > 0:
            target_peak = 10 ** (-0.3 / 20)  # -0.3 dB
            mixed_audio = mixed_audio * (target_peak / peak)
        
        # Energy smoothing across entire mix
        # Apply gentle compression to even out energy fluctuations
        # Simple approach: moving average of RMS energy
        if mixed_audio.ndim == 1:
            mixed_audio = np.column_stack([mixed_audio, mixed_audio])
        
        # Calculate RMS energy over windows
        window_size = int(2.0 * self.mixer.sr)  # 2 second windows
        n_windows = len(mixed_audio) // window_size
        rms_values = []
        
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            segment = mixed_audio[start:end]
            rms = np.sqrt(np.mean(segment ** 2))
            rms_values.append(rms)
        
        if rms_values:
            # Smooth RMS values
            target_rms = np.median(rms_values)
            
            # Apply gentle gain compensation
            for i in range(n_windows):
                start = i * window_size
                end = start + window_size
                current_rms = np.sqrt(np.mean(mixed_audio[start:end] ** 2))
                
                if current_rms > 0:
                    # Gentle gain adjustment (max 3dB)
                    gain_ratio = target_rms / current_rms
                    gain_ratio = np.clip(gain_ratio, 0.7, 1.4)  # ±3dB limit
                    
                    # Apply with fade to avoid clicks
                    fade_samples = int(0.1 * self.mixer.sr)  # 100ms fade
                    fade_in = np.linspace(1.0, gain_ratio, fade_samples)
                    fade_out = np.linspace(gain_ratio, 1.0, fade_samples)
                    
                    if start + fade_samples < len(mixed_audio):
                        mixed_audio[start:start+fade_samples] *= fade_in[:, np.newaxis]
                    if end - fade_samples > 0:
                        mixed_audio[end-fade_samples:end] *= fade_out[:, np.newaxis]
                    
                    if start + fade_samples < end - fade_samples:
                        mixed_audio[start+fade_samples:end-fade_samples] *= gain_ratio
        
        return mixed_audio
