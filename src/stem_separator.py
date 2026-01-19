"""
Stem Separator Module

Separates audio into stems (drums, bass, vocals, other) using demucs.
Optimized for transition segments to prevent heavy instruments from carrying over.
"""
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from pathlib import Path
import hashlib
import json
import warnings
warnings.filterwarnings('ignore')


class StemSeparator:
    """
    Separates audio into stems using demucs library.
    Optimized for short segments (transition regions).
    """
    
    def __init__(self, model_name: str = "htdemucs", device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize stem separator.
        
        Args:
            model_name: demucs model to use ('htdemucs', 'htdemucs_ft', 'mdx_extra')
            device: Device to use ('cuda', 'cpu', or None for auto)
            cache_dir: Directory for caching stems (None = disabled)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._model_loaded = False
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_model(self):
        """Lazy load the demucs model."""
        if self._model_loaded:
            return
            
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            self.get_model = get_model
            self.apply_model = apply_model
            self._model_loaded = True
            print(f"  ✓ Stem separation model ready ({self.model_name})")
        except ImportError:
            raise ImportError(
                "demucs not installed. Install with: pip install demucs"
            )
    
    def separate_stems(self, 
                     audio: np.ndarray, 
                     sr: int = 44100) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems using AI or robust spectral fallbacks.
        """
        # Ensure stereo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        
        try:
            self._load_model()
            
            # Load model (cached after first load)
            if self.model is None:
                self.model = self.get_model(self.model_name)
                self.model.to(self.device)
                self.model.eval()
            
            wav = torch.from_numpy(audio.T).float()
            wav = wav.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                sources = self.apply_model(self.model, wav, shifts=1, split=True, overlap=0.25, progress=False)
            
            sources = sources[0].cpu().numpy()
            
            stems = {
                'drums': sources[0].T,
                'bass': sources[1].T,
                'other': sources[2].T,
                'vocals': sources[3].T
            }
            return self._finalize_stems(stems)
            
        except Exception as e:
            print(f"  ⚠ AI Stem separation failed or not available: {e}")
            print(f"  → Using spectral fallback (HPSS + Filtering)")
            return self._separate_stems_fallback(audio, sr)

    def _separate_stems_fallback(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Robust spectral fallback using HPSS and filtering.
        Isolates Percussive (Drums) and Harmonic (Others), then filters.
        """
        import librosa
        from scipy import signal

        # 1. Harmonic-Percussive Source Separation
        # We process in mono for speed then apply mask to stereo
        audio_mono = np.mean(audio, axis=1)
        harmonic, percussive = librosa.effects.hpss(audio_mono)
        
        # Create masks for stereo
        # Normalize to prevent clipping on recombination
        total = np.abs(harmonic) + np.abs(percussive) + 1e-10
        h_mask = np.abs(harmonic) / total
        p_mask = np.abs(percussive) / total
        
        drums = audio * p_mask[:, np.newaxis]
        harm_stereo = audio * h_mask[:, np.newaxis]

        # 2. Extract Bass from Harmonic using Low-Pass
        nyq = sr / 2
        b, a = signal.butter(4, 250 / nyq, btype='low')
        bass = signal.filtfilt(b, a, harm_stereo, axis=0)
        
        # 3. Extract Vocals/Others from remaining Harmonic
        # Center-cut logic: Vocals are often center-panned
        # Side = (L - R), Mid = (L + R) / 2
        # Vocals are approx Mid - Side
        l, r = harm_stereo[:, 0], harm_stereo[:, 1]
        mid = (l + r) / 2
        side = (l - r) / 2
        
        # Simple vocal estimate: filter Mid for vocal frequencies (300Hz - 8kHz)
        b_v, a_v = signal.butter(4, [300/nyq, 5000/nyq], btype='bandpass')
        vocals_mono = signal.filtfilt(b_v, a_v, mid)
        vocals = np.column_stack([vocals_mono, vocals_mono])
        
        # Clip vocals to avoid exceeding harmonic content
        vocals = np.clip(vocals, -np.abs(harm_stereo), np.abs(harm_stereo))
        
        # 4. Other is what's left
        other = harm_stereo - bass - vocals
        
        stems = {
            'drums': drums,
            'bass': bass,
            'other': other,
            'vocals': vocals
        }
        return self._finalize_stems(stems)

    def _finalize_stems(self, stems: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize stem shapes and stereo format."""
        for key in stems:
            if stems[key].ndim == 1:
                stems[key] = np.column_stack([stems[key], stems[key]])
            elif stems[key].shape[1] == 1:
                stems[key] = np.column_stack([stems[key][:, 0], stems[key][:, 0]])
        return stems
    
    def separate_segment(self, 
                        segment: np.ndarray, 
                        sr: int = 44100,
                        song_id: Optional[str] = None,
                        start_time: Optional[float] = None,
                        duration: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Separate a transition segment (optimized for short segments).
        Supports caching by song_id + segment hash.
        
        Args:
            segment: Audio segment (typically 16 seconds)
            sr: Sample rate
            song_id: Optional song ID for caching
            start_time: Optional start time in song (for cache key)
            duration: Optional duration (for cache key)
            
        Returns:
            Dictionary with separated stems
        """
        # Try cache if enabled
        if self.cache_dir and song_id is not None:
            cached_stems = self._load_cached_stems(song_id, segment, start_time, duration)
            if cached_stems is not None:
                print(f"  ✓ Using cached stems for song {song_id}")
                return cached_stems
        
        # Separate stems
        stems = self.separate_stems(segment, sr)
        
        # Cache if enabled
        if self.cache_dir and song_id is not None:
            self._save_cached_stems(song_id, segment, stems, start_time, duration)
        
        return stems
    
    def _compute_segment_hash(self, segment: np.ndarray) -> str:
        """Compute hash for a segment for cache key."""
        # Use first 1 second for fast hashing
        sample = segment[:min(len(segment), 44100)]
        hash_obj = hashlib.sha256(sample.tobytes())
        return hash_obj.hexdigest()[:16]
    
    def _get_cache_path(self, song_id: str, segment_hash: str) -> Path:
        """Get cache file path for stems."""
        cache_file = self.cache_dir / f"{song_id}_{segment_hash}.npz"
        return cache_file
    
    def _load_cached_stems(self, song_id: str, segment: np.ndarray,
                          start_time: Optional[float], duration: Optional[float]) -> Optional[Dict[str, np.ndarray]]:
        """Load cached stems if available."""
        segment_hash = self._compute_segment_hash(segment)
        cache_path = self._get_cache_path(song_id, segment_hash)
        
        if cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=True)
                stems = {
                    'drums': cached['drums'],
                    'bass': cached['bass'],
                    'vocals': cached['vocals'],
                    'other': cached['other']
                }
                return stems
            except Exception as e:
                # Cache corrupted, delete it
                try:
                    cache_path.unlink()
                except:
                    pass
                return None
        return None
    
    def _save_cached_stems(self, song_id: str, segment: np.ndarray,
                          stems: Dict[str, np.ndarray],
                          start_time: Optional[float], duration: Optional[float]):
        """Save stems to cache."""
        segment_hash = self._compute_segment_hash(segment)
        cache_path = self._get_cache_path(song_id, segment_hash)
        
        try:
            np.savez_compressed(
                cache_path,
                drums=stems['drums'],
                bass=stems['bass'],
                vocals=stems['vocals'],
                other=stems['other'],
                song_id=song_id,
                segment_hash=segment_hash,
                start_time=start_time,
                duration=duration
            )
        except Exception as e:
            # If caching fails, continue without cache
            pass
    
    def recombine_stems(self, 
                       stems: Dict[str, np.ndarray],
                       include_stems: Optional[list] = None) -> np.ndarray:
        """
        Recombine stems into full audio.
        
        Args:
            stems: Dictionary of stems
            include_stems: List of stem names to include (None = all)
            
        Returns:
            Recombined audio array
        """
        if include_stems is None:
            include_stems = list(stems.keys())
        
        result = np.zeros_like(stems[include_stems[0]])
        for stem_name in include_stems:
            if stem_name in stems:
                result += stems[stem_name]
        
        return result
