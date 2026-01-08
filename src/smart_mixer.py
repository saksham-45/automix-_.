"""
Smart Mixer

Creates smooth transitions using found optimal transition points and beat alignment.
"""
import numpy as np
import soundfile as sf
import librosa
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Optional, Tuple
from pathlib import Path

from src.smart_transition_finder import SmartTransitionFinder, TransitionPair
from src.beat_aligner import BeatAligner


class SmartMixer:
    """
    Creates smooth, beat-matched transitions between songs.
    """
    
    def __init__(self, sr: int = 44100, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length
        self.transition_finder = SmartTransitionFinder(sr=sr, hop_length=hop_length)
        self.beat_aligner = BeatAligner(sr=sr, hop_length=hop_length)
    
    def create_smooth_mix(self,
                         song_a_path: str,
                         song_b_path: str,
                         transition_duration: float = 16.0,
                         song_a_analysis: Optional[Dict] = None,
                         song_b_analysis: Optional[Dict] = None,
                         ai_transition_data: Optional[Dict] = None) -> np.ndarray:
        """
        Create a smooth mix between two songs using optimal transition points.
        
        Args:
            song_a_path: Path to outgoing song
            song_b_path: Path to incoming song
            transition_duration: Duration of transition in seconds
            song_a_analysis: Pre-computed analysis (optional)
            song_b_analysis: Pre-computed analysis (optional)
            ai_transition_data: AI model prediction data (optional)
        
        Returns:
            Mixed audio array (stereo)
        """
        # Find optimal transition points
        transition_pair = self.transition_finder.find_best_transition_pair(
            song_a_path, song_b_path,
            song_a_analysis, song_b_analysis
        )
        
        # Load audio
        y_a, sr_a = librosa.load(song_a_path, sr=self.sr)
        y_b, sr_b = librosa.load(song_b_path, sr=self.sr)
        
        # Beat-align the transition points
        aligned_a, aligned_b = self.beat_aligner.align_beats(
            y_a, y_b, self.sr,
            transition_pair.song_a_point.time_sec,
            transition_pair.song_b_point.time_sec
        )
        
        print(f"\nBeat-aligned transition:")
        print(f"  Song A: {aligned_a:.2f}s → {aligned_a + transition_duration:.2f}s")
        print(f"  Song B: {aligned_b:.2f}s → {aligned_b + transition_duration:.2f}s")
        
        # Ensure stereo
        if y_a.ndim == 1:
            y_a = np.column_stack([y_a, y_a])
        if y_b.ndim == 1:
            y_b = np.column_stack([y_b, y_b])
        
        # Convert to samples
        aligned_a_samples = int(aligned_a * self.sr)
        aligned_b_samples = int(aligned_b * self.sr)
        transition_samples = int(transition_duration * self.sr)
        
        # Extract segments for transition
        seg_a_start = max(0, aligned_a_samples)
        seg_a_end = min(len(y_a), aligned_a_samples + transition_samples)
        seg_a = y_a[seg_a_start:seg_a_end]
        
        seg_b_start = max(0, aligned_b_samples)
        seg_b_end = min(len(y_b), aligned_b_samples + transition_samples)
        seg_b = y_b[seg_b_start:seg_b_end]
        
        # Ensure same length
        min_len = min(len(seg_a), len(seg_b))
        seg_a = seg_a[:min_len]
        seg_b = seg_b[:min_len]
        
        # Pad if needed
        if len(seg_a) < transition_samples:
            pad = transition_samples - len(seg_a)
            seg_a = np.pad(seg_a, ((0, pad), (0, 0)), mode='constant')
        if len(seg_b) < transition_samples:
            pad = transition_samples - len(seg_b)
            seg_b = np.pad(seg_b, ((0, pad), (0, 0)), mode='constant')
        
        # Apply smooth volume curves
        if ai_transition_data and 'curves' in ai_transition_data:
            curves = ai_transition_data['curves']
            seg_a, seg_b = self._apply_ai_curves(seg_a, seg_b, curves)
        else:
            # Fallback: smooth linear crossfade with easing
            seg_a, seg_b = self._apply_smooth_crossfade(seg_a, seg_b)
        
        # Mix overlapping segments
        mixed = seg_a + seg_b
        
        # Normalize
        max_val = np.max(np.abs(mixed))
        if max_val > 0.95:
            mixed = mixed * (0.95 / max_val)
        
        # Build final mix with context
        context_before = int(10 * self.sr)  # 10 seconds before
        context_after = int(10 * self.sr)   # 10 seconds after
        
        ctx_a_start = max(0, seg_a_start - context_before)
        ctx_a = y_a[ctx_a_start:seg_a_start]
        
        ctx_b_end = min(len(y_b), seg_b_end + context_after)
        ctx_b = y_b[seg_b_end:ctx_b_end]
        
        # Concatenate: context A → transition → context B
        final = np.concatenate([ctx_a, mixed, ctx_b], axis=0)
        
        return final
    
    def _apply_ai_curves(self, seg_a: np.ndarray, seg_b: np.ndarray, curves: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Apply AI-generated volume curves with smoothing."""
        vol_a = np.array(curves.get('volume_a', []))
        vol_b = np.array(curves.get('volume_b', []))
        
        if len(vol_a) == 0 or len(vol_b) == 0:
            return self._apply_smooth_crossfade(seg_a, seg_b)
        
        # Resample curves to match segment length
        idx = np.linspace(0, len(vol_a) - 1, len(seg_a))
        vol_a = np.interp(idx, np.arange(len(vol_a)), vol_a)
        vol_b = np.interp(idx, np.arange(len(vol_b)), vol_b)
        
        # Smooth curves for gradual transitions
        sigma_a = min(500, len(vol_a) / 20)
        sigma_b = min(500, len(vol_b) / 20)
        vol_a = gaussian_filter1d(vol_a, sigma=sigma_a)
        vol_b = gaussian_filter1d(vol_b, sigma=sigma_b)
        
        vol_a = np.clip(vol_a, 0, 1)
        vol_b = np.clip(vol_b, 0, 1)
        
        # Convert to gain
        gain_a = 10 ** ((vol_a * 60 - 60) / 20)
        gain_b = 10 ** ((vol_b * 60 - 60) / 20)
        
        # Smooth gains
        gain_a = gaussian_filter1d(gain_a, sigma=min(300, len(gain_a) / 25))
        gain_b = gaussian_filter1d(gain_b, sigma=min(300, len(gain_b) / 25))
        
        # Apply
        seg_a = seg_a * gain_a[:, np.newaxis]
        seg_b = seg_b * gain_b[:, np.newaxis]
        
        return seg_a, seg_b
    
    def _apply_smooth_crossfade(self, seg_a: np.ndarray, seg_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply smooth crossfade with easing."""
        t = np.linspace(0, 1, len(seg_a))
        # Smoothstep easing function
        ease_t = t * t * (3 - 2 * t)
        
        seg_a = seg_a * (1 - ease_t)[:, np.newaxis]
        seg_b = seg_b * ease_t[:, np.newaxis]
        
        return seg_a, seg_b

