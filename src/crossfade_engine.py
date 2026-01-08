"""
Perceptual Crossfade Engine

Perceptually-correct crossfading and volume automation:
- Equal-power crossfades
- LUFS-based loudness matching
- Adaptive curves based on frequency content
"""
import numpy as np
from typing import Dict, Tuple, Optional, List

from src.psychoacoustics import PsychoacousticAnalyzer


class CrossfadeEngine:
    """
    Perceptual crossfade engine using psychoacoustic principles.
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.psychoacoustics = PsychoacousticAnalyzer(sr=sr)
    
    def create_equal_power_crossfade(self, n_samples: int, curve_shape: str = 'smooth') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create equal-power crossfade curves.
        
        Equal-power maintains perceived loudness (not linear energy).
        """
        t = np.linspace(0, 1, n_samples)
        
        if curve_shape == 'linear':
            # Linear power curves
            fade_out = np.sqrt(1 - t)  # Square root for equal power
            fade_in = np.sqrt(t)
        
        elif curve_shape == 'smooth':
            # Smooth S-curve using cosine
            fade_out = np.sqrt(0.5 * (1 + np.cos(np.pi * t)))
            fade_in = np.sqrt(0.5 * (1 - np.cos(np.pi * t)))
        
        elif curve_shape == 'fast':
            # Faster fade (more aggressive)
            fade_out = np.sqrt((1 - t) ** 2)
            fade_in = np.sqrt(t ** 2)
        
        else:  # 'smooth' default
            fade_out = np.sqrt(0.5 * (1 + np.cos(np.pi * t)))
            fade_in = np.sqrt(0.5 * (1 - np.cos(np.pi * t)))
        
        return fade_out, fade_in
    
    def create_lufs_matched_curves(self,
                                  y_a: np.ndarray,
                                  y_b: np.ndarray,
                                  n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create volume curves that match LUFS (perceptual loudness).
        
        Maintains consistent perceived loudness throughout transition.
        """
        # Analyze LUFS for both signals
        lufs_a = self.psychoacoustics.analyze_loudness_lufs(y_a)
        lufs_b = self.psychoacoustics.analyze_loudness_lufs(y_b)
        
        # Target LUFS (average, or match to one)
        target_lufs = (lufs_a['integrated_lufs'] + lufs_b['integrated_lufs']) / 2
        
        # Calculate gain adjustments needed
        gain_a_db = target_lufs - lufs_a['integrated_lufs']
        gain_b_db = target_lufs - lufs_b['integrated_lufs']
        
        # Convert to linear gain
        gain_a = 10 ** (gain_a_db / 20)
        gain_b = 10 ** (gain_b_db / 20)
        
        # Create crossfade curves
        fade_out, fade_in = self.create_equal_power_crossfade(n_samples, 'smooth')
        
        # Apply gains
        vol_a = fade_out * gain_a
        vol_b = fade_in * gain_b
        
        return vol_a, vol_b
    
    def create_adaptive_crossfade(self,
                                 y_a: np.ndarray,
                                 y_b: np.ndarray,
                                 n_samples: int,
                                 frequency_aware: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create adaptive crossfade curves based on frequency content.
        
        Different fade rates for different frequency bands prevent masking.
        """
        # Base equal-power curves
        fade_out, fade_in = self.create_equal_power_crossfade(n_samples, 'smooth')
        
        if not frequency_aware:
            return fade_out, fade_in
        
        # Analyze frequency content
        # If signal A has strong bass, fade bass slower
        # If signal B has strong highs, fade highs in faster
        
        # Simplified: use overall spectral centroid
        import librosa
        
        centroid_a = np.mean(librosa.feature.spectral_centroid(y=y_a, sr=self.sr))
        centroid_b = np.mean(librosa.feature.spectral_centroid(y=y_b, sr=self.sr))
        
        # Adjust curves based on brightness
        # Brighter sounds (higher centroid) can be faded faster
        brightness_factor_a = min(1.5, centroid_a / 2000)  # Normalize
        brightness_factor_b = min(1.5, centroid_b / 2000)
        
        # Adjust fade rates
        # Brighter = faster fade
        fade_out_adj = fade_out ** (1 / brightness_factor_a)
        fade_in_adj = fade_in ** brightness_factor_b
        
        # Smooth the adjusted curves
        from scipy.ndimage import gaussian_filter1d
        fade_out_adj = gaussian_filter1d(fade_out_adj, sigma=n_samples / 100)
        fade_in_adj = gaussian_filter1d(fade_in_adj, sigma=n_samples / 100)
        
        # Normalize to prevent clipping
        fade_out_adj = np.clip(fade_out_adj, 0, 1)
        fade_in_adj = np.clip(fade_in_adj, 0, 1)
        
        return fade_out_adj, fade_in_adj
    
    def apply_crossfade(self,
                       y_a: np.ndarray,
                       y_b: np.ndarray,
                       vol_a: np.ndarray,
                       vol_b: np.ndarray) -> np.ndarray:
        """
        Apply crossfade curves to audio signals.
        
        Args:
            y_a: Outgoing audio
            y_b: Incoming audio
            vol_a: Volume curve for A
            vol_b: Volume curve for B
        
        Returns:
            Crossfaded audio
        """
        n_samples = min(len(y_a), len(y_b), len(vol_a), len(vol_b))
        
        # Ensure stereo
        if y_a.ndim == 1:
            y_a = y_a[:, np.newaxis]
        if y_b.ndim == 1:
            y_b = y_b[:, np.newaxis]
        
        # Resample volume curves if needed
        if len(vol_a) != n_samples:
            indices = np.linspace(0, len(vol_a) - 1, n_samples)
            vol_a = np.interp(indices, np.arange(len(vol_a)), vol_a)
        if len(vol_b) != n_samples:
            indices = np.linspace(0, len(vol_b) - 1, n_samples)
            vol_b = np.interp(indices, np.arange(len(vol_b)), vol_b)
        
        # Apply volumes
        if vol_a.ndim == 1:
            vol_a = vol_a[:, np.newaxis]
        if vol_b.ndim == 1:
            vol_b = vol_b[:, np.newaxis]
        
        mixed = y_a[:n_samples] * vol_a[:n_samples] + y_b[:n_samples] * vol_b[:n_samples]
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 0.95:
            mixed = mixed * (0.95 / max_val)
        
        return mixed
    
    def create_multi_stage_fade(self,
                               n_samples: int,
                               stages: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create multi-stage fade curves.
        
        Different fade rates for different frequency bands or time segments.
        
        Args:
            n_samples: Total samples
            stages: List of stage dicts with 'start_ratio', 'end_ratio', 'curve_shape'
        
        Returns:
            (fade_out, fade_in) curves
        """
        fade_out = np.ones(n_samples)
        fade_in = np.zeros(n_samples)
        
        for stage in stages:
            start_idx = int(stage['start_ratio'] * n_samples)
            end_idx = int(stage['end_ratio'] * n_samples)
            stage_samples = end_idx - start_idx
            
            if stage_samples <= 0:
                continue
            
            # Create curve for this stage
            stage_fade_out, stage_fade_in = self.create_equal_power_crossfade(
                stage_samples, stage.get('curve_shape', 'smooth')
            )
            
            # Insert into main curves
            fade_out[start_idx:end_idx] = stage_fade_out
            fade_in[start_idx:end_idx] = stage_fade_in
        
        return fade_out, fade_in

