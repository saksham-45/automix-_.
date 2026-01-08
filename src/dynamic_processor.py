"""
Dynamic EQ and Frequency Processing Module

Intelligent frequency management during transitions:
- Frequency masking prevention
- Dynamic EQ automation
- Bass swap automation
- Multi-band processing
"""
import numpy as np
import librosa
from scipy import signal
from typing import Dict, List, Tuple, Optional

from src.psychoacoustics import PsychoacousticAnalyzer


class DynamicProcessor:
    """
    Intelligent EQ and frequency processing for smooth transitions.
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.psychoacoustics = PsychoacousticAnalyzer(sr=sr)
        
        # Frequency bands (Hz)
        self.bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 6000),
            'high': (6000, 20000)
        }
    
    def analyze_frequency_clash(self,
                                y_a: np.ndarray,
                                y_b: np.ndarray) -> Dict:
        """
        Analyze frequency clashes and provide EQ recommendations.
        """
        clash_analysis = self.psychoacoustics.predict_frequency_clash(y_a, y_b)
        
        # Get band-specific clash analysis
        band_clashes = self._analyze_band_clashes(y_a, y_b)
        
        # Recommendations
        recommendations = []
        eq_curves = {
            'bass_a': np.ones(100),  # Default: no cut
            'bass_b': np.ones(100),
            'mid_a': np.ones(100),
            'mid_b': np.ones(100),
            'high_a': np.ones(100),
            'high_b': np.ones(100)
        }
        
        # Bass swap recommendation
        if clash_analysis['bass_clash'] > 0.4:
            recommendations.append('bass_swap')
            # Cut bass on outgoing, boost on incoming
            eq_curves['bass_a'] = np.linspace(1.0, 0.3, 100)  # Fade out bass
            eq_curves['bass_b'] = np.linspace(0.3, 1.0, 100)  # Fade in bass
        
        # Mid-range clash
        if band_clashes['mid_clash'] > 0.5:
            recommendations.append('mid_cut')
            # Cut mids on outgoing to make room
            eq_curves['mid_a'] = np.linspace(1.0, 0.6, 100)
        
        # High-frequency clash
        if band_clashes['high_clash'] > 0.5:
            recommendations.append('high_cut_outgoing')
            eq_curves['high_a'] = np.linspace(1.0, 0.7, 100)
        
        return {
            'clash_score': clash_analysis['clash_score'],
            'bass_clash': clash_analysis['bass_clash'],
            'band_clashes': band_clashes,
            'recommendations': recommendations,
            'eq_automation_curves': eq_curves
        }
    
    def _analyze_band_clashes(self, y_a: np.ndarray, y_b: np.ndarray) -> Dict:
        """Analyze clashes in specific frequency bands."""
        # Get spectral energy in each band
        S_a = np.abs(librosa.stft(y_a, n_fft=2048))
        S_b = np.abs(librosa.stft(y_b, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        
        band_clashes = {}
        
        for band_name, (low, high) in self.bands.items():
            # Find frequency bins in this band
            band_mask = (freqs >= low) & (freqs < high)
            
            # Energy in this band
            energy_a = np.mean(S_a[band_mask, :] ** 2)
            energy_b = np.mean(S_b[band_mask, :] ** 2)
            
            # Clash score (normalized)
            clash = (energy_a * energy_b) / (np.max(energy_a, energy_b) + 1e-10)
            band_clashes[band_name] = float(clash)
        
        return band_clashes
    
    def apply_dynamic_eq(self,
                        y: np.ndarray,
                        eq_curve: np.ndarray,
                        band: str = 'bass') -> np.ndarray:
        """
        Apply dynamic EQ curve to audio.
        
        Args:
            y: Audio signal
            eq_curve: Gain curve (0-1, normalized over transition duration)
            band: Frequency band ('bass', 'mid', 'high')
        
        Returns:
            EQ-processed audio
        """
        # Get frequency range for band
        low, high = self.bands.get(band, (60, 250))
        
        # Design filter
        # Use shelf filters for smooth transitions
        nyquist = self.sr / 2
        
        if band == 'bass':
            # Low shelf
            sos = signal.iirfilter(4, low / nyquist, btype='high', output='sos')
        elif band == 'high':
            # High shelf
            sos = signal.iirfilter(4, high / nyquist, btype='low', output='sos')
        else:
            # Bandpass for mid
            sos = signal.iirfilter(4, [low / nyquist, high / nyquist], btype='band', output='sos')
        
        # Apply filter
        y_filtered = signal.sosfilt(sos, y)
        
        # Apply gain curve (interpolate to match audio length)
        if len(eq_curve) != len(y):
            indices = np.linspace(0, len(eq_curve) - 1, len(y))
            gain = np.interp(indices, np.arange(len(eq_curve)), eq_curve)
        else:
            gain = eq_curve
        
        # Convert gain (0-1) to dB
        gain_db = 20 * np.log10(gain + 1e-10)
        
        # Apply gain
        y_processed = y.copy()
        if y.ndim > 1:
            gain = gain[:, np.newaxis]
        
        # Mix original and filtered
        y_processed = y * (1 - gain) + y_filtered * gain
        
        return y_processed
    
    def create_bass_swap_automation(self,
                                   transition_duration_sec: float,
                                   swap_point_ratio: float = 0.5) -> Dict:
        """
        Create automation curves for bass swap technique.
        
        Args:
            transition_duration_sec: Duration of transition
            swap_point_ratio: When to swap (0.5 = halfway)
        
        Returns:
            Dict with automation curves
        """
        n_samples = int(transition_duration_sec * self.sr)
        t = np.linspace(0, 1, n_samples)
        
        swap_point = swap_point_ratio
        
        # Bass cut on outgoing: gradual cut, then steep drop at swap point
        bass_a = np.ones(n_samples)
        before_swap = t < swap_point
        after_swap = t >= swap_point
        
        # Gradual cut before swap
        bass_a[before_swap] = 1.0 - (t[before_swap] / swap_point) * 0.5
        
        # Steep drop after swap
        bass_a[after_swap] = 0.5 * (1.0 - (t[after_swap] - swap_point) / (1 - swap_point))
        bass_a = np.maximum(bass_a, 0.1)  # Never fully silent
        
        # Bass boost on incoming: steep rise at swap point
        bass_b = np.ones(n_samples)
        bass_b[before_swap] = 0.3 + 0.2 * (t[before_swap] / swap_point)
        bass_b[after_swap] = 0.5 + 0.5 * ((t[after_swap] - swap_point) / (1 - swap_point))
        
        # High-pass filter automation for outgoing (removes bass)
        highpass_a = np.ones(n_samples)
        # Gradually increase high-pass frequency
        highpass_a = np.linspace(0, 200, n_samples)  # Hz
        
        return {
            'bass_gain_a': bass_a.tolist(),
            'bass_gain_b': bass_b.tolist(),
            'highpass_freq_a': highpass_a.tolist(),
            'time': t.tolist()
        }
    
    def apply_bass_swap(self,
                       y_a: np.ndarray,
                       y_b: np.ndarray,
                       automation: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply bass swap automation to two audio signals.
        """
        n_samples = min(len(y_a), len(y_b))
        
        # Get automation curves
        bass_gain_a = np.array(automation['bass_gain_a'][:n_samples])
        bass_gain_b = np.array(automation['bass_gain_b'][:n_samples])
        
        # Apply bass EQ
        # Low-pass filter for bass
        nyquist = self.sr / 2
        sos_low = signal.iirfilter(4, 250 / nyquist, btype='low', output='sos')
        
        y_a_bass = signal.sosfilt(sos_low, y_a[:n_samples])
        y_b_bass = signal.sosfilt(sos_low, y_b[:n_samples])
        
        # Apply gains
        if y_a.ndim > 1:
            bass_gain_a = bass_gain_a[:, np.newaxis]
            bass_gain_b = bass_gain_b[:, np.newaxis]
        
        y_a_processed = y_a[:n_samples].copy()
        y_a_processed = y_a_processed - y_a_bass * (1 - bass_gain_a)  # Remove bass
        
        y_b_processed = y_b[:n_samples].copy()
        y_b_processed = y_b_processed - y_b_bass * (1 - bass_gain_b) + y_b_bass * bass_gain_b  # Add bass
        
        return y_a_processed, y_b_processed
    
    def protect_vocal_frequencies(self,
                                 y: np.ndarray,
                                 gain_reduction: float = 0.2) -> np.ndarray:
        """
        Protect vocal frequencies (typically 200-2000 Hz) during transitions.
        
        Applies less gain reduction to vocal range to maintain clarity.
        """
        # Vocal frequency range: 200-2000 Hz
        nyquist = self.sr / 2
        sos_vocal = signal.iirfilter(4, [200 / nyquist, 2000 / nyquist], btype='band', output='sos')
        
        # Extract vocal range
        y_vocal = signal.sosfilt(sos_vocal, y)
        
        # Apply less reduction to vocals
        # (This is a simplified version - full implementation would be more complex)
        y_protected = y - y_vocal * gain_reduction
        
        return y_protected

