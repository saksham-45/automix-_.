"""
Quality Assessment Module

Perceptual quality metrics for transitions:
- Smoothness scoring
- Clarity assessment
- Harmonic tension detection
- Frequency balance analysis
"""
import numpy as np
from typing import Dict, Optional, List

from src.psychoacoustics import PsychoacousticAnalyzer
from src.harmonic_analyzer import HarmonicAnalyzer


class QualityAssessor:
    """
    Assesses transition quality using perceptual metrics.
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.psychoacoustics = PsychoacousticAnalyzer(sr=sr)
        self.harmonic_analyzer = HarmonicAnalyzer()
    
    def assess_transition_quality(self,
                                  transition_audio: np.ndarray,
                                  y_a: Optional[np.ndarray] = None,
                                  y_b: Optional[np.ndarray] = None,
                                  key_a: Optional[str] = None,
                                  key_b: Optional[str] = None) -> Dict:
        """
        Comprehensive quality assessment of a transition.
        
        Returns:
            Dict with quality scores and analysis
        """
        # Smoothness score
        smoothness = self._assess_smoothness(transition_audio)
        
        # Clarity score
        clarity = self._assess_clarity(transition_audio, y_a, y_b)
        
        # Harmonic tension
        harmonic_tension = 0.0
        if key_a and key_b:
            harmonic_score = self.harmonic_analyzer.score_transition_harmonics(
                key_a, key_b, 120, 120  # Tempo not critical for harmonic assessment
            )
            harmonic_tension = 1.0 - harmonic_score['overall_score']
        
        # Frequency balance
        frequency_balance = self._assess_frequency_balance(transition_audio)
        
        # Energy continuity
        energy_continuity = self._assess_energy_continuity(transition_audio)
        
        # Overall score (weighted average)
        overall_score = (
            smoothness['score'] * 0.3 +
            clarity['score'] * 0.25 +
            (1 - harmonic_tension) * 0.2 +
            frequency_balance['score'] * 0.15 +
            energy_continuity['score'] * 0.1
        )
        
        return {
            'overall_score': float(overall_score),
            'smoothness': smoothness,
            'clarity': clarity,
            'harmonic_tension': float(harmonic_tension),
            'frequency_balance': frequency_balance,
            'energy_continuity': energy_continuity,
            'quality_rating': self._rating_from_score(overall_score)
        }
    
    def _assess_smoothness(self, audio: np.ndarray) -> Dict:
        """
        Assess smoothness - no abrupt changes, gradual transitions.
        """
        # Analyze spectral flux (rate of change)
        import librosa
        
        # Spectral flux
        S = np.abs(librosa.stft(audio, n_fft=2048))
        flux = np.mean(np.diff(S, axis=1) ** 2)
        
        # Lower flux = smoother
        # Normalize flux (0-1, higher = less smooth)
        flux_normalized = min(1.0, flux / 1000)
        smoothness_score = 1.0 - flux_normalized
        
        # Also check for clicks/pops (sudden amplitude changes)
        amplitude_diff = np.abs(np.diff(np.abs(audio)))
        large_jumps = np.sum(amplitude_diff > 0.1) / len(audio)
        click_score = max(0, 1.0 - large_jumps * 10)
        
        # Combined smoothness
        score = (smoothness_score * 0.7 + click_score * 0.3)
        
        return {
            'score': float(score),
            'spectral_flux': float(flux),
            'click_score': float(click_score),
            'analysis': 'smooth' if score > 0.8 else 'moderate' if score > 0.6 else 'abrupt'
        }
    
    def _assess_clarity(self, transition: np.ndarray, y_a: Optional[np.ndarray], y_b: Optional[np.ndarray]) -> Dict:
        """
        Assess clarity - can you hear both tracks clearly?
        """
        # Analyze frequency masking during transition
        if y_a is not None and y_b is not None:
            clash_analysis = self.psychoacoustics.predict_frequency_clash(y_a, y_b)
            clash_score = clash_analysis['clash_score']
            
            # Lower clash = higher clarity
            clarity_from_clash = 1.0 - clash_score
        else:
            clash_score = 0.5
            clarity_from_clash = 0.5
        
        # Spectral clarity (how clear are individual frequencies)
        import librosa
        spectral_contrast = librosa.feature.spectral_contrast(y=transition, sr=self.sr)
        contrast_score = np.mean(spectral_contrast) / 1000  # Normalize
        contrast_score = min(1.0, contrast_score)
        
        # Combined clarity
        score = (clarity_from_clash * 0.6 + contrast_score * 0.4)
        
        return {
            'score': float(score),
            'frequency_clash': float(clash_score),
            'spectral_contrast': float(np.mean(spectral_contrast)),
            'analysis': 'clear' if score > 0.8 else 'moderate' if score > 0.6 else 'muddy'
        }
    
    def _assess_frequency_balance(self, audio: np.ndarray) -> Dict:
        """
        Assess frequency balance - are all frequencies represented well?
        """
        import librosa
        
        # Get spectral energy in bands
        S = np.abs(librosa.stft(audio, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        
        # Energy in bass, mid, high
        bass_mask = (freqs >= 60) & (freqs < 250)
        mid_mask = (freqs >= 250) & (freqs < 2000)
        high_mask = (freqs >= 2000) & (freqs < 8000)
        
        bass_energy = np.mean(S[bass_mask, :] ** 2)
        mid_energy = np.mean(S[mid_mask, :] ** 2)
        high_energy = np.mean(S[high_mask, :] ** 2)
        
        # Normalize
        total_energy = bass_energy + mid_energy + high_energy
        if total_energy > 0:
            bass_ratio = bass_energy / total_energy
            mid_ratio = mid_energy / total_energy
            high_ratio = high_energy / total_energy
        else:
            bass_ratio = mid_ratio = high_ratio = 0.33
        
        # Ideal balance: ~30% bass, ~40% mid, ~30% high (roughly)
        ideal_bass = 0.3
        ideal_mid = 0.4
        ideal_high = 0.3
        
        balance_score = 1.0 - (
            abs(bass_ratio - ideal_bass) +
            abs(mid_ratio - ideal_mid) +
            abs(high_ratio - ideal_high)
        ) / 3
        
        balance_score = max(0, min(1, balance_score))
        
        return {
            'score': float(balance_score),
            'bass_ratio': float(bass_ratio),
            'mid_ratio': float(mid_ratio),
            'high_ratio': float(high_ratio),
            'analysis': 'balanced' if balance_score > 0.8 else 'moderate' if balance_score > 0.6 else 'unbalanced'
        }
    
    def _assess_energy_continuity(self, audio: np.ndarray) -> Dict:
        """
        Assess energy continuity - does energy flow feel natural?
        """
        import librosa
        
        # RMS energy curve
        rms = librosa.feature.rms(y=audio, hop_length=512)[0]
        
        # Check for sudden drops or spikes
        rms_diff = np.abs(np.diff(rms))
        large_changes = np.sum(rms_diff > np.std(rms)) / len(rms_diff)
        
        # Continuity score: fewer large changes = better
        continuity_score = max(0, 1.0 - large_changes * 5)
        
        # Energy trend (should be smooth, not erratic)
        rms_smooth = np.convolve(rms, np.ones(10)/10, mode='same')
        trend_smoothness = 1.0 - (np.std(rms - rms_smooth) / (np.mean(rms) + 1e-10))
        trend_smoothness = max(0, min(1, trend_smoothness))
        
        combined_score = (continuity_score * 0.6 + trend_smoothness * 0.4)
        
        return {
            'score': float(combined_score),
            'large_changes': int(np.sum(rms_diff > np.std(rms))),
            'trend_smoothness': float(trend_smoothness),
            'analysis': 'smooth' if combined_score > 0.8 else 'moderate' if combined_score > 0.6 else 'erratic'
        }
    
    def _rating_from_score(self, score: float) -> str:
        """Convert numerical score to quality rating."""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.75:
            return 'very_good'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def compare_transitions(self, transitions: List[Dict]) -> Dict:
        """
        Compare multiple transition variations and rank them.
        
        Args:
            transitions: List of transition dicts with 'audio' and metadata
        
        Returns:
            Ranked transitions with quality scores
        """
        scored = []
        
        for i, trans in enumerate(transitions):
            quality = self.assess_transition_quality(
                trans['audio'],
                trans.get('y_a'),
                trans.get('y_b'),
                trans.get('key_a'),
                trans.get('key_b')
            )
            scored.append({
                'index': i,
                'quality': quality,
                'overall_score': quality['overall_score']
            })
        
        # Sort by score
        scored.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return {
            'rankings': scored,
            'best_index': scored[0]['index'] if scored else None,
            'best_score': scored[0]['overall_score'] if scored else 0.0
        }

