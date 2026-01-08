"""
Transition Analysis Module

Analyzes HOW transitions were executed:
- Volume curves (crossfade analysis)
- EQ automation (bass swaps, filter sweeps)
- Effect detection (reverb, delay)
- Beat alignment
- Technique classification
"""

import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json


@dataclass
class VolumeCurve:
    """Volume automation during transition"""
    times_relative_sec: List[float]
    track_a_gain_db: List[float]
    track_b_gain_db: List[float]
    crossfade_type: str  # "long_blend", "quick_cut", "fade_out", etc.


@dataclass
class EQAutomation:
    """EQ changes during transition"""
    bass_swap_detected: bool
    bass_swap_point_sec: Optional[float]
    track_a_bass_curve_db: Optional[List[float]]
    track_b_bass_curve_db: Optional[List[float]]
    
    highpass_sweep_detected: bool
    highpass_freq_curve_hz: Optional[List[float]]
    
    lowpass_sweep_detected: bool
    lowpass_freq_curve_hz: Optional[List[float]]


@dataclass
class BeatAlignment:
    """Beat matching information"""
    phase_offset_ms: float
    pitch_shift_semitones: float
    aligned_on_downbeat: bool
    alignment_quality: float


@dataclass
class TransitionAnalysis:
    """Complete analysis of a transition"""
    transition_id: str
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    
    technique_primary: str
    technique_secondary: List[str]
    technique_confidence: float
    
    volume_curves: VolumeCurve
    eq_automation: EQAutomation
    beat_alignment: BeatAlignment
    
    energy_during_transition: Dict[str, List[float]]
    spectral_during_transition: Dict[str, List[float]]
    
    quality_assessment: Dict[str, float]


class TransitionAnalyzer:
    """Analyzes how transitions were executed"""
    
    def __init__(self, sr: int = 22050, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length
    
    def analyze_transition(
        self,
        mix_audio: np.ndarray,
        transition_start_sec: float,
        transition_end_sec: float,
        track_a_audio: Optional[np.ndarray] = None,
        track_b_audio: Optional[np.ndarray] = None,
        track_a_bpm: Optional[float] = None,
        track_b_bpm: Optional[float] = None
    ) -> TransitionAnalysis:
        """
        Analyze a transition segment
        
        Args:
            mix_audio: Full mix audio
            transition_start_sec: Start of transition in mix
            transition_end_sec: End of transition in mix
            track_a_audio: Original track A (optional, for better analysis)
            track_b_audio: Original track B (optional, for better analysis)
            track_a_bpm: BPM of track A
            track_b_bpm: BPM of track B
            
        Returns:
            Complete transition analysis
        """
        transition_id = f"trans_{int(transition_start_sec)}"
        duration_sec = transition_end_sec - transition_start_sec
        
        # Extract transition segment
        start_sample = int(transition_start_sec * self.sr)
        end_sample = int(transition_end_sec * self.sr)
        transition_segment = mix_audio[start_sample:end_sample]
        
        # Analyze volume curves
        volume_curves = self._analyze_volume_curves(
            transition_segment,
            track_a_audio,
            track_b_audio,
            transition_start_sec,
            duration_sec
        )
        
        # Analyze EQ automation
        eq_automation = self._analyze_eq_automation(
            transition_segment,
            duration_sec
        )
        
        # Analyze beat alignment
        beat_alignment = self._analyze_beat_alignment(
            transition_segment,
            track_a_bpm,
            track_b_bpm
        )
        
        # Analyze energy and spectral content
        energy_analysis = self._analyze_energy_during_transition(transition_segment)
        spectral_analysis = self._analyze_spectral_during_transition(transition_segment)
        
        # Classify technique
        technique, technique_secondary, confidence = self._classify_technique(
            volume_curves,
            eq_automation,
            duration_sec,
            energy_analysis
        )
        
        # Quality assessment
        quality = self._assess_quality(
            volume_curves,
            eq_automation,
            beat_alignment,
            energy_analysis,
            spectral_analysis
        )
        
        return TransitionAnalysis(
            transition_id=transition_id,
            start_time_sec=transition_start_sec,
            end_time_sec=transition_end_sec,
            duration_sec=duration_sec,
            technique_primary=technique,
            technique_secondary=technique_secondary,
            technique_confidence=confidence,
            volume_curves=volume_curves,
            eq_automation=eq_automation,
            beat_alignment=beat_alignment,
            energy_during_transition=energy_analysis,
            spectral_during_transition=spectral_analysis,
            quality_assessment=quality
        )
    
    def _analyze_volume_curves(
        self,
        transition_segment: np.ndarray,
        track_a_audio: Optional[np.ndarray],
        track_b_audio: Optional[np.ndarray],
        transition_start_sec: float,
        duration_sec: float
    ) -> VolumeCurve:
        """
        Estimate volume curves for both tracks during transition
        
        If original tracks are available, use correlation.
        Otherwise, use spectral analysis to estimate presence.
        """
        # Create time points (every 4 seconds)
        n_points = max(9, int(duration_sec / 4) + 1)
        times = np.linspace(0, duration_sec, n_points)
        
        if track_a_audio is not None and track_b_audio is not None:
            # Use correlation with original tracks
            track_a_gain = []
            track_b_gain = []
            
            window_sec = 2
            window_samples = int(window_sec * self.sr)
            
            for t in times:
                # Extract window from transition
                window_start = int(t * self.sr)
                window_end = min(window_start + window_samples, len(transition_segment))
                
                if window_end <= window_start:
                    track_a_gain.append(-60)
                    track_b_gain.append(-60)
                    continue
                
                window = transition_segment[window_start:window_end]
                
                # Find best matching position in original tracks
                # (simplified - in practice, use more sophisticated matching)
                track_a_corr = self._compute_similarity(window, track_a_audio)
                track_b_corr = self._compute_similarity(window, track_b_audio)
                
                # Convert correlation to gain estimate
                track_a_gain.append(self._correlation_to_db(track_a_corr))
                track_b_gain.append(self._correlation_to_db(track_b_corr))
        else:
            # Estimate using spectral analysis
            # Assume gradual crossfade
            track_a_gain = []
            track_b_gain = []
            
            for t in times:
                # Linear crossfade assumption
                fade_out = 1.0 - (t / duration_sec)
                fade_in = t / duration_sec
                
                track_a_gain.append(20 * np.log10(max(0.001, fade_out)))
                track_b_gain.append(20 * np.log10(max(0.001, fade_in)))
        
        # Classify crossfade type
        crossfade_type = self._classify_crossfade_type(times, track_a_gain, track_b_gain, duration_sec)
        
        return VolumeCurve(
            times_relative_sec=times.tolist(),
            track_a_gain_db=track_a_gain,
            track_b_gain_db=track_b_gain,
            crossfade_type=crossfade_type
        )
    
    def _compute_similarity(self, segment: np.ndarray, reference: np.ndarray) -> float:
        """Compute similarity between segment and reference"""
        # Use spectral correlation
        S_seg = np.abs(librosa.stft(segment, hop_length=self.hop_length))
        S_ref = np.abs(librosa.stft(reference, hop_length=self.hop_length))
        
        # Normalize
        S_seg = S_seg / (np.linalg.norm(S_seg) + 1e-8)
        S_ref = S_ref / (np.linalg.norm(S_ref) + 1e-8)
        
        # Find best matching window
        min_len = min(S_seg.shape[1], S_ref.shape[1])
        S_seg_trim = S_seg[:, :min_len]
        S_ref_trim = S_ref[:, :min_len]
        
        # Cosine similarity
        similarity = np.mean(S_seg_trim * S_ref_trim)
        
        return float(similarity)
    
    def _correlation_to_db(self, correlation: float) -> float:
        """Convert correlation score to dB gain estimate"""
        # Map [0, 1] to [-60, 0] dB
        correlation = max(0, min(1, correlation))
        return -60 * (1 - correlation)
    
    def _classify_crossfade_type(
        self,
        times: np.ndarray,
        track_a_gain: List[float],
        track_b_gain: List[float],
        duration_sec: float
    ) -> str:
        """Classify the type of crossfade"""
        # Check if it's a quick cut
        if duration_sec < 4:
            return "quick_cut"
        
        # Check if track A fades out completely
        final_a_gain = track_a_gain[-1]
        if final_a_gain < -40:
            if duration_sec < 16:
                return "fade_out"
            else:
                return "long_blend"
        
        # Check if it's a drop mix (sudden cut)
        if track_a_gain[0] > -6 and len(track_a_gain) > 2:
            if track_a_gain[1] < -30:
                return "drop_mix"
        
        return "long_blend"
    
    def _analyze_eq_automation(
        self,
        transition_segment: np.ndarray,
        duration_sec: float
    ) -> EQAutomation:
        """Detect EQ changes during transition"""
        # Compute spectrogram
        S = np.abs(librosa.stft(transition_segment, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # Define frequency bands
        bass_mask = (freqs >= 60) & (freqs < 250)
        mid_mask = (freqs >= 250) & (freqs < 2000)
        high_mask = freqs >= 2000
        
        # Compute energy per band over time
        bass_energy = S[bass_mask].mean(axis=0)
        mid_energy = S[mid_mask].mean(axis=0)
        high_energy = S[high_mask].mean(axis=0)
        
        # Normalize
        bass_energy = bass_energy / (bass_energy.max() + 1e-8)
        mid_energy = mid_energy / (mid_energy.max() + 1e-8)
        high_energy = high_energy / (high_energy.max() + 1e-8)
        
        # Detect bass swap
        bass_swap_detected, bass_swap_point, bass_curves = self._detect_bass_swap(
            bass_energy, duration_sec
        )
        
        # Detect filter sweeps
        highpass_detected, highpass_curve = self._detect_highpass_sweep(
            high_energy, mid_energy, duration_sec
        )
        
        lowpass_detected, lowpass_curve = self._detect_lowpass_sweep(
            high_energy, duration_sec
        )
        
        return EQAutomation(
            bass_swap_detected=bass_swap_detected,
            bass_swap_point_sec=bass_swap_point,
            track_a_bass_curve_db=bass_curves[0] if bass_curves else None,
            track_b_bass_curve_db=bass_curves[1] if bass_curves else None,
            highpass_sweep_detected=highpass_detected,
            highpass_freq_curve_hz=highpass_curve,
            lowpass_sweep_detected=lowpass_detected,
            lowpass_freq_curve_hz=lowpass_curve
        )
    
    def _detect_bass_swap(
        self,
        bass_energy: np.ndarray,
        duration_sec: float
    ) -> Tuple[bool, Optional[float], Optional[Tuple[List[float], List[float]]]]:
        """Detect bass swap pattern"""
        # Look for significant dip in bass energy
        # Bass swap: steady → dip → steady (different level)
        
        if len(bass_energy) < 5:
            return False, None, None
        
        # Find minimum point
        min_idx = np.argmin(bass_energy)
        min_value = bass_energy[min_idx]
        
        # Check if dip is significant (>30% drop)
        if min_value < 0.7:
            # Estimate swap point
            swap_point_sec = (min_idx / len(bass_energy)) * duration_sec
            
            # Estimate curves (simplified)
            n_points = len(bass_energy)
            times = np.linspace(0, duration_sec, n_points)
            
            # Track A: fade out bass
            track_a_curve = []
            for i, t in enumerate(times):
                if t < swap_point_sec:
                    # Fade out
                    fade = 1.0 - (t / swap_point_sec)
                    track_a_curve.append(20 * np.log10(max(0.001, fade)))
                else:
                    track_a_curve.append(-60)
            
            # Track B: fade in bass
            track_b_curve = []
            for i, t in enumerate(times):
                if t < swap_point_sec:
                    track_b_curve.append(-60)
                else:
                    # Fade in
                    fade = (t - swap_point_sec) / (duration_sec - swap_point_sec)
                    track_b_curve.append(20 * np.log10(max(0.001, fade)))
            
            return True, float(swap_point_sec), (track_a_curve, track_b_curve)
        
        return False, None, None
    
    def _detect_highpass_sweep(
        self,
        high_energy: np.ndarray,
        mid_energy: np.ndarray,
        duration_sec: float
    ) -> Tuple[bool, Optional[List[float]]]:
        """Detect highpass filter sweep"""
        # Highpass sweep: high frequencies increase, low frequencies decrease
        if len(high_energy) < 3:
            return False, None
        
        # Check if high energy increases while mid decreases
        high_slope = (high_energy[-1] - high_energy[0]) / duration_sec
        mid_slope = (mid_energy[-1] - mid_energy[0]) / duration_sec
        
        if high_slope > 0.01 and mid_slope < -0.01:
            # Estimate frequency curve (simplified)
            n_points = len(high_energy)
            times = np.linspace(0, duration_sec, n_points)
            freq_curve = []
            
            for i, t in enumerate(times):
                # Estimate cutoff frequency (20 Hz to 2000 Hz)
                progress = t / duration_sec
                cutoff = 20 + (progress * 1980)
                freq_curve.append(cutoff)
            
            return True, freq_curve
        
        return False, None
    
    def _detect_lowpass_sweep(
        self,
        high_energy: np.ndarray,
        duration_sec: float
    ) -> Tuple[bool, Optional[List[float]]]:
        """Detect lowpass filter sweep"""
        # Lowpass sweep: high frequencies decrease
        if len(high_energy) < 3:
            return False, None
        
        high_slope = (high_energy[-1] - high_energy[0]) / duration_sec
        
        if high_slope < -0.01:
            # Estimate frequency curve
            n_points = len(high_energy)
            times = np.linspace(0, duration_sec, n_points)
            freq_curve = []
            
            for i, t in enumerate(times):
                # Estimate cutoff frequency (20000 Hz down to 2000 Hz)
                progress = t / duration_sec
                cutoff = 20000 - (progress * 18000)
                freq_curve.append(cutoff)
            
            return True, freq_curve
        
        return False, None
    
    def _analyze_beat_alignment(
        self,
        transition_segment: np.ndarray,
        track_a_bpm: Optional[float],
        track_b_bpm: Optional[float]
    ) -> BeatAlignment:
        """Analyze beat alignment during transition"""
        # Estimate tempo of transition segment
        onset_env = librosa.onset.onset_strength(y=transition_segment, sr=self.sr)
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=self.hop_length
        )
        
        # Estimate phase offset (simplified)
        beat_times = librosa.frames_to_time(beats, sr=self.sr, hop_length=self.hop_length)
        if len(beat_times) > 0:
            # Check if first beat is close to 0
            phase_offset_ms = abs(beat_times[0]) * 1000
            aligned_on_downbeat = phase_offset_ms < 50
        else:
            phase_offset_ms = 0
            aligned_on_downbeat = False
        
        # Estimate pitch shift
        pitch_shift = 0.0
        if track_a_bpm and track_b_bpm:
            # Estimate pitch shift from BPM difference
            bpm_ratio = track_b_bpm / track_a_bpm
            pitch_shift = 12 * np.log2(bpm_ratio)
        
        # Estimate alignment quality (simplified)
        if len(beats) > 4:
            beat_intervals = np.diff(beat_times)
            regularity = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
            alignment_quality = max(0, min(1, regularity))
        else:
            alignment_quality = 0.5
        
        return BeatAlignment(
            phase_offset_ms=float(phase_offset_ms),
            pitch_shift_semitones=float(pitch_shift),
            aligned_on_downbeat=aligned_on_downbeat,
            alignment_quality=float(alignment_quality)
        )
    
    def _analyze_energy_during_transition(
        self,
        transition_segment: np.ndarray
    ) -> Dict[str, List[float]]:
        """Analyze energy during transition"""
        # Compute RMS energy
        rms = librosa.feature.rms(y=transition_segment, hop_length=self.hop_length)[0]
        
        # Normalize
        rms = rms / (rms.max() + 1e-8)
        
        # Downsample to reasonable number of points
        n_points = min(9, len(rms))
        indices = np.linspace(0, len(rms) - 1, n_points, dtype=int)
        rms_downsampled = rms[indices]
        
        return {
            "times_relative_sec": (indices * self.hop_length / self.sr).tolist(),
            "combined_energy": rms_downsampled.tolist()
        }
    
    def _analyze_spectral_during_transition(
        self,
        transition_segment: np.ndarray
    ) -> Dict[str, List[float]]:
        """Analyze spectral content during transition"""
        S = np.abs(librosa.stft(transition_segment, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # Frequency bands
        bass_mask = (freqs >= 60) & (freqs < 250)
        mid_mask = (freqs >= 250) & (freqs < 2000)
        high_mask = freqs >= 2000
        
        bass_energy = S[bass_mask].mean(axis=0)
        mid_energy = S[mid_mask].mean(axis=0)
        high_energy = S[high_mask].mean(axis=0)
        
        # Normalize
        bass_energy = bass_energy / (bass_energy.max() + 1e-8)
        mid_energy = mid_energy / (mid_energy.max() + 1e-8)
        high_energy = high_energy / (high_energy.max() + 1e-8)
        
        # Downsample
        n_points = min(9, len(bass_energy))
        indices = np.linspace(0, len(bass_energy) - 1, n_points, dtype=int)
        
        times = (indices * self.hop_length / self.sr).tolist()
        
        return {
            "times_relative_sec": times,
            "combined_bass": bass_energy[indices].tolist(),
            "combined_mid": mid_energy[indices].tolist(),
            "combined_high": high_energy[indices].tolist()
        }
    
    def _classify_technique(
        self,
        volume_curves: VolumeCurve,
        eq_automation: EQAutomation,
        duration_sec: float,
        energy_analysis: Dict[str, List[float]]
    ) -> Tuple[str, List[str], float]:
        """Classify the transition technique"""
        technique_secondary = []
        confidence = 0.7
        
        # Primary technique based on duration and volume curve
        if duration_sec < 4:
            primary = "quick_cut"
        elif volume_curves.crossfade_type == "drop_mix":
            primary = "drop_mix"
        elif duration_sec > 30:
            primary = "long_blend"
        else:
            primary = "crossfade"
        
        # Secondary techniques
        if eq_automation.bass_swap_detected:
            technique_secondary.append("bass_swap")
            confidence += 0.1
        
        if eq_automation.highpass_sweep_detected:
            technique_secondary.append("filter_sweep")
            confidence += 0.05
        
        if eq_automation.lowpass_sweep_detected:
            technique_secondary.append("lowpass_sweep")
            confidence += 0.05
        
        confidence = min(1.0, confidence)
        
        return primary, technique_secondary, confidence
    
    def _assess_quality(
        self,
        volume_curves: VolumeCurve,
        eq_automation: EQAutomation,
        beat_alignment: BeatAlignment,
        energy_analysis: Dict[str, List[float]],
        spectral_analysis: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Assess overall transition quality"""
        quality = {}
        
        # Beat match quality
        quality["beat_match_quality"] = beat_alignment.alignment_quality
        
        # Energy flow smoothness
        energy_curve = energy_analysis["combined_energy"]
        if len(energy_curve) > 2:
            energy_changes = np.abs(np.diff(energy_curve))
            smoothness = 1.0 - np.mean(energy_changes)
            quality["energy_flow_smoothness"] = max(0, min(1, smoothness))
        else:
            quality["energy_flow_smoothness"] = 0.5
        
        # Spectral balance (check for muddiness)
        bass_curve = spectral_analysis["combined_bass"]
        mid_curve = spectral_analysis["combined_mid"]
        
        if len(bass_curve) > 0 and len(mid_curve) > 0:
            # High bass + high mid = mud
            mud_score = np.mean([b * m for b, m in zip(bass_curve, mid_curve)])
            quality["spectral_balance_maintained"] = 1.0 - mud_score
        else:
            quality["spectral_balance_maintained"] = 0.5
        
        # Overall quality (weighted average)
        quality["overall_transition_quality"] = (
            quality["beat_match_quality"] * 0.4 +
            quality["energy_flow_smoothness"] * 0.3 +
            quality["spectral_balance_maintained"] * 0.3
        )
        
        return quality

