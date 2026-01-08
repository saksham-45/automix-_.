"""
Training Data Extractor for DJ Transition AI

Extracts ML-ready training data from mix analyses:
- Input features (song states, compatibility, context)
- Output labels (transition execution parameters)
- Quality labels (reward signals)
"""

import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


class TrainingDataExtractor:
    """
    Extracts training data from mix analyses for ML model training.
    Converts raw transition analyses into structured input/output pairs.
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
    
    def extract_training_example(
        self,
        mix_audio: np.ndarray,
        transition_start_sec: float,
        transition_end_sec: float,
        track_a_analysis: Dict,
        track_b_analysis: Dict,
        transition_analysis: Dict,
        mix_context: Optional[Dict] = None
    ) -> Dict:
        """
        Extract complete training data for one transition.
        
        Args:
            mix_audio: Full mix audio array
            transition_start_sec: Start time of transition in mix
            transition_end_sec: End time of transition in mix
            track_a_analysis: Complete analysis of track A
            track_b_analysis: Complete analysis of track B
            transition_analysis: Transition execution analysis
            mix_context: Optional mix-level context
            
        Returns:
            Complete training example with input_features, output_labels, quality_labels
        """
        
        # Extract track states at transition points
        track_a_state = self._extract_song_state_at_time(
            track_a_analysis,
            transition_start_sec,
            lookback_bars=32
        )
        
        track_b_state = self._extract_song_state_at_time(
            track_b_analysis,
            0.0,  # Start of track B
            lookahead_bars=32
        )
        
        # Compute compatibility metrics
        compatibility = self._compute_compatibility(
            track_a_state,
            track_b_state
        )
        
        # Extract transition execution parameters
        output_labels = self._extract_output_labels(
            transition_analysis,
            transition_start_sec,
            transition_end_sec,
            track_a_analysis,
            track_b_analysis
        )
        
        # Assess quality
        quality_labels = self._assess_quality(
            transition_analysis,
            track_a_state,
            track_b_state,
            compatibility
        )
        
        # Build context
        context = mix_context or {}
        context.update({
            "transition_start_sec": transition_start_sec,
            "transition_end_sec": transition_end_sec,
            "transition_duration_sec": transition_end_sec - transition_start_sec
        })
        
        return {
            "input_features": {
                "track_a": track_a_state,
                "track_b": track_b_state,
                "compatibility": compatibility,
                "context": context
            },
            "output_labels": output_labels,
            "quality_labels": quality_labels
        }
    
    def _extract_song_state_at_time(
        self,
        song_analysis: Dict,
        time_sec: float,
        lookback_bars: int = 32,
        lookahead_bars: int = 0
    ) -> Dict:
        """
        Extract song state at a specific time point.
        Includes current features and recent history.
        """
        tempo = song_analysis.get("tempo", {}).get("bpm", 120)
        duration = song_analysis.get("metadata", {}).get("duration_sec", 0)
        
        # Clamp time to valid range
        time_sec = max(0, min(time_sec, duration))
        
        # Determine which section we're in
        section_info = self._get_section_at_time(song_analysis, time_sec)
        
        # Extract energy trend
        energy_curve = song_analysis.get("energy", {}).get("energy_curve", {})
        energy_trend = self._compute_energy_trend(
            energy_curve,
            time_sec,
            lookback_bars,
            tempo
        )
        
        # Extract recent energy history
        recent_energy = self._extract_recent_energy(
            energy_curve,
            time_sec,
            lookback_bars,
            tempo
        )
        
        # Extract recent spectral history
        recent_spectrum = self._extract_recent_spectrum(
            song_analysis.get("spectrum", {}),
            time_sec,
            lookback_bars,
            tempo
        )
        
        # Current features
        current_energy = self._get_value_at_time(
            energy_curve.get("values", []),
            energy_curve.get("times_sec", []),
            time_sec
        )
        
        spectral_centroid = self._get_value_at_time(
            song_analysis.get("spectrum", {}).get("spectral_curve", {}).get("values_hz", []),
            song_analysis.get("spectrum", {}).get("spectral_curve", {}).get("times_sec", []),
            time_sec
        )
        
        # Vocal presence
        vocals = song_analysis.get("vocals", {})
        has_vocals = self._has_vocals_at_time(vocals, time_sec)
        
        # Frequency band energies
        bands = song_analysis.get("spectrum", {}).get("frequency_bands", {})
        bass_energy = bands.get("bass_60_250", {}).get("energy_mean", 0.5)
        
        return {
            "bpm": tempo,
            "key": song_analysis.get("harmony", {}).get("key", {}).get("estimated_key", "C"),
            "mode": song_analysis.get("harmony", {}).get("key", {}).get("mode", "major"),
            "camelot": song_analysis.get("harmony", {}).get("key", {}).get("camelot", "1A"),
            "energy": current_energy or song_analysis.get("energy", {}).get("energy_statistics", {}).get("mean", 0.5),
            "section_type": section_info["type"],
            "bars_into_section": section_info["bars_into_section"],
            "bars_into_song": int((time_sec / 60) * tempo / 4),
            "has_vocals": has_vocals,
            "bass_energy": bass_energy,
            "spectral_centroid_hz": spectral_centroid or song_analysis.get("spectrum", {}).get("spectral_shape", {}).get("centroid_hz_mean", 2000),
            "energy_trend": energy_trend["direction"],
            "energy_slope": energy_trend["slope"],
            "last_32_bars_energy": recent_energy,
            "last_32_bars_spectrum": recent_spectrum,
            "time_sec": time_sec,
            "duration_sec": duration
        }
    
    def _get_section_at_time(self, song_analysis: Dict, time_sec: float) -> Dict:
        """Determine which section we're in at given time."""
        structure = song_analysis.get("structure", {})
        sections = structure.get("sections", [])
        
        if not sections:
            return {"type": "unknown", "bars_into_section": 0}
        
        for section in sections:
            start = section.get("start_sec", 0)
            end = section.get("end_sec", float('inf'))
            
            if start <= time_sec < end:
                bars_into = int(((time_sec - start) / 60) * 
                               song_analysis.get("tempo", {}).get("bpm", 120) / 4)
                return {
                    "type": section.get("type", "unknown"),
                    "bars_into_section": bars_into
                }
        
        # Default to last section
        last_section = sections[-1]
        return {
            "type": last_section.get("type", "unknown"),
            "bars_into_section": 0
        }
    
    def _compute_energy_trend(
        self,
        energy_curve: Dict,
        time_sec: float,
        lookback_bars: int,
        bpm: float
    ) -> Dict:
        """Compute energy trend (increasing, stable, decreasing)."""
        times = energy_curve.get("times_sec", [])
        values = energy_curve.get("values", [])
        
        if len(times) < 2:
            return {"direction": "stable", "slope": 0.0}
        
        # Look back over recent bars
        lookback_sec = (lookback_bars * 4 * 60) / bpm
        start_time = max(0, time_sec - lookback_sec)
        
        # Get values in this window
        window_values = [
            v for t, v in zip(times, values)
            if start_time <= t <= time_sec
        ]
        
        if len(window_values) < 2:
            return {"direction": "stable", "slope": 0.0}
        
        # Compute slope
        slope = (window_values[-1] - window_values[0]) / len(window_values)
        
        if slope > 0.01:
            direction = "increasing"
        elif slope < -0.01:
            direction = "decreasing"
        else:
            direction = "stable"
        
        return {"direction": direction, "slope": float(slope)}
    
    def _extract_recent_energy(
        self,
        energy_curve: Dict,
        time_sec: float,
        lookback_bars: int,
        bpm: float
    ) -> List[float]:
        """Extract energy values for recent bars."""
        times = energy_curve.get("times_sec", [])
        values = energy_curve.get("values", [])
        
        if not times or not values:
            return [0.5] * 32  # Default
        
        lookback_sec = (lookback_bars * 4 * 60) / bpm
        start_time = max(0, time_sec - lookback_sec)
        
        # Sample every bar
        bar_duration_sec = (4 * 60) / bpm
        n_samples = min(lookback_bars, 32)
        
        recent_energy = []
        for i in range(n_samples):
            sample_time = time_sec - (i * bar_duration_sec)
            value = self._get_value_at_time(values, times, sample_time)
            recent_energy.insert(0, value or 0.5)
        
        return recent_energy
    
    def _extract_recent_spectrum(
        self,
        spectrum_analysis: Dict,
        time_sec: float,
        lookback_bars: int,
        bpm: float
    ) -> Dict:
        """Extract recent spectral features."""
        spectral_curve = spectrum_analysis.get("spectral_curve", {})
        times = spectral_curve.get("times_sec", [])
        
        lookback_sec = (lookback_bars * 4 * 60) / bpm
        start_time = max(0, time_sec - lookback_sec)
        
        bar_duration_sec = (4 * 60) / bpm
        n_samples = min(lookback_bars, 32)
        
        recent_centroid = []
        for i in range(n_samples):
            sample_time = time_sec - (i * bar_duration_sec)
            centroid = self._get_value_at_time(
                spectral_curve.get("centroid_hz", []),
                times,
                sample_time
            )
            recent_centroid.insert(0, centroid or 2000)
        
        return {
            "spectral_centroid_hz": recent_centroid
        }
    
    def _has_vocals_at_time(self, vocals_analysis: Dict, time_sec: float) -> bool:
        """Check if vocals are present at given time."""
        timeline = vocals_analysis.get("vocal_timeline", {})
        times = timeline.get("times_sec", [])
        presence = timeline.get("vocal_presence", [])
        
        if not times or not presence:
            return vocals_analysis.get("has_vocals", False)
        
        value = self._get_value_at_time(presence, times, time_sec)
        return (value or 0) > 0.5
    
    def _get_value_at_time(
        self,
        values: List[float],
        times: List[float],
        time_sec: float
    ) -> Optional[float]:
        """Get interpolated value at specific time."""
        if not values or not times or len(values) != len(times):
            return None
        
        if time_sec <= times[0]:
            return values[0]
        if time_sec >= times[-1]:
            return values[-1]
        
        # Linear interpolation
        try:
            interp = interp1d(times, values, kind='linear', bounds_error=False, fill_value='extrapolate')
            return float(interp(time_sec))
        except:
            # Fallback: find nearest
            idx = np.argmin(np.abs(np.array(times) - time_sec))
            return values[idx]
    
    def _compute_compatibility(
        self,
        track_a_state: Dict,
        track_b_state: Dict
    ) -> Dict:
        """Compute compatibility metrics between two tracks."""
        bpm_a = track_a_state["bpm"]
        bpm_b = track_b_state["bpm"]
        bpm_diff = abs(bpm_b - bpm_a)
        bpm_ratio = bpm_b / bpm_a if bpm_a > 0 else 1.0
        
        # Camelot distance
        camelot_a = track_a_state["camelot"]
        camelot_b = track_b_state["camelot"]
        camelot_distance = self._camelot_distance(camelot_a, camelot_b)
        
        # Harmonic compatibility
        harmonic_compat = self._harmonic_compatibility(
            track_a_state["key"],
            track_a_state["mode"],
            track_b_state["key"],
            track_b_state["mode"]
        )
        
        # Energy delta
        energy_delta = track_b_state["energy"] - track_a_state["energy"]
        
        # Spectral similarity
        spectral_sim = 1.0 - abs(
            track_b_state["spectral_centroid_hz"] - track_a_state["spectral_centroid_hz"]
        ) / 5000.0  # Normalize by typical range
        spectral_sim = max(0, min(1, spectral_sim))
        
        # Vocal clash risk
        vocal_clash_risk = 1.0 if (track_a_state["has_vocals"] and track_b_state["has_vocals"]) else 0.0
        
        return {
            "bpm_difference": float(bpm_diff),
            "bpm_ratio": float(bpm_ratio),
            "camelot_distance": camelot_distance,
            "harmonic_compatibility": harmonic_compat,
            "energy_delta": float(energy_delta),
            "spectral_similarity": float(spectral_sim),
            "vocal_clash_risk": vocal_clash_risk,
            "sounds_good": harmonic_compat > 0.7 and camelot_distance <= 2
        }
    
    def _camelot_distance(self, camelot_a: str, camelot_b: str) -> int:
        """Compute Camelot wheel distance."""
        try:
            num_a = int(camelot_a[:-1])
            num_b = int(camelot_b[:-1])
            letter_a = camelot_a[-1]
            letter_b = camelot_b[-1]
            
            # Same number = perfect match
            if num_a == num_b:
                return 0 if letter_a == letter_b else 1
            
            # Different numbers
            diff = abs(num_b - num_a)
            if diff == 1 or diff == 11:  # Adjacent on wheel
                return 1
            elif diff == 2 or diff == 10:
                return 2
            else:
                return 3
        except:
            return 3  # Unknown = assume incompatible
    
    def _harmonic_compatibility(
        self,
        key_a: str,
        mode_a: str,
        key_b: str,
        mode_b: str
    ) -> float:
        """Compute harmonic compatibility score."""
        # Simplified: same key = 1.0, relative = 0.9, etc.
        if key_a == key_b and mode_a == mode_b:
            return 1.0
        elif key_a == key_b:
            return 0.8  # Relative major/minor
        else:
            return 0.6  # Different keys
    
    def _extract_output_labels(
        self,
        transition_analysis: Dict,
        transition_start_sec: float,
        transition_end_sec: float,
        track_a_analysis: Dict,
        track_b_analysis: Dict
    ) -> Dict:
        """Extract output labels (what the DJ did)."""
        duration_sec = transition_end_sec - transition_start_sec
        bpm_a = track_a_analysis.get("tempo", {}).get("bpm", 120)
        duration_bars = (duration_sec / 60) * bpm_a / 4
        
        # Extract volume curves and parameterize
        volume_curves = transition_analysis.get("volume_curves", {})
        param_volume = self._parameterize_volume_curves(volume_curves, duration_sec)
        
        # Extract EQ automation
        eq_automation = transition_analysis.get("eq_automation", {})
        param_eq = self._parameterize_eq_automation(eq_automation, duration_sec)
        
        # Beat matching
        beat_alignment = transition_analysis.get("beat_alignment", {})
        
        # Determine start bar (which bar of track A)
        start_bar = self._determine_start_bar(
            transition_start_sec,
            track_a_analysis
        )
        
        return {
            "timing": {
                "start_on_bar": start_bar,
                "start_on_downbeat": beat_alignment.get("aligned_on_downbeat", True),
                "duration_bars": int(duration_bars),
                "duration_sec": float(duration_sec)
            },
            "technique": {
                "primary": transition_analysis.get("technique_primary", "long_blend"),
                "secondary": transition_analysis.get("technique_secondary", []),
                "confidence": transition_analysis.get("technique_confidence", 0.7)
            },
            "volume_curves": param_volume,
            "eq_automation": param_eq,
            "beat_matching": {
                "pitch_shift_semitones": beat_alignment.get("pitch_shift_semitones", 0.0),
                "phase_offset_ms": beat_alignment.get("phase_offset_ms", 0.0),
                "align_on_downbeat": beat_alignment.get("aligned_on_downbeat", True),
                "maintain_alignment": True
            },
            "effects": transition_analysis.get("effects", {})
        }
    
    def _parameterize_volume_curves(
        self,
        volume_curves: Dict,
        duration_sec: float
    ) -> Dict:
        """Parameterize volume curves into smooth functions."""
        times = np.array(volume_curves.get("times_relative_sec", []))
        track_a_gain = np.array(volume_curves.get("track_a_gain_db", []))
        track_b_gain = np.array(volume_curves.get("track_b_gain_db", []))
        
        if len(times) < 2:
            # Default linear crossfade
            return {
                "track_a_curve_type": "linear",
                "track_a_params": {"start": 0, "end": -60},
                "track_b_curve_type": "linear",
                "track_b_params": {"start": -60, "end": 0},
                "crossfade_shape": "symmetric"
            }
        
        # Fit exponential curves
        try:
            # Track A: exponential decay
            a_params = self._fit_exponential_decay(times, track_a_gain)
            # Track B: exponential growth
            b_params = self._fit_exponential_growth(times, track_b_gain)
            
            return {
                "track_a_curve_type": "exponential",
                "track_a_params": {
                    "a": float(a_params[0]),
                    "b": float(a_params[1]),
                    "c": float(a_params[2])
                },
                "track_b_curve_type": "exponential",
                "track_b_params": {
                    "a": float(b_params[0]),
                    "b": float(b_params[1]),
                    "c": float(b_params[2])
                },
                "crossfade_shape": self._classify_crossfade_shape(times, track_a_gain, track_b_gain)
            }
        except:
            # Fallback to linear
            return {
                "track_a_curve_type": "linear",
                "track_a_params": {
                    "start": float(track_a_gain[0]),
                    "end": float(track_a_gain[-1])
                },
                "track_b_curve_type": "linear",
                "track_b_params": {
                    "start": float(track_b_gain[0]),
                    "end": float(track_b_gain[-1])
                },
                "crossfade_shape": "symmetric"
            }
    
    def _fit_exponential_decay(self, times: np.ndarray, values: np.ndarray) -> Tuple[float, float, float]:
        """Fit exponential decay: a * exp(-b * t) + c"""
        def exp_decay(t, a, b, c):
            return a * np.exp(-b * t) + c
        
        # Initial guess
        p0 = [values[0], 0.1, values[-1]]
        
        try:
            params, _ = curve_fit(exp_decay, times, values, p0=p0, maxfev=1000)
            return tuple(params)
        except:
            return (values[0], 0.1, values[-1])
    
    def _fit_exponential_growth(self, times: np.ndarray, values: np.ndarray) -> Tuple[float, float, float]:
        """Fit exponential growth: a * (1 - exp(-b * t)) + c"""
        def exp_growth(t, a, b, c):
            return a * (1 - np.exp(-b * t)) + c
        
        p0 = [values[-1] - values[0], 0.1, values[0]]
        
        try:
            params, _ = curve_fit(exp_growth, times, values, p0=p0, maxfev=1000)
            return tuple(params)
        except:
            return (values[-1] - values[0], 0.1, values[0])
    
    def _classify_crossfade_shape(
        self,
        times: np.ndarray,
        track_a_gain: np.ndarray,
        track_b_gain: np.ndarray
    ) -> str:
        """Classify crossfade shape."""
        # Check symmetry
        mid_point = len(times) // 2
        a_mid = track_a_gain[mid_point]
        b_mid = track_b_gain[mid_point]
        
        # If both are around -30dB at midpoint, symmetric
        if abs(a_mid - (-30)) < 5 and abs(b_mid - (-30)) < 5:
            return "symmetric"
        
        # If track A fades faster
        if a_mid < b_mid:
            return "track_a_faster"
        else:
            return "track_b_faster"
    
    def _parameterize_eq_automation(
        self,
        eq_automation: Dict,
        duration_sec: float
    ) -> Dict:
        """Parameterize EQ automation curves."""
        result = {}
        
        # Bass swap
        if eq_automation.get("bass_swap_detected", False):
            swap_point = eq_automation.get("bass_swap_point_sec", duration_sec / 2)
            track_a_curve = eq_automation.get("track_a_bass_curve_db", [])
            track_b_curve = eq_automation.get("track_b_bass_curve_db", [])
            
            result["bass_swap"] = {
                "enabled": True,
                "swap_point_sec": float(swap_point),
                "transition_speed": self._classify_transition_speed(swap_point, duration_sec),
                "track_a_cut_db": float(track_a_curve[-1]) if track_a_curve else -40.0,
                "track_b_cut_db": float(track_b_curve[0]) if track_b_curve else -40.0
            }
        else:
            result["bass_swap"] = {"enabled": False}
        
        # Highpass sweep
        if eq_automation.get("highpass_sweep_detected", False):
            freq_curve = eq_automation.get("highpass_freq_curve_hz", [])
            result["track_a_highpass"] = {
                "enabled": True,
                "start_freq_hz": float(freq_curve[0]) if freq_curve else 20.0,
                "end_freq_hz": float(freq_curve[-1]) if freq_curve else 2000.0
            }
        else:
            result["track_a_highpass"] = {"enabled": False}
        
        return result
    
    def _classify_transition_speed(self, swap_point_sec: float, duration_sec: float) -> str:
        """Classify transition speed."""
        ratio = swap_point_sec / duration_sec if duration_sec > 0 else 0.5
        
        if ratio < 0.3:
            return "quick"
        elif ratio < 0.7:
            return "moderate"
        else:
            return "slow"
    
    def _determine_start_bar(
        self,
        transition_start_sec: float,
        track_a_analysis: Dict
    ) -> int:
        """Determine which bar of track A the transition starts on."""
        bpm = track_a_analysis.get("tempo", {}).get("bpm", 120)
        bar_duration_sec = (4 * 60) / bpm
        bar_number = int(transition_start_sec / bar_duration_sec)
        return bar_number
    
    def _assess_quality(
        self,
        transition_analysis: Dict,
        track_a_state: Dict,
        track_b_state: Dict,
        compatibility: Dict
    ) -> Dict:
        """Assess transition quality."""
        quality_assessment = transition_analysis.get("quality_assessment", {})
        
        # Beat match quality
        beat_match = quality_assessment.get("beat_match_quality", 0.5)
        
        # Harmonic clash
        harmonic_clash = 1.0 - compatibility.get("harmonic_compatibility", 0.5)
        
        # Vocal clash
        vocal_clash = compatibility.get("vocal_clash_risk", 0.0)
        
        # Energy flow
        energy_flow = quality_assessment.get("energy_flow_smoothness", 0.5)
        
        # Spectral balance
        spectral_balance = quality_assessment.get("spectral_balance_maintained", 0.5)
        
        # Overall quality
        overall = (
            beat_match * 0.3 +
            (1 - harmonic_clash) * 0.2 +
            (1 - vocal_clash) * 0.2 +
            energy_flow * 0.15 +
            spectral_balance * 0.15
        )
        
        return {
            "beat_match_quality": float(beat_match),
            "harmonic_clash_score": float(harmonic_clash),
            "vocal_clash_score": float(vocal_clash),
            "energy_flow_smoothness": float(energy_flow),
            "spectral_balance_maintained": float(spectral_balance),
            "overall_transition_quality": float(overall),
            "failure_modes": {
                "beat_mismatch": beat_match < 0.7,
                "harmonic_clash": harmonic_clash > 0.3,
                "vocal_overlap": vocal_clash > 0.5,
                "energy_drop": compatibility.get("energy_delta", 0) < -0.3,
                "spectral_mud": spectral_balance < 0.6
            }
        }

