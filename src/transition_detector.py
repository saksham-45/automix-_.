"""
Transition Detection Module

Detects transitions in DJ mixes using multiple techniques:
- Novelty detection (self-similarity matrix)
- Beat grid discontinuity
- Source count estimation
- Volume curve analysis
"""

import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass


@dataclass
class DetectedTransition:
    """Represents a detected transition in a mix"""
    start_sec: float
    end_sec: float
    duration_sec: float
    confidence: float
    detection_method: str
    novelty_score: Optional[float] = None
    bpm_change: Optional[float] = None
    source_count_change: Optional[float] = None


class TransitionDetector:
    """Detects transitions in continuous DJ mixes"""
    
    def __init__(self, sr: int = 22050, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length
        
    def detect_all(self, audio_path: str) -> List[DetectedTransition]:
        """
        Detect all transitions using multiple methods and combine results
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of detected transitions
        """
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Run all detection methods
        novelty_transitions = self._detect_novelty(y, sr)
        tempo_transitions = self._detect_tempo_discontinuity(y, sr)
        
        # Combine and deduplicate
        all_transitions = novelty_transitions + tempo_transitions
        merged = self._merge_transitions(all_transitions)
        
        return merged
    
    def _detect_novelty(self, y: np.ndarray, sr: int) -> List[DetectedTransition]:
        """
        Detect transitions using novelty detection (self-similarity matrix)
        
        This method looks for sudden changes in the audio's self-similarity,
        which typically occur at track boundaries.
        """
        # Compute chromagram (pitch content)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        
        # Build self-similarity matrix
        S = librosa.segment.recurrence_matrix(
            chroma, 
            mode='affinity',
            metric='cosine',
            sparse=False
        )
        
        # Compute novelty curve
        novelty = librosa.segment.novelty(S)
        
        # Normalize novelty
        novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-8)
        
        # Peak picking to find transition points
        peaks = librosa.util.peak_pick(
            novelty,
            pre_max=10,
            post_max=10,
            pre_avg=10,
            post_avg=10,
            delta=0.1,
            wait=50  # Minimum frames between peaks
        )
        
        # Convert to time
        times = librosa.frames_to_time(peaks, sr=sr, hop_length=self.hop_length)
        
        transitions = []
        for i, peak_time in enumerate(times):
            # Estimate transition duration (typically 8-32 bars)
            # Use next peak or end of audio as boundary
            if i < len(times) - 1:
                end_time = min(peak_time + 32, times[i + 1])
            else:
                end_time = min(peak_time + 32, len(y) / sr)
            
            transitions.append(DetectedTransition(
                start_sec=peak_time,
                end_sec=end_time,
                duration_sec=end_time - peak_time,
                confidence=float(novelty[peaks[i]]),
                detection_method="novelty",
                novelty_score=float(novelty[peaks[i]])
            ))
        
        return transitions
    
    def _detect_tempo_discontinuity(self, y: np.ndarray, sr: int) -> List[DetectedTransition]:
        """
        Detect transitions by finding tempo discontinuities
        
        During transitions, the underlying tempo may change as one track
        fades out and another fades in.
        """
        # Compute onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        
        # Sliding window tempo estimation
        window_sec = 10
        hop_sec = 2
        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)
        
        tempo_curve = []
        time_points = []
        
        for start in range(0, len(y) - window_samples, hop_samples):
            segment = y[start:start + window_samples]
            try:
                tempo, _ = librosa.beat.beat_track(
                    y=segment, 
                    sr=sr,
                    hop_length=self.hop_length,
                    start_bpm=120,
                    std_bpm=30
                )
                tempo_curve.append(tempo)
                time_points.append(start / sr)
            except:
                tempo_curve.append(np.nan)
                time_points.append(start / sr)
        
        tempo_curve = np.array(tempo_curve)
        time_points = np.array(time_points)
        
        # Find significant tempo changes
        valid_mask = ~np.isnan(tempo_curve)
        if valid_mask.sum() < 2:
            return []
        
        tempo_diff = np.abs(np.diff(tempo_curve[valid_mask]))
        time_diff = np.diff(time_points[valid_mask])
        
        # Normalize by time difference
        tempo_change_rate = tempo_diff / (time_diff + 1e-8)
        
        # Threshold for significant change
        threshold = np.percentile(tempo_change_rate, 85)
        
        transitions = []
        for i, change_rate in enumerate(tempo_change_rate):
            if change_rate > threshold:
                transition_time = time_points[valid_mask][i]
                transitions.append(DetectedTransition(
                    start_sec=transition_time,
                    end_sec=min(transition_time + 32, len(y) / sr),
                    duration_sec=32.0,
                    confidence=min(1.0, change_rate / threshold),
                    detection_method="tempo_discontinuity",
                    bpm_change=float(tempo_diff[i])
                ))
        
        return transitions
    
    def _merge_transitions(self, transitions: List[DetectedTransition]) -> List[DetectedTransition]:
        """
        Merge overlapping transitions detected by different methods
        """
        if not transitions:
            return []
        
        # Sort by start time
        transitions.sort(key=lambda t: t.start_sec)
        
        merged = []
        current = transitions[0]
        
        for next_trans in transitions[1:]:
            # If transitions overlap or are close (< 10 sec apart), merge them
            if next_trans.start_sec <= current.end_sec + 10:
                # Merge: take union of time ranges, max confidence
                current.end_sec = max(current.end_sec, next_trans.end_sec)
                current.duration_sec = current.end_sec - current.start_sec
                current.confidence = max(current.confidence, next_trans.confidence)
                
                # Combine detection methods
                if current.detection_method != next_trans.detection_method:
                    current.detection_method = f"{current.detection_method}+{next_trans.detection_method}"
            else:
                merged.append(current)
                current = next_trans
        
        merged.append(current)
        
        return merged
    
    def refine_transition_boundaries(
        self, 
        audio_path: str, 
        transition: DetectedTransition
    ) -> DetectedTransition:
        """
        Refine transition boundaries using beat alignment
        
        Finds the nearest downbeat for cleaner transitions
        """
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Find beats around transition
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length
        )
        
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
        
        # Find nearest beat to transition start
        start_idx = np.argmin(np.abs(beat_times - transition.start_sec))
        refined_start = beat_times[start_idx]
        
        # Find nearest beat to transition end
        end_idx = np.argmin(np.abs(beat_times - transition.end_sec))
        refined_end = beat_times[end_idx]
        
        return DetectedTransition(
            start_sec=refined_start,
            end_sec=refined_end,
            duration_sec=refined_end - refined_start,
            confidence=transition.confidence,
            detection_method=transition.detection_method,
            novelty_score=transition.novelty_score,
            bpm_change=transition.bpm_change
        )

