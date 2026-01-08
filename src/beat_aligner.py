"""
Beat Alignment Module

Aligns beats between two songs for smooth rhythmic transitions.
"""
import numpy as np
import librosa
from typing import Tuple, List, Optional


class BeatAligner:
    """
    Aligns beats between two songs for smooth transitions.
    """
    
    def __init__(self, sr: int = 44100, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length
    
    def align_beats(self,
                    y_a: np.ndarray,
                    y_b: np.ndarray,
                    sr: int,
                    point_a_sec: float,
                    point_b_sec: float) -> Tuple[float, float]:
        """
        Align beats between two songs at transition points.
        
        Args:
            y_a: Audio from song A
            y_b: Audio from song B
            sr: Sample rate
            point_a_sec: Transition point in song A (seconds)
            point_b_sec: Transition point in song B (seconds)
        
        Returns:
            (aligned_point_a_sec, aligned_point_b_sec) - Beat-aligned points
        """
        # Get beat positions for both songs
        tempo_a, beats_a = librosa.beat.beat_track(y=y_a, sr=sr, hop_length=self.hop_length)
        tempo_b, beats_b = librosa.beat.beat_track(y=y_b, sr=sr, hop_length=self.hop_length)
        
        beat_times_a = librosa.frames_to_time(beats_a, sr=sr, hop_length=self.hop_length)
        beat_times_b = librosa.frames_to_time(beats_b, sr=sr, hop_length=self.hop_length)
        
        # Find nearest beat to transition points
        aligned_a = self._find_nearest_beat(point_a_sec, beat_times_a)
        aligned_b = self._find_nearest_beat(point_b_sec, beat_times_b)
        
        # If tempos are close, try to align to same beat phase (e.g., both on downbeat)
        if abs(tempo_a - tempo_b) < 5:
            # Find downbeats (every 4th beat)
            aligned_a = self._align_to_downbeat(aligned_a, beat_times_a)
            aligned_b = self._align_to_downbeat(aligned_b, beat_times_b)
        
        return aligned_a, aligned_b
    
    def _find_nearest_beat(self, time_sec: float, beat_times: np.ndarray) -> float:
        """Find the nearest beat to a given time."""
        if len(beat_times) == 0:
            return time_sec
        
        distances = np.abs(beat_times - time_sec)
        nearest_idx = np.argmin(distances)
        
        # Only snap if within 0.2 seconds (200ms)
        if distances[nearest_idx] < 0.2:
            return float(beat_times[nearest_idx])
        else:
            return time_sec
    
    def _align_to_downbeat(self, time_sec: float, beat_times: np.ndarray) -> float:
        """Align to the nearest downbeat (every 4th beat)."""
        if len(beat_times) < 4:
            return time_sec
        
        distances = np.abs(beat_times - time_sec)
        nearest_idx = np.argmin(distances)
        
        # Find nearest downbeat (every 4th beat)
        downbeat_idx = (nearest_idx // 4) * 4
        if downbeat_idx < len(beat_times):
            return float(beat_times[downbeat_idx])
        
        return time_sec
    
    def get_beat_grid(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
        """
        Get beat grid for a song.
        
        Returns:
            (beat_times, tempo) - Beat positions in seconds and tempo
        """
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
        return beat_times, float(tempo)

