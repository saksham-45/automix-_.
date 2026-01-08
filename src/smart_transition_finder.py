"""
Smart Transition Point Finder

Finds optimal transition points in both songs for smooth, beat-matched blending.
Analyzes song structure, energy, and beat positions to find best "out" and "in" points.
"""
import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from src.structure_analyzer import StructureAnalyzer
from src.harmonic_analyzer import HarmonicAnalyzer


@dataclass
class TransitionPoint:
    """A candidate transition point in a song."""
    time_sec: float
    beat_aligned: bool
    energy: float
    energy_trend: str  # 'rising', 'falling', 'stable', 'dip'
    structural_label: str  # 'intro', 'verse', 'chorus', 'breakdown', 'outro', 'unknown'
    score: float  # Overall quality score
    beat_position: int  # Beat number at this point


@dataclass
class TransitionPair:
    """A pair of transition points (outgoing + incoming)."""
    song_a_point: TransitionPoint
    song_b_point: TransitionPoint
    compatibility_score: float
    tempo_match: bool
    key_match: bool
    beat_aligned: bool


class SmartTransitionFinder:
    """
    Finds optimal transition points in both songs.
    """
    
    def __init__(self, sr: int = 44100, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length
        self.structure_analyzer = StructureAnalyzer(sr=sr, hop_length=hop_length)
        self.harmonic_analyzer = HarmonicAnalyzer()
    
    def find_best_transition_pair(self,
                                  song_a_path: str,
                                  song_b_path: str,
                                  song_a_analysis: Optional[Dict] = None,
                                  song_b_analysis: Optional[Dict] = None) -> TransitionPair:
        """
        Find the best transition pair between two songs.
        
        Args:
            song_a_path: Path to outgoing song
            song_b_path: Path to incoming song
            song_a_analysis: Pre-computed analysis (optional)
            song_b_analysis: Pre-computed analysis (optional)
        
        Returns:
            Best TransitionPair for smooth blending
        """
        # Load audio
        y_a, sr_a = librosa.load(song_a_path, sr=self.sr)
        y_b, sr_b = librosa.load(song_b_path, sr=self.sr)
        
        # Find candidate points in both songs
        print("Analyzing Song A for transition out points...")
        song_a_points = self._find_transition_points(y_a, sr_a, is_outgoing=True)
        
        print("Analyzing Song B for transition in points...")
        song_b_points = self._find_transition_points(y_b, sr_b, is_outgoing=False)
        
        # Get tempo/key for compatibility scoring
        tempo_a = self._get_tempo(y_a, sr_a)
        tempo_b = self._get_tempo(y_b, sr_b)
        key_a = self._get_key(song_a_analysis) if song_a_analysis else None
        key_b = self._get_key(song_b_analysis) if song_b_analysis else None
        
        # Score all pairs and find best
        print("Scoring transition pairs...")
        best_pair = None
        best_score = -1
        
        for point_a in song_a_points[:20]:  # Top 20 from song A
            for point_b in song_b_points[:20]:  # Top 20 from song B
                pair_score = self._score_transition_pair(
                    point_a, point_b,
                    tempo_a, tempo_b,
                    key_a, key_b
                )
                
                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = TransitionPair(
                        song_a_point=point_a,
                        song_b_point=point_b,
                        compatibility_score=pair_score,
                        tempo_match=abs(tempo_a - tempo_b) < 5,  # Within 5 BPM
                        key_match=self._keys_compatible(key_a, key_b) if key_a and key_b else True,
                        beat_aligned=point_a.beat_aligned and point_b.beat_aligned
                    )
        
        if best_pair is None:
            # Fallback: use last 30s of A, first 30s of B
            duration_a = len(y_a) / sr_a
            duration_b = len(y_b) / sr_b
            fallback_a = TransitionPoint(
                time_sec=max(0, duration_a - 30),
                beat_aligned=True,
                energy=0.5,
                energy_trend='stable',
                structural_label='outro',
                score=0.5,
                beat_position=0
            )
            fallback_b = TransitionPoint(
                time_sec=min(30, duration_b * 0.1),
                beat_aligned=True,
                energy=0.5,
                energy_trend='rising',
                structural_label='intro',
                score=0.5,
                beat_position=0
            )
            best_pair = TransitionPair(
                song_a_point=fallback_a,
                song_b_point=fallback_b,
                compatibility_score=0.5,
                tempo_match=True,
                key_match=True,
                beat_aligned=True
            )
        
        print(f"\n✓ Best transition found:")
        print(f"  Song A @ {best_pair.song_a_point.time_sec:.1f}s ({best_pair.song_a_point.structural_label})")
        print(f"  Song B @ {best_pair.song_b_point.time_sec:.1f}s ({best_pair.song_b_point.structural_label})")
        print(f"  Score: {best_pair.compatibility_score:.3f}")
        
        return best_pair
    
    def _find_transition_points(self,
                                y: np.ndarray,
                                sr: int,
                                is_outgoing: bool) -> List[TransitionPoint]:
        """
        Find candidate transition points in a song.
        
        Args:
            y: Audio signal
            sr: Sample rate
            is_outgoing: True for outgoing song (find out points), False for incoming (find in points)
        
        Returns:
            List of candidate TransitionPoints, sorted by score (best first)
        """
        duration = len(y) / sr
        
        # Analyze structure
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)
        
        # Extract energy curve
        energy = np.abs(y)
        if y.ndim > 1:
            energy = np.mean(energy, axis=1)
        
        # Frame energy (smoother)
        frame_length = int(sr * 0.5)  # 0.5 second frames
        hop = frame_length // 2
        n_frames = (len(energy) - frame_length) // hop + 1
        frame_energy = np.zeros(n_frames)
        frame_times = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * hop
            end = start + frame_length
            frame_energy[i] = np.mean(energy[start:end])
            frame_times[i] = (start + frame_length // 2) / sr
        
        # Find structural segments
        segments = self._find_structural_segments(y, sr, frame_times, frame_energy)
        
        # Find good transition points based on type
        candidates = []
        
        if is_outgoing:
            # Good "out" points: endings, breakdowns, energy dips, last 50% of song
            search_start = duration * 0.3  # Start searching from 30% into song
            search_end = duration * 0.95   # Up to 95%
        else:
            # Good "in" points: intros, energy rises, first 70% of song
            search_start = 0
            search_end = duration * 0.7
        
        # Sample points at beat positions in search region
        search_beats = [bt for bt in beat_times if search_start <= bt <= search_end]
        
        for beat_time in search_beats[::4]:  # Every 4th beat (bar boundaries)
            beat_idx = np.argmin(np.abs(frame_times - beat_time))
            if beat_idx >= len(frame_energy):
                continue
            
            energy_val = frame_energy[beat_idx]
            
            # Energy trend
            window = 8  # frames
            if beat_idx >= window and beat_idx < len(frame_energy) - window:
                energy_before = np.mean(frame_energy[beat_idx-window:beat_idx])
                energy_after = np.mean(frame_energy[beat_idx:beat_idx+window])
                if energy_after > energy_before * 1.2:
                    trend = 'rising'
                elif energy_after < energy_before * 0.8:
                    trend = 'falling'
                elif energy_after < energy_before * 0.6:
                    trend = 'dip'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            # Structural label
            label = self._get_segment_label(beat_time, segments, duration, is_outgoing)
            
            # Score this point
            score = self._score_transition_point_candidate(
                beat_time, energy_val, trend, label,
                is_outgoing, duration, beat_times
            )
            
            if score > 0.3:  # Threshold
                beat_pos = np.argmin(np.abs(beat_times - beat_time))
                candidates.append(TransitionPoint(
                    time_sec=beat_time,
                    beat_aligned=True,
                    energy=float(energy_val),
                    energy_trend=trend,
                    structural_label=label,
                    score=float(score),
                    beat_position=beat_pos
                ))
        
        # Sort by score (best first)
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        return candidates[:30]  # Return top 30 candidates
    
    def _find_structural_segments(self,
                                  y: np.ndarray,
                                  sr: int,
                                  frame_times: np.ndarray,
                                  frame_energy: np.ndarray) -> List[Dict]:
        """Find structural segments (intro, verse, chorus, etc.) using energy and novelty."""
        # Simple structural segmentation based on energy changes
        duration = len(y) / sr
        
        # Divide song into equal segments based on energy variations
        n_segments = min(8, max(4, int(duration / 30)))  # ~30 seconds per segment
        segment_duration = duration / n_segments
        
        segments = []
        for i in range(n_segments):
            start = i * segment_duration
            end = (i + 1) * segment_duration if i < n_segments - 1 else duration
            mid = (start + end) / 2
            
            # Get average energy in this segment
            start_frame = np.argmin(np.abs(frame_times - start))
            end_frame = np.argmin(np.abs(frame_times - min(end, frame_times[-1])))
            
            if start_frame < len(frame_energy) and end_frame < len(frame_energy):
                avg_energy = np.mean(frame_energy[start_frame:end_frame+1])
            else:
                avg_energy = 0.5
            
            segments.append({
                'start': start,
                'end': end,
                'mid': mid,
                'energy': float(avg_energy)
            })
        
        return segments
    
    def _get_segment_label(self,
                          time_sec: float,
                          segments: List[Dict],
                          duration: float,
                          is_outgoing: bool) -> str:
        """Label a segment based on position and energy."""
        if is_outgoing:
            # For outgoing: prefer endings, breakdowns
            if time_sec > duration * 0.85:
                return 'outro'
            elif time_sec > duration * 0.7:
                return 'breakdown'
            else:
                # Check energy in nearby segment
                for seg in segments:
                    if seg['start'] <= time_sec <= seg['end']:
                        if seg['energy'] < 0.3:
                            return 'breakdown'
                return 'verse'
        else:
            # For incoming: prefer intros, build-ups
            if time_sec < duration * 0.15:
                return 'intro'
            elif time_sec < duration * 0.3:
                # Check if it's a build-up
                for seg in segments:
                    if seg['start'] <= time_sec <= seg['end']:
                        if seg['energy'] > 0.7:
                            return 'build'
                return 'intro'
            else:
                return 'verse'
    
    def _score_transition_point_candidate(self,
                                         time_sec: float,
                                         energy: float,
                                         trend: str,
                                         label: str,
                                         is_outgoing: bool,
                                         duration: float,
                                         beat_times: np.ndarray) -> float:
        """Score a single transition point candidate."""
        score = 0.5  # Base score
        
        if is_outgoing:
            # Good out points: low energy, falling trend, endings
            if label in ['outro', 'breakdown']:
                score += 0.3
            if trend in ['falling', 'dip']:
                score += 0.2
            if energy < 0.4:
                score += 0.2
            # Prefer later in song (but not very end)
            position_score = (time_sec / duration) * 0.5
            if 0.4 < position_score < 0.9:
                score += position_score * 0.3
        else:
            # Good in points: rising energy, intros, build-ups
            if label in ['intro', 'build']:
                score += 0.3
            if trend == 'rising':
                score += 0.2
            if 0.3 < energy < 0.8:  # Not too quiet, not too loud
                score += 0.2
            # Prefer earlier in song
            position_score = 1.0 - (time_sec / duration)
            if position_score > 0.5:
                score += position_score * 0.3
        
        # Beat alignment bonus
        score += 0.1  # Already beat-aligned
        
        return min(1.0, score)
    
    def _score_transition_pair(self,
                              point_a: TransitionPoint,
                              point_b: TransitionPoint,
                              tempo_a: float,
                              tempo_b: float,
                              key_a: Optional[str],
                              key_b: Optional[str]) -> float:
        """Score a transition pair for compatibility."""
        score = 0.0
        
        # Tempo compatibility (within 5 BPM is good)
        tempo_diff = abs(tempo_a - tempo_b)
        if tempo_diff < 2:
            score += 0.4
        elif tempo_diff < 5:
            score += 0.3
        elif tempo_diff < 10:
            score += 0.1
        
        # Key compatibility
        if key_a and key_b and self._keys_compatible(key_a, key_b):
            score += 0.2
        
        # Energy flow (outgoing should fade, incoming should rise)
        if point_a.energy_trend in ['falling', 'dip'] and point_b.energy_trend == 'rising':
            score += 0.2
        elif point_a.energy_trend in ['falling', 'stable'] and point_b.energy_trend != 'falling':
            score += 0.1
        
        # Structural compatibility
        if point_a.structural_label in ['outro', 'breakdown'] and point_b.structural_label in ['intro', 'build']:
            score += 0.2
        
        # Both beat-aligned
        if point_a.beat_aligned and point_b.beat_aligned:
            score += 0.1
        
        # Individual point quality
        score += (point_a.score + point_b.score) * 0.2
        
        return score
    
    def _get_tempo(self, y: np.ndarray, sr: int) -> float:
        """Extract tempo from audio."""
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        return float(tempo)
    
    def _get_key(self, analysis: Optional[Dict]) -> Optional[str]:
        """Extract key from analysis."""
        if analysis and 'harmony' in analysis:
            key = analysis['harmony'].get('key', 'C')
            if isinstance(key, dict):
                key = key.get('key', 'C')
            elif isinstance(key, list) and len(key) > 0:
                key = key[0]
            return str(key) if isinstance(key, str) else 'C'
        return None
    
    def _keys_compatible(self, key_a: str, key_b: str) -> bool:
        """Check if two keys are compatible for mixing."""
        key_to_idx = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }
        
        # Strip minor/major
        key_a_base = key_a.replace('m', '').replace('M', '').replace('min', '').replace('maj', '')
        key_b_base = key_b.replace('m', '').replace('M', '').replace('min', '').replace('maj', '')
        
        idx_a = key_to_idx.get(key_a_base, 0)
        idx_b = key_to_idx.get(key_b_base, 0)
        
        diff = abs(idx_a - idx_b)
        compatible = [0, 5, 7, 3, 4, 8, 9]  # Same, perfect 4th/5th, etc.
        
        return diff in compatible or (12 - diff) in compatible

