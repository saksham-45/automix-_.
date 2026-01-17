"""
Smart Transition Point Finder

Finds optimal transition points in both songs for smooth, beat-matched blending.
Analyzes song structure, energy, and beat positions to find best "out" and "in" points.
"""
import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks

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
    quality_factors: Optional[Dict] = None  # Quality prediction factors (vocal_overlap_risk, etc.)


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
        Now uses intelligent multi-candidate evaluation with quality prediction.
        
        Args:
            song_a_path: Path to outgoing song
            song_b_path: Path to incoming song
            song_a_analysis: Pre-computed analysis (optional)
            song_b_analysis: Pre-computed analysis (optional)
        
        Returns:
            Best TransitionPair for smooth blending
        """
        # Use intelligent evaluation by default
        return self.find_best_transition_pair_intelligent(
            song_a_path, song_b_path, song_a_analysis, song_b_analysis
        )
    
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
        
        # Find structural segments (enhanced for full song analysis)
        segments = self._find_structural_segments_intelligent(y, sr, frame_times, frame_energy, duration)
        
        # Find good transition points based on type
        candidates = []
        
        if is_outgoing:
            # Good "out" points: endings, breakdowns, energy dips
            # User requirement: transition out should be at least 3/4 (75%) through the song
            search_start = duration * 0.75  # Start searching from 75% into song (3/4 point)
            search_end = duration * 0.95   # Up to 95%
        else:
            # Good "in" points: can be ANYWHERE in song (intros, build-ups, drops, choruses)
            # Allow transitions INTO any section of Song B, not just early parts
            search_start = 0
            search_end = duration * 0.95  # Search almost entire song (leave small buffer at very end)
        
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
        """Legacy method - redirects to intelligent version."""
        duration = len(y) / sr
        return self._find_structural_segments_intelligent(y, sr, frame_times, frame_energy, duration)
    
    def _find_structural_segments_intelligent(self,
                                             y: np.ndarray,
                                             sr: int,
                                             frame_times: np.ndarray,
                                             frame_energy: np.ndarray,
                                             duration: float) -> List[Dict]:
        """
        Intelligent structural segmentation that adapts to song complexity.
        For simple songs: full analysis
        For complex songs: strategic sampling + key point detection
        """
        # Detect structural changes using energy variations and novelty
        # Use a sliding window to detect significant changes
        
        # Compute energy variance in sliding windows
        window_size = max(10, int(4 / (frame_times[1] - frame_times[0]) if len(frame_times) > 1 else 10))  # ~4 second windows
        energy_smooth = np.convolve(frame_energy, np.ones(window_size) / window_size, mode='same')
        
        # Detect peaks and valleys (section boundaries)
        # Find energy peaks (potential chorus/drop sections)
        peaks, _ = find_peaks(energy_smooth, distance=max(5, window_size // 2), prominence=0.1)
        
        # Find energy valleys (potential breakdowns/verses)
        valleys, _ = find_peaks(-energy_smooth, distance=max(5, window_size // 2), prominence=0.1)
        
        # Combine peaks and valleys to create segments
        key_points = sorted(set([0, len(frame_times) - 1] + peaks.tolist() + valleys.tolist()))
        
        segments = []
        for i in range(len(key_points) - 1):
            start_idx = key_points[i]
            end_idx = key_points[i + 1]
            
            start_time = frame_times[start_idx] if start_idx < len(frame_times) else 0
            end_time = frame_times[end_idx] if end_idx < len(frame_times) else duration
            mid_time = (start_time + end_time) / 2
            
            # Get average energy in this segment
            if start_idx < len(frame_energy) and end_idx <= len(frame_energy):
                avg_energy = np.mean(frame_energy[start_idx:end_idx])
            else:
                avg_energy = 0.5
            
            segments.append({
                'start': float(start_time),
                'end': float(end_time),
                'mid': float(mid_time),
                'energy': float(avg_energy)
            })
        
        # If we didn't find enough segments, fall back to equal division
        if len(segments) < 4:
            n_segments = min(8, max(4, int(duration / 30)))
            segment_duration = duration / n_segments
            
            segments = []
            for i in range(n_segments):
                start = i * segment_duration
                end = (i + 1) * segment_duration if i < n_segments - 1 else duration
                mid = (start + end) / 2
                
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
            # For incoming: can be intros, build-ups, drops, choruses, verses - anywhere!
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
                # Later in song - could be drop, chorus, verse, etc.
                # Check energy in segment to determine
                for seg in segments:
                    if seg['start'] <= time_sec <= seg['end']:
                        if seg['energy'] > 0.75:
                            return 'drop'  # High energy = drop/chorus
                        elif seg['energy'] < 0.4:
                            return 'breakdown'  # Low energy = breakdown/verse
                return 'verse'  # Default to verse
    
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
            # STRONGLY prefer later positions - must be at least 3/4 (75%) through song
            position_ratio = time_sec / duration
            if position_ratio < 0.75:  # Penalize early positions (< 75%)
                score -= 0.5  # Heavy penalty for being too early
            elif position_ratio >= 0.75 and position_ratio < 0.95:  # Reward 75-95%
                # Bonus increases with position (later = better, but not too close to end)
                position_bonus = (position_ratio - 0.75) / 0.2 * 0.4  # 0 to 0.4 bonus
                score += position_bonus
        else:
            # Good in points: rising energy, intros, build-ups, drops, choruses
            # Don't bias toward early positions - allow transitions into any section
            if label in ['intro', 'build', 'drop', 'chorus']:
                score += 0.3
            if trend == 'rising':
                score += 0.2
            if 0.3 < energy < 0.8:  # Not too quiet, not too loud
                score += 0.2
            # Slight preference for structural sections (intro/build/drop) regardless of position
            if label in ['intro', 'build', 'drop']:
                score += 0.1  # Bonus for structural sections anywhere in song
        
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
    
    def find_best_transition_pair_intelligent(self,
                                             song_a_path: str,
                                             song_b_path: str,
                                             song_a_analysis: Optional[Dict] = None,
                                             song_b_analysis: Optional[Dict] = None) -> TransitionPair:
        """
        Enhanced version with multi-candidate evaluation and quality prediction.
        Tries top candidates from each song and predicts quality before committing.
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
        
        # Evaluate TOP 10 candidates from each song (not 20x20 brute force)
        print("Evaluating top candidates with quality prediction...")
        top_candidates_a = song_a_points[:10]
        top_candidates_b = song_b_points[:10]
        
        evaluated_pairs = []
        for point_a in top_candidates_a:
            for point_b in top_candidates_b:
                # Predict quality BEFORE creating transition
                quality_score, quality_factors = self._predict_transition_quality(
                    point_a, point_b,
                    y_a, y_b,
                    tempo_a, tempo_b,
                    key_a, key_b,
                    song_a_analysis, song_b_analysis
                )
                
                # Combine basic compatibility with predicted quality
                basic_score = self._score_transition_pair(
                    point_a, point_b,
                    tempo_a, tempo_b,
                    key_a, key_b
                )
                
                # Weighted combination: 60% quality prediction, 40% basic compatibility
                combined_score = quality_score * 0.6 + basic_score * 0.4
                
                evaluated_pairs.append({
                    'point_a': point_a,
                    'point_b': point_b,
                    'quality_score': quality_score,
                    'basic_score': basic_score,
                    'combined_score': combined_score,
                    'quality_factors': quality_factors
                })
        
        # Sort by combined score (best first)
        evaluated_pairs.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top 3 candidates for final selection
        if not evaluated_pairs:
            # Fallback to original method
            return self.find_best_transition_pair(song_a_path, song_b_path, song_a_analysis, song_b_analysis)
        
        # Use best candidate
        best = evaluated_pairs[0]
        
        best_pair = TransitionPair(
            song_a_point=best['point_a'],
            song_b_point=best['point_b'],
            compatibility_score=best['combined_score'],
            tempo_match=abs(tempo_a - tempo_b) < 5,
            key_match=self._keys_compatible(key_a, key_b) if key_a and key_b else True,
            beat_aligned=best['point_a'].beat_aligned and best['point_b'].beat_aligned,
            quality_factors=best['quality_factors']  # Store quality factors for use in mixer
        )
        
        print(f"\n✓ Best transition found (with quality prediction):")
        print(f"  Song A @ {best_pair.song_a_point.time_sec:.1f}s ({best_pair.song_a_point.structural_label})")
        print(f"  Song B @ {best_pair.song_b_point.time_sec:.1f}s ({best_pair.song_b_point.structural_label})")
        print(f"  Predicted Quality: {best['quality_score']:.3f}")
        print(f"  Combined Score: {best['combined_score']:.3f}")
        print(f"  Quality Factors:")
        for factor, value in best['quality_factors'].items():
            print(f"    - {factor}: {value:.3f}")
        
        return best_pair
    
    def _predict_transition_quality(self,
                                   point_a: TransitionPoint,
                                   point_b: TransitionPoint,
                                   y_a: np.ndarray,
                                   y_b: np.ndarray,
                                   tempo_a: float,
                                   tempo_b: float,
                                   key_a: Optional[str],
                                   key_b: Optional[str],
                                   analysis_a: Optional[Dict],
                                   analysis_b: Optional[Dict]) -> Tuple[float, Dict]:
        """
        Predict if a transition will sound good BEFORE creating it.
        Returns (overall_quality_score, quality_factors_dict)
        """
        quality_factors = {}
        
        # 1. Harmonic compatibility
        if key_a and key_b:
            if self._keys_compatible(key_a, key_b):
                quality_factors['harmonic_compatibility'] = 1.0
            else:
                # Calculate dissonance level
                key_to_idx = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                             'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
                key_a_base = key_a.replace('m', '').replace('M', '').replace('min', '').replace('maj', '')
                key_b_base = key_b.replace('m', '').replace('M', '').replace('min', '').replace('maj', '')
                idx_a = key_to_idx.get(key_a_base, 0)
                idx_b = key_to_idx.get(key_b_base, 0)
                diff = min(abs(idx_a - idx_b), 12 - abs(idx_a - idx_b))
                quality_factors['harmonic_compatibility'] = max(0.3, 1.0 - (diff / 6.0))
        else:
            quality_factors['harmonic_compatibility'] = 0.7  # Unknown is moderate
        
        # 2. Energy compatibility
        energy_comp = self._score_energy_match(point_a, point_b)
        quality_factors['energy_compatibility'] = energy_comp
        
        # 3. Structural compatibility
        struct_comp = self._score_structural_match(point_a, point_b)
        quality_factors['structural_compatibility'] = struct_comp
        
        # 4. Tempo/phase match prediction
        tempo_phase = self._score_tempo_phase_match(point_a, point_b, tempo_a, tempo_b)
        quality_factors['tempo_phase_match'] = tempo_phase
        
        # 5. Spectral clash risk prediction
        spectral_clash = self._predict_spectral_clash(point_a, point_b, y_a, y_b)
        quality_factors['spectral_clash_risk'] = spectral_clash  # Lower is better
        
        # 6. Vocal overlap risk prediction
        vocal_overlap = self._predict_vocal_overlap(point_a, point_b, y_a, y_b)
        quality_factors['vocal_overlap_risk'] = vocal_overlap  # Lower is better
        
        # 7. Beat alignment quality
        beat_align = 1.0 if (point_a.beat_aligned and point_b.beat_aligned) else 0.6
        quality_factors['beat_alignment_quality'] = beat_align
        
        # Weighted quality score
        weights = {
            'harmonic_compatibility': 0.20,
            'energy_compatibility': 0.15,
            'structural_compatibility': 0.15,
            'tempo_phase_match': 0.20,
            'spectral_clash_risk': 0.15,  # Lower is better
            'vocal_overlap_risk': 0.10,   # Lower is better
            'beat_alignment_quality': 0.05
        }
        
        overall = sum(
            quality_factors[k] * weights[k] 
            if 'risk' not in k else (1 - quality_factors[k]) * weights[k]
            for k in weights
        )
        
        return float(overall), quality_factors
    
    def _score_energy_match(self, point_a: TransitionPoint, point_b: TransitionPoint) -> float:
        """Score energy compatibility between two points."""
        # Ideal: outgoing fading, incoming rising
        if point_a.energy_trend in ['falling', 'dip'] and point_b.energy_trend == 'rising':
            return 1.0
        elif point_a.energy_trend in ['falling', 'stable'] and point_b.energy_trend != 'falling':
            return 0.8
        elif point_a.energy_trend == 'stable' and point_b.energy_trend == 'stable':
            return 0.6
        else:
            return 0.4
    
    def _score_structural_match(self, point_a: TransitionPoint, point_b: TransitionPoint) -> float:
        """Score structural compatibility."""
        # Ideal pairs
        ideal_pairs = [
            ('outro', 'intro'), ('outro', 'build'),
            ('breakdown', 'intro'), ('breakdown', 'build'),
            ('verse', 'verse'), ('verse', 'intro')
        ]
        
        pair = (point_a.structural_label, point_b.structural_label)
        if pair in ideal_pairs:
            return 1.0
        elif point_a.structural_label == 'outro' and point_b.structural_label in ['intro', 'verse']:
            return 0.8
        elif point_a.structural_label in ['breakdown', 'verse'] and point_b.structural_label == 'intro':
            return 0.8
        else:
            return 0.6
    
    def _score_tempo_phase_match(self, 
                                 point_a: TransitionPoint,
                                 point_b: TransitionPoint,
                                 tempo_a: float,
                                 tempo_b: float) -> float:
        """Predict tempo/phase alignment quality."""
        tempo_diff = abs(tempo_a - tempo_b)
        
        if tempo_diff < 1:
            return 1.0
        elif tempo_diff < 2:
            return 0.9
        elif tempo_diff < 5:
            return 0.8
        elif tempo_diff < 10:
            return 0.6
        else:
            return 0.3
    
    def _predict_spectral_clash(self,
                               point_a: TransitionPoint,
                               point_b: TransitionPoint,
                               y_a: np.ndarray,
                               y_b: np.ndarray) -> float:
        """Predict spectral clash risk between two transition points."""
        # Extract short segments around transition points
        window_sec = 2.0  # 2 seconds before/after
        sr = self.sr
        
        # Extract segment A (around point_a)
        idx_a_start = max(0, int((point_a.time_sec - window_sec) * sr))
        idx_a_end = min(len(y_a), int((point_a.time_sec + window_sec) * sr))
        seg_a = y_a[idx_a_start:idx_a_end]
        
        # Extract segment B (around point_b)
        idx_b_start = max(0, int((point_b.time_sec - window_sec) * sr))
        idx_b_end = min(len(y_b), int((point_b.time_sec + window_sec) * sr))
        seg_b = y_b[idx_b_start:idx_b_end]
        
        if len(seg_a) == 0 or len(seg_b) == 0:
            return 0.5  # Unknown
        
        # Convert to mono if needed
        if seg_a.ndim > 1:
            seg_a = np.mean(seg_a, axis=1)
        if seg_b.ndim > 1:
            seg_b = np.mean(seg_b, axis=1)
        
        # Compute spectral centroid (brightness)
        try:
            centroid_a = librosa.feature.spectral_centroid(y=seg_a, sr=sr)[0]
            centroid_b = librosa.feature.spectral_centroid(y=seg_b, sr=sr)[0]
            
            centroid_a_mean = np.mean(centroid_a)
            centroid_b_mean = np.mean(centroid_b)
            
            # Compute spectral bandwidth (spread)
            bandwidth_a = librosa.feature.spectral_bandwidth(y=seg_a, sr=sr)[0]
            bandwidth_b = librosa.feature.spectral_bandwidth(y=seg_b, sr=sr)[0]
            
            bandwidth_a_mean = np.mean(bandwidth_a)
            bandwidth_b_mean = np.mean(bandwidth_b)
            
            # High clash risk if:
            # - Similar spectral centroid (similar frequency content)
            # - Both have high bandwidth (both occupy full spectrum)
            centroid_diff = abs(centroid_a_mean - centroid_b_mean) / (centroid_a_mean + centroid_b_mean + 1e-10)
            bandwidth_overlap = min(bandwidth_a_mean, bandwidth_b_mean) / (max(bandwidth_a_mean, bandwidth_b_mean) + 1e-10)
            
            # Clash risk: low difference + high overlap = high clash
            clash_risk = (1 - centroid_diff * 0.5) * bandwidth_overlap
            
            return float(np.clip(clash_risk, 0, 1))
        except:
            return 0.5  # Fallback
    
    def _predict_vocal_overlap(self,
                              point_a: TransitionPoint,
                              point_b: TransitionPoint,
                              y_a: np.ndarray,
                              y_b: np.ndarray) -> float:
        """Predict vocal overlap risk."""
        # Extract segments
        window_sec = 3.0
        sr = self.sr
        
        idx_a_start = max(0, int((point_a.time_sec - window_sec) * sr))
        idx_a_end = min(len(y_a), int((point_a.time_sec + window_sec) * sr))
        seg_a = y_a[idx_a_start:idx_a_end]
        
        idx_b_start = max(0, int((point_b.time_sec - window_sec) * sr))
        idx_b_end = min(len(y_b), int((point_b.time_sec + window_sec) * sr))
        seg_b = y_b[idx_b_start:idx_b_end]
        
        if len(seg_a) == 0 or len(seg_b) == 0:
            return 0.5
        
        # Convert to mono
        if seg_a.ndim > 1:
            seg_a = np.mean(seg_a, axis=1)
        if seg_b.ndim > 1:
            seg_b = np.mean(seg_b, axis=1)
        
        try:
            # Estimate vocal presence using spectral rolloff and zero crossing rate
            # Vocals typically have:
            # - Moderate spectral rolloff (not too high, not too low)
            # - Moderate zero crossing rate
            
            rolloff_a = librosa.feature.spectral_rolloff(y=seg_a, sr=sr)[0]
            rolloff_b = librosa.feature.spectral_rolloff(y=seg_b, sr=sr)[0]
            
            zcr_a = librosa.feature.zero_crossing_rate(seg_a)[0]
            zcr_b = librosa.feature.zero_crossing_rate(seg_b)[0]
            
            # Normalize to 0-1 range
            rolloff_a_norm = np.clip((np.mean(rolloff_a) - 1000) / 5000, 0, 1)
            rolloff_b_norm = np.clip((np.mean(rolloff_b) - 1000) / 5000, 0, 1)
            zcr_a_norm = np.clip(np.mean(zcr_a) / 0.1, 0, 1)
            zcr_b_norm = np.clip(np.mean(zcr_b) / 0.1, 0, 1)
            
            # Vocal presence score (both high = both have vocals)
            vocal_a = (rolloff_a_norm + zcr_a_norm) / 2
            vocal_b = (rolloff_b_norm + zcr_b_norm) / 2
            
            # Overlap risk: both have vocals
            overlap_risk = vocal_a * vocal_b
            
            return float(np.clip(overlap_risk, 0, 1))
        except:
            return 0.5  # Fallback

