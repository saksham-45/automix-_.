"""
Harmonic Analysis Module

Advanced harmonic theory for DJ mixing:
- Camelot Wheel integration
- Chord progression analysis
- Voice leading analysis
- Key compatibility scoring
"""
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional


class HarmonicAnalyzer:
    """
    Analyzes harmonic content using music theory and Camelot Wheel system.
    """
    
    # Camelot Wheel mapping: Key -> (Number, Letter)
    # Letter: A = Minor, B = Major
    CAMELOT_WHEEL = {
        'C': (8, 'B'), 'Am': (8, 'A'),
        'C#': (3, 'B'), 'A#m': (3, 'A'),
        'D': (10, 'B'), 'Bm': (10, 'A'),
        'D#': (5, 'B'), 'Cm': (5, 'A'),
        'E': (12, 'B'), 'C#m': (12, 'A'),
        'F': (7, 'B'), 'Dm': (7, 'A'),
        'F#': (2, 'B'), 'D#m': (2, 'A'),
        'G': (9, 'B'), 'Em': (9, 'A'),
        'G#': (4, 'B'), 'Fm': (4, 'A'),
        'A': (11, 'B'), 'F#m': (11, 'A'),
        'A#': (6, 'B'), 'Gm': (6, 'A'),
        'B': (1, 'B'), 'G#m': (1, 'A'),
    }
    
    # Reverse mapping: (Number, Letter) -> Key
    CAMELOT_TO_KEY = {v: k for k, v in CAMELOT_WHEEL.items()}
    
    # Key compatibility rules for Camelot Wheel
    # Perfect: Same key
    # +1/-1: Harmonic matches (perfect 5th/4th)
    # +7/-7: Energy matches (relative major/minor)
    # +2/-2: Energy build/drop
    COMPATIBLE_DELTAS = [0, 1, -1, 7, -7, 2, -2]
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
    
    def detect_key_camelot(self, y: np.ndarray) -> Dict:
        """
        Detect musical key and convert to Camelot notation.
        
        Returns:
            Dict with key, Camelot notation, and confidence
        """
        # Use chroma CQT for key detection
        chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Key names
        keys_major = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        keys_minor = ['Am', 'A#m', 'Bm', 'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m']
        
        # Template matching for major and minor
        # Major: Ionian mode template
        major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        # Minor: Aeolian mode template
        minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        
        # Normalize chroma
        chroma_norm = chroma_mean / (np.sum(chroma_mean) + 1e-10)
        
        # Match against all keys
        best_score_major = -1
        best_key_major = 'C'
        best_score_minor = -1
        best_key_minor = 'Am'
        
        for i, key in enumerate(keys_major):
            # Rotate template
            template = np.roll(major_template, i)
            score = np.dot(chroma_norm, template)
            if score > best_score_major:
                best_score_major = score
                best_key_major = key
        
        for i, key in enumerate(keys_minor):
            template = np.roll(minor_template, i)
            score = np.dot(chroma_norm, template)
            if score > best_score_minor:
                best_score_minor = score
                best_key_minor = key
        
        # Choose major or minor based on score
        if best_score_major > best_score_minor:
            detected_key = best_key_major
            mode = 'major'
            confidence = float(best_score_major)
        else:
            detected_key = best_key_minor
            mode = 'minor'
            confidence = float(best_score_minor)
        
        # Convert to Camelot
        camelot_num, camelot_letter = self.CAMELOT_WHEEL.get(detected_key, (8, 'B'))
        camelot_code = f"{camelot_num}{camelot_letter}"
        
        return {
            'key': detected_key,
            'mode': mode,
            'camelot': camelot_code,
            'camelot_number': camelot_num,
            'camelot_letter': camelot_letter,
            'confidence': confidence,
            'chroma_distribution': chroma_mean.tolist()
        }
    
    def are_keys_compatible(self, key_a: str, key_b: str, strict: bool = False) -> Dict:
        """
        Check if two keys are compatible using Camelot Wheel.
        
        Args:
            key_a: Key of first song (standard notation or Camelot)
            key_b: Key of second song
            strict: If True, only allow perfect matches and +1/-1
        
        Returns:
            Dict with compatibility score and details
        """
        # Convert to Camelot if needed
        camelot_a = self._key_to_camelot(key_a)
        camelot_b = self._key_to_camelot(key_b)
        
        if camelot_a is None or camelot_b is None:
            return {
                'compatible': False,
                'score': 0.0,
                'reason': 'Invalid key format'
            }
        
        num_a, letter_a = camelot_a
        num_b, letter_b = camelot_b
        
        # Calculate delta (wrap around at 12)
        delta = (num_b - num_a) % 12
        if delta > 6:
            delta = delta - 12
        
        # Check compatibility
        if delta == 0:
            # Same key - perfect match
            score = 1.0
            compatible = True
            reason = 'Perfect match - same key'
        elif abs(delta) == 1:
            # Harmonic match (perfect 4th/5th)
            score = 0.9
            compatible = True
            reason = 'Harmonic match - perfect 4th/5th'
        elif abs(delta) == 7 or (abs(delta) == 5 and letter_a != letter_b):
            # Relative major/minor or parallel
            score = 0.85
            compatible = True
            reason = 'Relative/parallel key relationship'
        elif abs(delta) == 2:
            # Energy build/drop
            score = 0.75
            compatible = not strict
            reason = 'Energy shift - good for transitions' if not strict else 'Too far apart'
        elif abs(delta) in [3, 4]:
            # Somewhat compatible
            score = 0.5
            compatible = not strict
            reason = 'Moderate compatibility' if not strict else 'Weak match'
        else:
            # Clashing keys
            score = 0.2
            compatible = False
            reason = 'Keys clash - avoid transition'
        
        return {
            'compatible': compatible,
            'score': score,
            'delta': delta,
            'camelot_a': f"{num_a}{letter_a}",
            'camelot_b': f"{num_b}{letter_b}",
            'reason': reason
        }
    
    def suggest_modulation_semitones(self,
                                     key_a: str,
                                     key_b: str,
                                     strategy: str = 'match_b',
                                     max_semitones: int = 2) -> Dict:
        """
        Suggest semitone shifts so A and B end up in the same or compatible key.
        Camelot wheel: one step = circle of 5ths = 7 semitones.
        
        Args:
            key_a: Key of outgoing song (A)
            key_b: Key of incoming song (B)
            strategy: 'match_b' (shift A toward B), 'match_a' (shift B toward A), 'midpoint' (both toward middle)
            max_semitones: Cap absolute shift per segment (e.g. 2 to avoid obvious timbre change)
        
        Returns:
            {'shift_a_semitones': int, 'shift_b_semitones': int, 'reason': str}
        """
        camelot_a = self._key_to_camelot(key_a)
        camelot_b = self._key_to_camelot(key_b)
        if camelot_a is None or camelot_b is None:
            return {'shift_a_semitones': 0, 'shift_b_semitones': 0, 'reason': 'invalid_key'}
        num_a, letter_a = camelot_a
        num_b, letter_b = camelot_b
        delta = (num_b - num_a) % 12
        if delta > 6:
            delta -= 12
        # One Camelot step = 7 semitones (circle of 5ths)
        semitones_a_to_b = (delta * 7) % 12
        if semitones_a_to_b > 6:
            semitones_a_to_b -= 12
        
        shift_a = 0
        shift_b = 0
        reason = 'same_or_compatible'
        if abs(semitones_a_to_b) <= 0:
            return {'shift_a_semitones': 0, 'shift_b_semitones': 0, 'reason': reason}
        
        if strategy == 'match_b':
            shift_a = int(np.clip(semitones_a_to_b, -max_semitones, max_semitones))
            shift_b = 0
            reason = 'shift_a_toward_b'
        elif strategy == 'match_a':
            shift_a = 0
            shift_b = int(np.clip(-semitones_a_to_b, -max_semitones, max_semitones))
            reason = 'shift_b_toward_a'
        else:
            half = semitones_a_to_b / 2
            shift_a = int(np.clip(round(half), -max_semitones, max_semitones))
            shift_b = int(np.clip(round(-half), -max_semitones, max_semitones))
            reason = 'midpoint'
        
        return {
            'shift_a_semitones': shift_a,
            'shift_b_semitones': shift_b,
            'reason': reason,
            'semitones_a_to_b': semitones_a_to_b
        }
    
    def _key_to_camelot(self, key: str) -> Optional[Tuple[int, str]]:
        """Convert key notation to Camelot (number, letter)."""
        # If already Camelot format (e.g., "8B", "3A")
        if len(key) >= 2 and key[-1] in ['A', 'B']:
            try:
                num = int(key[:-1])
                letter = key[-1]
                if 1 <= num <= 12:
                    return (num, letter)
            except:
                pass
        
        # Standard key notation
        key_normalized = key.strip().replace('maj', '').replace('min', 'm').replace('minor', 'm').replace('major', '')
        return self.CAMELOT_WHEEL.get(key_normalized)
    
    def analyze_chord_progression(self, y: np.ndarray, window_sec: float = 0.5) -> Dict:
        """
        Analyze chord progression and harmonic rhythm.
        
        Args:
            y: Audio signal
            window_sec: Analysis window size
        
        Returns:
            Dict with chord analysis
        """
        # Chroma for chord detection
        chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr)
        hop_length = self.sr // 4  # 0.25s hops
        
        # Time axis
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=self.sr, hop_length=hop_length)
        
        # Detect chord changes (simplified)
        # Use chroma difference between frames
        chroma_diff = np.diff(chroma, axis=1)
        change_energy = np.sum(np.abs(chroma_diff), axis=0)
        
        # Threshold for chord change
        threshold = np.percentile(change_energy, 75)
        change_frames = np.where(change_energy > threshold)[0]
        
        # Convert to times
        change_times = times[change_frames].tolist() if len(change_frames) > 0 else [0]
        
        # Estimate harmonic rhythm (time between chord changes)
        if len(change_times) > 1:
            intervals = np.diff(change_times)
            harmonic_rhythm = float(np.median(intervals))
        else:
            harmonic_rhythm = 4.0  # Default: 4 seconds
        
        # Detect cadences (phrase endings)
        # Cadences typically have longer harmonic rhythm or specific chord progressions
        cadence_times = []
        if len(change_times) > 3:
            # Look for longer intervals (potential phrase endings)
            long_intervals = np.where(intervals > harmonic_rhythm * 1.5)[0]
            cadence_times = [change_times[i+1] for i in long_intervals]
        
        return {
            'chord_change_times': change_times,
            'harmonic_rhythm_sec': harmonic_rhythm,
            'cadence_times': cadence_times,
            'num_chord_changes': len(change_times)
        }
    
    def analyze_voice_leading(self, key_a: str, key_b: str) -> Dict:
        """
        Analyze voice leading smoothness between two keys.
        
        Good voice leading minimizes large pitch leaps.
        """
        # Get notes in each key
        notes_a = self._get_key_notes(key_a)
        notes_b = self._get_key_notes(key_b)
        
        # Find common tones (shared notes)
        common_tones = set(notes_a) & set(notes_b)
        num_common = len(common_tones)
        common_ratio = num_common / max(len(notes_a), len(notes_b))
        
        # Calculate average voice leading distance
        # Map each note in key A to nearest note in key B
        total_distance = 0
        for note_a in notes_a:
            distances = [self._note_distance(note_a, note_b) for note_b in notes_b]
            total_distance += min(distances)
        
        avg_distance = total_distance / len(notes_a) if len(notes_a) > 0 else 0
        
        # Smooth voice leading: low average distance and many common tones
        smoothness_score = (1.0 - (avg_distance / 6)) * 0.5 + common_ratio * 0.5
        
        return {
            'common_tones': list(common_tones),
            'num_common_tones': num_common,
            'common_ratio': float(common_ratio),
            'avg_voice_leading_distance': float(avg_distance),
            'smoothness_score': float(smoothness_score)  # 0-1, higher = smoother
        }
    
    def _get_key_notes(self, key: str) -> List[int]:
        """Get pitch classes (0-11) in a given key."""
        key_to_notes = {
            'C': [0, 2, 4, 5, 7, 9, 11], 'Am': [9, 11, 0, 2, 4, 5, 7],
            'C#': [1, 3, 5, 6, 8, 10, 0], 'A#m': [10, 0, 1, 3, 5, 6, 8],
            'D': [2, 4, 6, 7, 9, 11, 1], 'Bm': [11, 1, 2, 4, 6, 7, 9],
            'D#': [3, 5, 7, 8, 10, 0, 2], 'Cm': [0, 2, 3, 5, 7, 8, 10],
            'E': [4, 6, 8, 9, 11, 1, 3], 'C#m': [1, 3, 4, 6, 8, 9, 11],
            'F': [5, 7, 9, 10, 0, 2, 4], 'Dm': [2, 4, 5, 7, 9, 10, 0],
            'F#': [6, 8, 10, 11, 1, 3, 5], 'D#m': [3, 5, 6, 8, 10, 11, 1],
            'G': [7, 9, 11, 0, 2, 4, 6], 'Em': [4, 6, 7, 9, 11, 0, 2],
            'G#': [8, 10, 0, 1, 3, 5, 7], 'Fm': [5, 7, 8, 10, 0, 1, 3],
            'A': [9, 11, 1, 2, 4, 6, 8], 'F#m': [6, 8, 9, 11, 1, 2, 4],
            'A#': [10, 0, 2, 3, 5, 7, 9], 'Gm': [7, 9, 10, 0, 2, 3, 5],
            'B': [11, 1, 3, 4, 6, 8, 10], 'G#m': [8, 10, 11, 1, 3, 4, 6],
        }
        
        key_normalized = key.strip().replace('maj', '').replace('min', 'm')
        return key_to_notes.get(key_normalized, [0, 2, 4, 5, 7, 9, 11])  # Default: C major
    
    def _note_distance(self, note_a: int, note_b: int) -> int:
        """Calculate semitone distance between two notes (0-11 pitch classes)."""
        dist = abs(note_b - note_a)
        return min(dist, 12 - dist)  # Wrap around
    
    def score_transition_harmonics(self,
                                   key_a: str,
                                   key_b: str,
                                   tempo_a: float,
                                   tempo_b: float) -> Dict:
        """
        Comprehensive harmonic scoring for transition compatibility.
        
        Combines key compatibility, voice leading, and tempo considerations.
        """
        # Key compatibility
        key_compat = self.are_keys_compatible(key_a, key_b)
        
        # Voice leading
        voice_leading = self.analyze_voice_leading(key_a, key_b)
        
        # Tempo compatibility (affects harmonic perception)
        tempo_diff = abs(tempo_a - tempo_b)
        tempo_score = 1.0 if tempo_diff < 2 else max(0, 1.0 - (tempo_diff / 10))
        
        # Combined score
        harmonic_score = (
            key_compat['score'] * 0.5 +
            voice_leading['smoothness_score'] * 0.3 +
            tempo_score * 0.2
        )
        
        return {
            'overall_score': float(harmonic_score),
            'key_compatibility': key_compat,
            'voice_leading': voice_leading,
            'tempo_compatibility': float(tempo_score),
            'recommendation': 'excellent' if harmonic_score > 0.8 else
                            'good' if harmonic_score > 0.6 else
                            'moderate' if harmonic_score > 0.4 else
                            'poor'
        }

