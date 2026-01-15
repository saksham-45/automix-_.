"""
Musical Structure Analysis Module

Analyzes song structure for optimal transition points:
- Phrase-level segmentation (8-bar, 16-bar)
- Section identification (intro, verse, chorus, etc.)
- Energy contour mapping
- Groove and rhythm analysis
"""
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MusicalPhrase:
    """Represents a musical phrase."""
    start_sec: float
    end_sec: float
    duration_bars: float
    phrase_number: int
    energy: float
    energy_trend: str  # 'rising', 'falling', 'stable', 'peak', 'valley'


@dataclass
class SongSection:
    """Represents a structural section."""
    section_type: str  # 'intro', 'verse', 'pre-chorus', 'chorus', 'bridge', 'outro'
    start_sec: float
    end_sec: float
    start_phrase: int
    end_phrase: int
    energy: float
    energy_level: str  # 'low', 'medium', 'high'


class StructureAnalyzer:
    """
    Analyzes musical structure - phrases, sections, and energy flow.
    """
    
    def __init__(self, sr: int = 44100, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length
    
    def analyze_structure(self, y: np.ndarray) -> Dict:
        """
        Comprehensive structure analysis.
        
        Returns:
            Dict with phrases, sections, and energy analysis
        """
        import time, json
        log_path = '/Users/saksham/untitled folder 7/.cursor/debug.log'
        
        #region agent log
        struct_start = time.time()
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"structure_analyzer.py:47","message":"Starting structure analysis","data":{"audio_len":len(y),"duration_sec":len(y)/self.sr},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # CRITICAL: Limit to 60 seconds max to prevent hanging on long files
        max_duration_sec = 60
        max_samples = max_duration_sec * self.sr
        if len(y) > max_samples:
            # Use first portion for structure analysis
            y = y[:max_samples]
            #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"structure_analyzer.py:62","message":"Truncated audio for speed","data":{"original_len":len(y)+max_samples,"truncated_len":len(y)},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
        
        # Detect tempo and beats first
        beat_start = time.time()
        tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr, hop_length=self.hop_length)
        
        #region agent log
        beat_time = time.time() - beat_start
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"structure_analyzer.py:54","message":"Beat tracking complete","data":{"time_sec":beat_time,"num_beats":len(beats)},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        beat_times = librosa.frames_to_time(beats, sr=self.sr, hop_length=self.hop_length)
        
        # Detect downbeats (bar boundaries)
        downbeat_start = time.time()
        downbeats = self._detect_downbeats(y, beat_times, tempo)
        
        #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"structure_analyzer.py:62","message":"Downbeat detection complete","data":{"time_sec":time.time()-downbeat_start},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Detect phrases (typically 8 or 16 bars)
        phrase_start = time.time()
        phrases = self._detect_phrases(y, downbeats, tempo)
        
        #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"structure_analyzer.py:68","message":"Phrase detection complete","data":{"time_sec":time.time()-phrase_start,"num_phrases":len(phrases)},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Detect sections (intro, verse, chorus, etc.)
        section_start = time.time()
        sections = self._detect_sections(y, phrases, downbeats)
        
        #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"structure_analyzer.py:75","message":"Section detection complete","data":{"time_sec":time.time()-section_start,"num_sections":len(sections)},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Energy contour
        energy_start = time.time()
        energy_contour = self._analyze_energy_contour(y, downbeats)
        
        #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"structure_analyzer.py:82","message":"Energy contour complete","data":{"time_sec":time.time()-energy_start},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Groove analysis
        groove_start = time.time()
        groove = self._analyze_groove(y, beat_times, tempo)
        
        #region agent log
        struct_total = time.time() - struct_start
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"structure_analyzer.py:90","message":"Structure analysis complete","data":{"total_time_sec":struct_total,"groove_time_sec":time.time()-groove_start},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        return {
            'tempo': float(tempo),
            'downbeats': downbeats,
            'phrases': [
                {
                    'start_sec': p.start_sec,
                    'end_sec': p.end_sec,
                    'duration_bars': p.duration_bars,
                    'phrase_number': p.phrase_number,
                    'energy': p.energy,
                    'energy_trend': p.energy_trend
                } for p in phrases
            ],
            'sections': [
                {
                    'type': s.section_type,
                    'start_sec': s.start_sec,
                    'end_sec': s.end_sec,
                    'start_phrase': s.start_phrase,
                    'end_phrase': s.end_phrase,
                    'energy': s.energy,
                    'energy_level': s.energy_level
                } for s in sections
            ],
            'energy_contour': energy_contour,
            'groove': groove,
            'best_mix_in_points': self._find_best_mix_in_points(sections, phrases),
            'best_mix_out_points': self._find_best_mix_out_points(sections, phrases)
        }
    
    def _detect_downbeats(self, y: np.ndarray, beat_times: np.ndarray, tempo: float) -> List[float]:
        """Detect downbeats (bar boundaries) - every 4th beat."""
        # Use onset detection to find strong beats (downbeats)
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length)
        onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=self.sr, hop_length=self.hop_length)
        
        # Find peaks in onset strength that align with beats
        from scipy.signal import find_peaks
        
        # Average onset strength around each beat
        beat_onsets = []
        for beat_time in beat_times:
            # Find nearest onset frame
            idx = np.argmin(np.abs(onset_times - beat_time))
            if idx < len(onset_env):
                beat_onsets.append(onset_env[idx])
            else:
                beat_onsets.append(0)
        
        # Downbeats typically have stronger onsets every 4 beats
        downbeats = []
        for i in range(0, len(beat_times), 4):
            if i < len(beat_times):
                downbeats.append(float(beat_times[i]))
        
        return downbeats
    
    def _detect_phrases(self, y: np.ndarray, downbeats: List[float], tempo: float) -> List[MusicalPhrase]:
        """Detect musical phrases (8-bar or 16-bar segments)."""
        if len(downbeats) < 8:
            # Fallback: use regular intervals
            duration = len(y) / self.sr
            bar_duration = 4 * (60 / tempo)  # 4 beats per bar
            n_bars = int(duration / bar_duration)
            phrases = []
            for i in range(0, n_bars, 8):  # 8-bar phrases
                start = i * bar_duration
                end = min((i + 8) * bar_duration, duration)
                if end > start:
                    seg = y[int(start * self.sr):int(end * self.sr)]
                    energy = np.mean(np.abs(seg) ** 2)
                    phrases.append(MusicalPhrase(
                        start_sec=start,
                        end_sec=end,
                        duration_bars=8.0,
                        phrase_number=len(phrases),
                        energy=float(energy),
                        energy_trend='stable'
                    ))
            return phrases
        
        # Group bars into phrases (8 or 16 bars)
        phrases = []
        phrase_length_bars = 8  # Default to 8 bars
        
        for i in range(0, len(downbeats) - phrase_length_bars, phrase_length_bars):
            start_beat = i
            end_beat = min(i + phrase_length_bars, len(downbeats) - 1)
            
            start_sec = downbeats[start_beat]
            end_sec = downbeats[end_beat] if end_beat < len(downbeats) else len(y) / self.sr
            
            # Extract phrase segment
            start_sample = int(start_sec * self.sr)
            end_sample = int(end_sec * self.sr)
            if end_sample > start_sample:
                phrase_seg = y[start_sample:end_sample]
                energy = np.mean(np.abs(phrase_seg) ** 2)
                
                # Determine energy trend
                # Split phrase in half
                mid = len(phrase_seg) // 2
                energy_first = np.mean(np.abs(phrase_seg[:mid]) ** 2)
                energy_second = np.mean(np.abs(phrase_seg[mid:]) ** 2)
                
                if energy_second > energy_first * 1.2:
                    trend = 'rising'
                elif energy_second < energy_first * 0.8:
                    trend = 'falling'
                else:
                    trend = 'stable'
                
                phrases.append(MusicalPhrase(
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    duration_bars=float(end_beat - start_beat),
                    phrase_number=len(phrases),
                    energy=float(energy),
                    energy_trend=trend
                ))
        
        return phrases
    
    def _detect_sections(self, y: np.ndarray, phrases: List[MusicalPhrase], downbeats: List[float]) -> List[SongSection]:
        """Detect structural sections (intro, verse, chorus, etc.)."""
        if len(phrases) == 0:
            return []
        
        # Use energy and repetition analysis to identify sections
        duration = len(y) / self.sr
        
        # Analyze chroma similarity for repetition
        chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=self.hop_length)
        
        sections = []
        
        # Simple heuristic-based section detection
        # Intro: First 10-15% of song, lower energy
        # Outro: Last 10-15% of song, lower energy
        # Verse: Repeated sections with medium energy
        # Chorus: Highest energy, most repeated
        
        # Find phrase energies
        phrase_energies = [p.energy for p in phrases]
        if len(phrase_energies) == 0:
            return []
        
        avg_energy = np.mean(phrase_energies)
        max_energy = np.max(phrase_energies)
        min_energy = np.min(phrase_energies)
        
        # Intro
        intro_end = min(2, len(phrases))  # First 2 phrases
        if intro_end > 0:
            intro_energy = np.mean([p.energy for p in phrases[:intro_end]])
            sections.append(SongSection(
                section_type='intro',
                start_sec=phrases[0].start_sec,
                end_sec=phrases[intro_end-1].end_sec,
                start_phrase=0,
                end_phrase=intro_end-1,
                energy=float(intro_energy),
                energy_level='low' if intro_energy < avg_energy else 'medium'
            ))
        
        # Middle sections (verse/chorus)
        remaining_phrases = phrases[intro_end:]
        if len(remaining_phrases) > 4:
            # Find highest energy phrases (likely chorus)
            remaining_energies = [p.energy for p in remaining_phrases]
            chorus_threshold = np.percentile(remaining_energies, 75)
            
            # Label sections based on energy
            current_type = 'verse'
            section_start = intro_end
            
            for i, phrase in enumerate(remaining_phrases):
                if phrase.energy > chorus_threshold and current_type == 'verse':
                    # Start of chorus
                    if i > 0:
                        sections.append(SongSection(
                            section_type='verse',
                            start_sec=remaining_phrases[section_start].start_sec,
                            end_sec=remaining_phrases[i-1].end_sec,
                            start_phrase=section_start + intro_end,
                            end_phrase=i-1 + intro_end,
                            energy=float(np.mean([p.energy for p in remaining_phrases[section_start:i]])),
                            energy_level='medium'
                        ))
                    section_start = i
                    current_type = 'chorus'
                elif phrase.energy <= chorus_threshold and current_type == 'chorus':
                    # End of chorus
                    sections.append(SongSection(
                        section_type='chorus',
                        start_sec=remaining_phrases[section_start].start_sec,
                        end_sec=remaining_phrases[i-1].end_sec,
                        start_phrase=section_start + intro_end,
                        end_phrase=i-1 + intro_end,
                        energy=float(np.mean([p.energy for p in remaining_phrases[section_start:i]])),
                        energy_level='high'
                    ))
                    section_start = i
                    current_type = 'verse'
            
            # Final section
            if section_start < len(remaining_phrases):
                final_type = 'chorus' if current_type == 'chorus' else 'verse'
                sections.append(SongSection(
                    section_type=final_type,
                    start_sec=remaining_phrases[section_start].start_sec,
                    end_sec=remaining_phrases[-1].end_sec,
                    start_phrase=section_start + intro_end,
                    end_phrase=len(phrases) - 1,
                    energy=float(np.mean([p.energy for p in remaining_phrases[section_start:]])),
                    energy_level='high' if final_type == 'chorus' else 'medium'
                ))
        
        # Outro (last 1-2 phrases)
        outro_start = max(0, len(phrases) - 2)
        if outro_start < len(phrases):
            outro_energy = np.mean([p.energy for p in phrases[outro_start:]])
            sections.append(SongSection(
                section_type='outro',
                start_sec=phrases[outro_start].start_sec,
                end_sec=phrases[-1].end_sec,
                start_phrase=outro_start,
                end_phrase=len(phrases) - 1,
                energy=float(outro_energy),
                energy_level='low'
            ))
        
        return sections
    
    def _analyze_energy_contour(self, y: np.ndarray, downbeats: List[float]) -> Dict:
        """Analyze energy contour over time."""
        # Compute RMS energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=self.sr, hop_length=self.hop_length)
        
        # Find peaks and valleys
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(rms, distance=int(self.sr / self.hop_length))
        valleys, _ = find_peaks(-rms, distance=int(self.sr / self.hop_length))
        
        # Build-up detection: consecutive rising energy
        # Drop detection: sudden energy fall
        
        return {
            'energy_curve': {
                'times_sec': times.tolist(),
                'rms_values': rms.tolist()
            },
            'peaks': times[peaks].tolist() if len(peaks) > 0 else [],
            'valleys': times[valleys].tolist() if len(valleys) > 0 else []
        }
    
    def _analyze_groove(self, y: np.ndarray, beat_times: np.ndarray, tempo: float) -> Dict:
        """Analyze groove and microtiming."""
        if len(beat_times) < 2:
            return {'swing': 0.0, 'groove_type': 'straight'}
        
        # Calculate beat intervals
        intervals = np.diff(beat_times)
        expected_interval = 60.0 / tempo
        
        # Microtiming deviations
        deviations = (intervals - expected_interval) * 1000  # ms
        
        # Swing detection: alternating long-short pattern
        # In swing, even beats are slightly delayed
        swing_scores = []
        for i in range(0, len(deviations) - 1, 2):
            if i + 1 < len(deviations):
                # Even beat (2nd, 4th) should be later
                swing_score = deviations[i+1] - deviations[i]
                swing_scores.append(swing_score)
        
        avg_swing = np.mean(swing_scores) if len(swing_scores) > 0 else 0.0
        
        # Determine groove type
        if abs(avg_swing) > 20:  # >20ms swing
            groove_type = 'swing'
        elif abs(avg_swing) > 5:
            groove_type = 'shuffle'
        else:
            groove_type = 'straight'
        
        return {
            'swing_ms': float(avg_swing),
            'groove_type': groove_type,
            'microtiming_std_ms': float(np.std(deviations)),
            'groove_locked': float(np.std(deviations)) < 10  # Tight timing
        }
    
    def _find_best_mix_in_points(self, sections: List[SongSection], phrases: List[MusicalPhrase]) -> List[float]:
        """Find best points to mix INTO this song."""
        points = []
        # Ensure we return list of floats, not numpy arrays
        
        # Prefer intro sections, phrase starts
        for section in sections:
            if section.section_type in ['intro', 'verse']:
                # Start of section or phrase boundaries
                if section.start_phrase < len(phrases):
                    points.append(phrases[section.start_phrase].start_sec)
        
        # Also consider phrase starts with rising energy
        for phrase in phrases[:min(8, len(phrases))]:  # First 8 phrases
            if phrase.energy_trend == 'rising':
                points.append(phrase.start_sec)
        
        # Convert to list of floats (ensure no numpy arrays)
        return sorted([float(p) for p in set(points)])
    
    def _find_best_mix_out_points(self, sections: List[SongSection], phrases: List[MusicalPhrase]) -> List[float]:
        """Find best points to mix OUT OF this song."""
        points = []
        
        # Prefer outro, phrase endings, energy valleys
        for section in sections:
            if section.section_type in ['outro', 'verse']:
                if section.end_phrase < len(phrases):
                    points.append(phrases[section.end_phrase].end_sec)
        
        # Phrase endings with falling energy
        for phrase in phrases[-min(8, len(phrases)):]:  # Last 8 phrases
            if phrase.energy_trend in ['falling', 'stable']:
                points.append(phrase.end_sec)
        
        # Convert to list of floats (ensure no numpy arrays)
        return sorted([float(p) for p in set(points)])

