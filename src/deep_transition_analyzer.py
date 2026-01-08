"""
Deep Transition Analyzer

Takes known transition timestamps and performs deep analysis to understand:
1. HOW the transition happens (techniques used)
2. WHY it's effective (what makes it smooth)
3. WHAT patterns indicate good transitions
4. Training data for AI to learn transition execution
"""

import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from scipy.stats import pearsonr
import json


@dataclass
class TransitionPoint:
    """A manually marked transition point."""
    time_sec: float
    from_track: str
    to_track: str
    transition_type: str = "blend"  # blend, cut, drop, etc.


@dataclass
class DeepTransitionAnalysis:
    """Comprehensive analysis of a single transition."""
    # Basic info
    transition_start_sec: float
    transition_end_sec: float
    from_track: str
    to_track: str
    duration_sec: float
    
    # Timing analysis
    beat_aligned: bool
    downbeat_aligned: bool
    bars_duration: int
    phase_offset_ms: float
    
    # Volume/Crossfade analysis
    crossfade_type: str  # linear, exponential, equal_power, s_curve
    crossfade_center_sec: float  # where 50/50 mix happens
    outgoing_fade_start_sec: float
    incoming_fade_start_sec: float
    overlap_duration_sec: float
    
    # EQ/Frequency analysis
    bass_swap_detected: bool
    bass_swap_time_sec: Optional[float]
    low_cut_on_incoming: bool
    high_cut_on_outgoing: bool
    eq_automation_detected: bool
    
    # Harmonic analysis
    key_outgoing: str
    key_incoming: str
    key_compatible: bool
    harmonic_tension: float  # 0-1, how dissonant the overlap is
    
    # Energy analysis
    energy_before: float
    energy_during: float
    energy_after: float
    energy_dip: bool  # did energy drop during transition?
    energy_build: bool  # did energy build up?
    
    # Spectral analysis
    spectral_smoothness: float  # how smooth the spectral transition is
    frequency_masking: float  # how much frequencies overlap/clash
    
    # Rhythmic analysis
    tempo_outgoing: float
    tempo_incoming: float
    tempo_match: bool
    beat_phase_alignment: float  # 0-1, how well beats align
    
    # Effectiveness metrics
    perceived_smoothness: float  # 0-1, overall smoothness estimate
    technique_complexity: float  # 0-1, how complex the transition technique is
    
    # Raw curves for training
    volume_curve_outgoing: List[Tuple[float, float]]  # (time_rel, db)
    volume_curve_incoming: List[Tuple[float, float]]
    bass_energy_curve: List[Tuple[float, float]]
    mid_energy_curve: List[Tuple[float, float]]
    high_energy_curve: List[Tuple[float, float]]
    
    # What makes this transition effective
    effectiveness_factors: List[str]
    techniques_used: List[str]


class DeepTransitionAnalyzer:
    """
    Performs deep analysis on known transition points.
    """
    
    def __init__(self, sr: int = 22050, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length
    
    def analyze_transitions(self, 
                           audio_path: str, 
                           transitions: List[TransitionPoint],
                           context_before_sec: float = 10.0,
                           context_after_sec: float = 10.0) -> List[DeepTransitionAnalysis]:
        """
        Analyze all marked transitions in an audio file.
        """
        print(f"Loading audio: {audio_path}")
        y, sr = librosa.load(audio_path, sr=self.sr)
        duration = len(y) / sr
        
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Analyzing {len(transitions)} transitions...")
        
        analyses = []
        
        for i, trans in enumerate(transitions):
            print(f"\n{'='*60}")
            print(f"Transition {i+1}: {trans.from_track} → {trans.to_track}")
            print(f"Time: {trans.time_sec/60:.2f} min ({trans.time_sec:.1f}s)")
            print('='*60)
            
            analysis = self._analyze_single_transition(
                y, sr, trans, 
                context_before_sec, 
                context_after_sec,
                duration
            )
            analyses.append(analysis)
            
            self._print_analysis_summary(analysis)
        
        return analyses
    
    def _analyze_single_transition(self,
                                   y: np.ndarray,
                                   sr: int,
                                   trans: TransitionPoint,
                                   context_before: float,
                                   context_after: float,
                                   total_duration: float) -> DeepTransitionAnalysis:
        """Deep analysis of a single transition."""
        
        # Define analysis region
        start_sec = max(0, trans.time_sec - context_before)
        end_sec = min(total_duration, trans.time_sec + context_after)
        
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        trans_sample = int(trans.time_sec * sr)
        
        # Extract segment
        segment = y[start_sample:end_sample]
        
        # Get pre/during/post segments
        pre_end = int((trans.time_sec - 2) * sr) - start_sample
        post_start = int((trans.time_sec + 2) * sr) - start_sample
        
        pre_segment = segment[:max(1, pre_end)]
        trans_segment = segment[max(0, pre_end):min(len(segment), post_start)]
        post_segment = segment[min(len(segment)-1, post_start):]
        
        # 1. BEAT AND TIMING ANALYSIS
        print("  Analyzing beat alignment...")
        tempo, beats = librosa.beat.beat_track(y=segment, sr=sr, hop_length=self.hop_length)
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
        
        # Find beats around transition point
        trans_local_time = trans.time_sec - start_sec
        beat_aligned, downbeat_aligned, phase_offset = self._analyze_beat_alignment(
            beat_times, trans_local_time, tempo
        )
        
        # Estimate transition duration in bars
        bar_duration = 4 * (60 / tempo)  # 4 beats per bar
        
        # 2. VOLUME/CROSSFADE ANALYSIS
        print("  Analyzing crossfade...")
        volume_analysis = self._analyze_volume_curves(segment, sr, trans_local_time)
        
        # 3. FREQUENCY/EQ ANALYSIS
        print("  Analyzing frequency bands...")
        eq_analysis = self._analyze_eq_automation(segment, sr, trans_local_time)
        
        # 4. HARMONIC ANALYSIS
        print("  Analyzing harmony...")
        harmonic_analysis = self._analyze_harmony(
            pre_segment, trans_segment, post_segment, sr
        )
        
        # 5. ENERGY ANALYSIS
        print("  Analyzing energy flow...")
        energy_analysis = self._analyze_energy_flow(
            pre_segment, trans_segment, post_segment, sr
        )
        
        # 6. SPECTRAL ANALYSIS
        print("  Analyzing spectral characteristics...")
        spectral_analysis = self._analyze_spectral_transition(
            pre_segment, trans_segment, post_segment, sr
        )
        
        # 7. DETERMINE TECHNIQUES USED
        techniques = self._identify_techniques(
            volume_analysis, eq_analysis, harmonic_analysis, 
            energy_analysis, spectral_analysis
        )
        
        # 8. ASSESS EFFECTIVENESS
        effectiveness = self._assess_effectiveness(
            volume_analysis, eq_analysis, harmonic_analysis,
            energy_analysis, spectral_analysis, beat_aligned
        )
        
        # Build analysis result
        return DeepTransitionAnalysis(
            transition_start_sec=trans.time_sec - 5,  # Estimated blend start
            transition_end_sec=trans.time_sec + 5,
            from_track=trans.from_track,
            to_track=trans.to_track,
            duration_sec=10.0,
            
            beat_aligned=beat_aligned,
            downbeat_aligned=downbeat_aligned,
            bars_duration=max(1, int(10.0 / bar_duration)),
            phase_offset_ms=phase_offset * 1000,
            
            crossfade_type=volume_analysis['crossfade_type'],
            crossfade_center_sec=trans.time_sec,
            outgoing_fade_start_sec=trans.time_sec - volume_analysis['fade_duration'] / 2,
            incoming_fade_start_sec=trans.time_sec - volume_analysis['fade_duration'] / 2,
            overlap_duration_sec=volume_analysis['overlap_duration'],
            
            bass_swap_detected=eq_analysis['bass_swap'],
            bass_swap_time_sec=eq_analysis.get('bass_swap_time'),
            low_cut_on_incoming=eq_analysis['low_cut_incoming'],
            high_cut_on_outgoing=eq_analysis['high_cut_outgoing'],
            eq_automation_detected=eq_analysis['automation_detected'],
            
            key_outgoing=harmonic_analysis['key_before'],
            key_incoming=harmonic_analysis['key_after'],
            key_compatible=harmonic_analysis['compatible'],
            harmonic_tension=harmonic_analysis['tension'],
            
            energy_before=energy_analysis['before'],
            energy_during=energy_analysis['during'],
            energy_after=energy_analysis['after'],
            energy_dip=energy_analysis['has_dip'],
            energy_build=energy_analysis['has_build'],
            
            spectral_smoothness=spectral_analysis['smoothness'],
            frequency_masking=spectral_analysis['masking'],
            
            tempo_outgoing=tempo,
            tempo_incoming=tempo,  # Assumed same in a good DJ mix
            tempo_match=True,
            beat_phase_alignment=1.0 - (phase_offset / (60/tempo/2)),
            
            perceived_smoothness=effectiveness['smoothness'],
            technique_complexity=effectiveness['complexity'],
            
            volume_curve_outgoing=volume_analysis['outgoing_curve'],
            volume_curve_incoming=volume_analysis['incoming_curve'],
            bass_energy_curve=eq_analysis['bass_curve'],
            mid_energy_curve=eq_analysis['mid_curve'],
            high_energy_curve=eq_analysis['high_curve'],
            
            effectiveness_factors=effectiveness['factors'],
            techniques_used=techniques
        )
    
    def _analyze_beat_alignment(self, 
                                beat_times: np.ndarray, 
                                trans_time: float,
                                tempo: float) -> Tuple[bool, bool, float]:
        """Analyze if transition happens on beat/downbeat."""
        if len(beat_times) == 0:
            return False, False, 0.0
        
        # Find nearest beat
        distances = np.abs(beat_times - trans_time)
        nearest_idx = np.argmin(distances)
        nearest_distance = distances[nearest_idx]
        
        beat_duration = 60 / tempo
        
        # Beat aligned if within 50ms of a beat
        beat_aligned = nearest_distance < 0.05
        
        # Downbeat aligned if on every 4th beat (roughly)
        downbeat_aligned = beat_aligned and (nearest_idx % 4 == 0)
        
        return beat_aligned, downbeat_aligned, nearest_distance
    
    def _analyze_volume_curves(self, 
                              segment: np.ndarray, 
                              sr: int,
                              trans_time: float) -> Dict:
        """Analyze volume/crossfade curves around transition."""
        hop = self.hop_length
        
        # Compute RMS in short frames
        frame_length = int(0.1 * sr)  # 100ms frames
        hop_length = int(0.05 * sr)   # 50ms hop
        
        rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms + 1e-10)
        
        # Smooth
        rms_db_smooth = gaussian_filter1d(rms_db, sigma=3)
        
        # Find the pattern - is it linear, exponential, s-curve?
        # Split into before/after transition
        trans_idx = np.argmin(np.abs(rms_times - trans_time))
        
        before = rms_db_smooth[:trans_idx]
        after = rms_db_smooth[trans_idx:]
        
        # Determine crossfade type based on curve shape
        crossfade_type = "linear"  # Default
        
        if len(before) > 10 and len(after) > 10:
            # Check if outgoing volume decreases
            before_trend = np.polyfit(np.arange(len(before)), before, 1)[0]
            after_trend = np.polyfit(np.arange(len(after)), after, 1)[0]
            
            # If both relatively flat, it might be a quick cut
            if abs(before_trend) < 0.1 and abs(after_trend) < 0.1:
                crossfade_type = "cut"
            # If smooth S-curve pattern
            elif before_trend < -0.2 and after_trend > 0.1:
                crossfade_type = "s_curve"
            elif before_trend < -0.5:
                crossfade_type = "exponential_out"
        
        # Estimate overlap duration
        # Find where outgoing starts to fade and incoming becomes audible
        threshold_db = -20
        above_threshold = rms_db_smooth > threshold_db
        
        fade_regions = np.diff(above_threshold.astype(int))
        overlap_duration = 5.0  # Default estimate
        
        # Build curves for training data
        outgoing_curve = [(float(t), float(db)) for t, db in 
                          zip(rms_times[:trans_idx][-20:], rms_db_smooth[:trans_idx][-20:])]
        incoming_curve = [(float(t), float(db)) for t, db in 
                          zip(rms_times[trans_idx:][:20], rms_db_smooth[trans_idx:][:20])]
        
        return {
            'crossfade_type': crossfade_type,
            'fade_duration': 5.0,
            'overlap_duration': overlap_duration,
            'outgoing_curve': outgoing_curve,
            'incoming_curve': incoming_curve,
            'max_db_during': float(np.max(rms_db_smooth)),
            'min_db_during': float(np.min(rms_db_smooth))
        }
    
    def _analyze_eq_automation(self, 
                              segment: np.ndarray, 
                              sr: int,
                              trans_time: float) -> Dict:
        """Analyze EQ/frequency band automation."""
        hop = self.hop_length
        
        # Compute spectrogram
        S = np.abs(librosa.stft(segment, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
        
        # Define bands
        bass_mask = freqs < 200
        mid_mask = (freqs >= 200) & (freqs < 2000)
        high_mask = freqs >= 2000
        
        # Energy in each band over time
        bass_energy = np.mean(S[bass_mask, :], axis=0) if np.any(bass_mask) else np.zeros(S.shape[1])
        mid_energy = np.mean(S[mid_mask, :], axis=0) if np.any(mid_mask) else np.zeros(S.shape[1])
        high_energy = np.mean(S[high_mask, :], axis=0) if np.any(high_mask) else np.zeros(S.shape[1])
        
        # Normalize
        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        bass_norm = gaussian_filter1d(norm(bass_energy), sigma=5)
        mid_norm = gaussian_filter1d(norm(mid_energy), sigma=5)
        high_norm = gaussian_filter1d(norm(high_energy), sigma=5)
        
        # Find transition point in frames
        trans_frame = np.argmin(np.abs(times - trans_time))
        
        # Detect bass swap: bass drops then comes back up around transition
        bass_before = np.mean(bass_norm[max(0, trans_frame-50):trans_frame])
        bass_at = bass_norm[trans_frame] if trans_frame < len(bass_norm) else bass_before
        bass_after = np.mean(bass_norm[trans_frame:min(len(bass_norm), trans_frame+50)])
        
        bass_swap = (bass_at < bass_before * 0.7) or (bass_at < bass_after * 0.7)
        
        # Find bass swap point (where bass is minimum)
        bass_swap_time = None
        if bass_swap:
            min_idx = np.argmin(bass_norm[max(0, trans_frame-50):trans_frame+50])
            bass_swap_time = times[max(0, trans_frame-50) + min_idx]
        
        # Detect high/low cuts
        high_before = np.mean(high_norm[max(0, trans_frame-50):trans_frame])
        high_after = np.mean(high_norm[trans_frame:min(len(high_norm), trans_frame+50)])
        
        low_cut_incoming = bass_before > bass_after * 1.3  # Bass lower after = low cut on incoming
        high_cut_outgoing = high_before > high_after * 1.3  # Highs lower after
        
        # Detect automation (significant changes in any band)
        bass_variance = np.var(bass_norm[max(0, trans_frame-50):trans_frame+50])
        automation_detected = bass_variance > 0.02 or bass_swap
        
        # Build curves for training
        curve_range = 30  # frames before/after
        start_idx = max(0, trans_frame - curve_range)
        end_idx = min(len(times), trans_frame + curve_range)
        
        bass_curve = [(float(times[i] - trans_time), float(bass_norm[i])) 
                      for i in range(start_idx, end_idx)]
        mid_curve = [(float(times[i] - trans_time), float(mid_norm[i])) 
                     for i in range(start_idx, end_idx)]
        high_curve = [(float(times[i] - trans_time), float(high_norm[i])) 
                      for i in range(start_idx, end_idx)]
        
        return {
            'bass_swap': bass_swap,
            'bass_swap_time': float(bass_swap_time) if bass_swap_time else None,
            'low_cut_incoming': low_cut_incoming,
            'high_cut_outgoing': high_cut_outgoing,
            'automation_detected': automation_detected,
            'bass_curve': bass_curve,
            'mid_curve': mid_curve,
            'high_curve': high_curve
        }
    
    def _analyze_harmony(self, 
                        pre: np.ndarray, 
                        trans: np.ndarray, 
                        post: np.ndarray,
                        sr: int) -> Dict:
        """Analyze harmonic compatibility."""
        # Get chroma for each section
        if len(pre) > 1024:
            chroma_pre = np.mean(librosa.feature.chroma_cqt(y=pre, sr=sr), axis=1)
        else:
            chroma_pre = np.zeros(12)
            
        if len(post) > 1024:
            chroma_post = np.mean(librosa.feature.chroma_cqt(y=post, sr=sr), axis=1)
        else:
            chroma_post = np.zeros(12)
            
        if len(trans) > 1024:
            chroma_trans = np.mean(librosa.feature.chroma_cqt(y=trans, sr=sr), axis=1)
        else:
            chroma_trans = np.zeros(12)
        
        # Estimate keys
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_before = key_names[np.argmax(chroma_pre)]
        key_after = key_names[np.argmax(chroma_post)]
        
        # Key compatibility (perfect 5th, relative major/minor, parallel)
        key_diff = (np.argmax(chroma_post) - np.argmax(chroma_pre)) % 12
        compatible_intervals = [0, 5, 7, 3, 4, 8, 9]  # Unison, 4th, 5th, minor 3rd, major 3rd, etc.
        compatible = key_diff in compatible_intervals
        
        # Harmonic tension during transition
        # Higher when chroma during transition is dissimilar to both before and after
        if np.sum(chroma_trans) > 0:
            sim_to_before = np.dot(chroma_trans, chroma_pre) / (np.linalg.norm(chroma_trans) * np.linalg.norm(chroma_pre) + 1e-8)
            sim_to_after = np.dot(chroma_trans, chroma_post) / (np.linalg.norm(chroma_trans) * np.linalg.norm(chroma_post) + 1e-8)
            tension = 1 - max(sim_to_before, sim_to_after)
        else:
            tension = 0.5
        
        return {
            'key_before': key_before,
            'key_after': key_after,
            'compatible': compatible,
            'tension': float(tension)
        }
    
    def _analyze_energy_flow(self,
                            pre: np.ndarray,
                            trans: np.ndarray,
                            post: np.ndarray,
                            sr: int) -> Dict:
        """Analyze energy dynamics through transition."""
        def get_energy(segment):
            if len(segment) < 512:
                return 0.0
            return float(np.mean(librosa.feature.rms(y=segment)))
        
        energy_before = get_energy(pre)
        energy_during = get_energy(trans)
        energy_after = get_energy(post)
        
        # Detect dip (energy drops during transition)
        has_dip = energy_during < min(energy_before, energy_after) * 0.8
        
        # Detect build (energy increases through transition)
        has_build = energy_after > energy_before * 1.2
        
        return {
            'before': energy_before,
            'during': energy_during,
            'after': energy_after,
            'has_dip': has_dip,
            'has_build': has_build
        }
    
    def _analyze_spectral_transition(self,
                                    pre: np.ndarray,
                                    trans: np.ndarray,
                                    post: np.ndarray,
                                    sr: int) -> Dict:
        """Analyze spectral characteristics of transition."""
        def get_spectral_features(segment):
            if len(segment) < 1024:
                return np.zeros(128)
            S = np.abs(librosa.stft(segment, n_fft=256))
            return np.mean(S, axis=1)
        
        spec_before = get_spectral_features(pre)
        spec_trans = get_spectral_features(trans)
        spec_after = get_spectral_features(post)
        
        # Smoothness: how gradually does spectrum change?
        # Compare transition spectrum to interpolation of before/after
        expected_trans = (spec_before + spec_after) / 2
        if np.linalg.norm(spec_trans) > 0 and np.linalg.norm(expected_trans) > 0:
            smoothness = np.dot(spec_trans, expected_trans) / (
                np.linalg.norm(spec_trans) * np.linalg.norm(expected_trans)
            )
        else:
            smoothness = 0.5
        
        # Frequency masking: how much do frequencies overlap during transition?
        # Higher values in transition = more overlap
        masking = 0.0
        if np.max(spec_before) > 0 and np.max(spec_after) > 0:
            overlap = np.minimum(spec_before / np.max(spec_before), 
                                spec_after / np.max(spec_after))
            masking = float(np.mean(overlap))
        
        return {
            'smoothness': float(smoothness),
            'masking': masking
        }
    
    def _identify_techniques(self, 
                            volume: Dict, 
                            eq: Dict, 
                            harmony: Dict,
                            energy: Dict,
                            spectral: Dict) -> List[str]:
        """Identify which DJ techniques were used."""
        techniques = []
        
        # Crossfade type
        if volume['crossfade_type'] == 'cut':
            techniques.append('quick_cut')
        elif volume['crossfade_type'] == 's_curve':
            techniques.append('s_curve_crossfade')
        elif volume['crossfade_type'] == 'exponential_out':
            techniques.append('exponential_fade')
        else:
            techniques.append('linear_crossfade')
        
        # EQ techniques
        if eq['bass_swap']:
            techniques.append('bass_swap')
        if eq['low_cut_incoming']:
            techniques.append('low_cut_on_incoming')
        if eq['high_cut_outgoing']:
            techniques.append('high_cut_on_outgoing')
        if eq['automation_detected'] and not eq['bass_swap']:
            techniques.append('eq_automation')
        
        # Energy techniques
        if energy['has_dip']:
            techniques.append('energy_dip')
        if energy['has_build']:
            techniques.append('energy_build')
        
        # Harmonic
        if harmony['compatible']:
            techniques.append('key_matched')
        if harmony['tension'] < 0.2:
            techniques.append('harmonic_blend')
        
        return techniques
    
    def _assess_effectiveness(self,
                             volume: Dict,
                             eq: Dict,
                             harmony: Dict,
                             energy: Dict,
                             spectral: Dict,
                             beat_aligned: bool) -> Dict:
        """Assess overall transition effectiveness."""
        factors = []
        scores = []
        
        # Beat alignment is crucial
        if beat_aligned:
            factors.append("beat_aligned")
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # Key compatibility matters
        if harmony['compatible']:
            factors.append("key_compatible")
            scores.append(1.0)
        else:
            scores.append(0.6)
        
        # Low harmonic tension is good
        if harmony['tension'] < 0.3:
            factors.append("low_harmonic_tension")
            scores.append(1.0)
        else:
            scores.append(0.7)
        
        # Spectral smoothness
        if spectral['smoothness'] > 0.7:
            factors.append("smooth_spectral_transition")
            scores.append(1.0)
        else:
            scores.append(0.6)
        
        # Bass swap is a sophisticated technique
        if eq['bass_swap']:
            factors.append("clean_bass_swap")
            scores.append(0.9)
        
        # Energy management
        if not energy['has_dip'] or energy['has_build']:
            factors.append("good_energy_management")
            scores.append(0.9)
        else:
            scores.append(0.7)
        
        smoothness = np.mean(scores)
        
        # Complexity based on number of techniques
        techniques_count = len([f for f in factors if f != "beat_aligned"])
        complexity = min(1.0, techniques_count / 4)
        
        return {
            'smoothness': float(smoothness),
            'complexity': float(complexity),
            'factors': factors
        }
    
    def _print_analysis_summary(self, analysis: DeepTransitionAnalysis):
        """Print a human-readable summary."""
        print(f"\n  TECHNIQUES USED: {', '.join(analysis.techniques_used)}")
        print(f"  EFFECTIVENESS FACTORS: {', '.join(analysis.effectiveness_factors)}")
        print(f"  ")
        print(f"  Crossfade Type: {analysis.crossfade_type}")
        print(f"  Duration: {analysis.duration_sec:.1f}s ({analysis.bars_duration} bars)")
        print(f"  Beat Aligned: {'✓' if analysis.beat_aligned else '✗'}")
        print(f"  Downbeat Aligned: {'✓' if analysis.downbeat_aligned else '✗'}")
        print(f"  ")
        print(f"  Key: {analysis.key_outgoing} → {analysis.key_incoming} ({'compatible' if analysis.key_compatible else 'tension'})")
        print(f"  Bass Swap: {'✓ at ' + str(round(analysis.bass_swap_time_sec or 0, 1)) + 's' if analysis.bass_swap_detected else '✗'}")
        print(f"  Energy Flow: {analysis.energy_before:.3f} → {analysis.energy_during:.3f} → {analysis.energy_after:.3f}")
        print(f"  ")
        print(f"  SMOOTHNESS SCORE: {analysis.perceived_smoothness:.0%}")
        print(f"  TECHNIQUE COMPLEXITY: {analysis.technique_complexity:.0%}")
    
    def export_training_data(self, 
                            analyses: List[DeepTransitionAnalysis],
                            output_path: str):
        """Export analyses as training data."""
        training_data = []
        
        for analysis in analyses:
            # Convert to dict for JSON
            data = asdict(analysis)
            training_data.append(data)
        
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        print(f"\nExported {len(analyses)} transition analyses to: {output_path}")


def parse_transitions_from_text(text: str) -> List[TransitionPoint]:
    """
    Parse transition points from user-provided text.
    
    Expected format:
    0:00 Track Name
    3:02 Transition
    3:10 Next Track Name
    """
    import re
    
    lines = text.strip().split('\n')
    transitions = []
    
    current_track = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Parse time and track name
        match = re.match(r'(\d+):(\d+)\s+(.+)', line)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            time_sec = minutes * 60 + seconds
            track_name = match.group(3).strip()
            
            # Handle "transition" markers - treat them as actual transitions
            if track_name.lower() == 'transition':
                if current_track is not None:
                    transitions.append(TransitionPoint(
                        time_sec=time_sec,
                        from_track=current_track,
                        to_track=f"Track_{len(transitions)+2}"
                    ))
                    current_track = f"Track_{len(transitions)+1}"
                else:
                    current_track = "Track_1"
                continue
            
            if current_track is not None:
                # This is a transition from current_track to this track
                transitions.append(TransitionPoint(
                    time_sec=time_sec,
                    from_track=current_track,
                    to_track=track_name
                ))
            
            current_track = track_name
    
    return transitions

