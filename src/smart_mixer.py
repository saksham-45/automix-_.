"""
Smart Mixer - Human-Level Mixing System

Uses all advanced modules for perceptually-correct, smooth transitions.
"""
import numpy as np
import soundfile as sf
import librosa
import json
import time
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from src.smart_transition_finder import SmartTransitionFinder, TransitionPair
from src.beat_aligner import BeatAligner
from src.psychoacoustics import PsychoacousticAnalyzer
from src.harmonic_analyzer import HarmonicAnalyzer
from src.structure_analyzer import StructureAnalyzer
from src.advanced_beatmatcher import AdvancedBeatMatcher
from src.dynamic_processor import DynamicProcessor
from src.transition_strategist import TransitionStrategist
from src.crossfade_engine import CrossfadeEngine
from src.quality_assessor import QualityAssessor
from src.stem_separator import StemSeparator
from src.technique_executor import TechniqueExecutor

# Debug logging helper
DEBUG_LOG_PATH = str(Path(__file__).parent.parent / '.cursor' / 'debug.log')
def debug_log(message, data, hypothesis_id="ALL", location=""):
    """Helper function for debug logging."""
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
        with open(DEBUG_LOG_PATH, 'a') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000)
            }) + '\n')
    except Exception as e:
        pass  # Silently fail to not disrupt main flow


class SmartMixer:
    """
    Creates human-level smooth transitions using all advanced analysis modules.
    """
    
    def __init__(self, sr: int = 44100, hop_length: int = 512):
        import time, json
        log_path = str(Path(__file__).parent.parent / '.cursor' / 'debug.log')
        
        #region agent log
        init_start = time.time()
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"smart_mixer.py:27","message":"SmartMixer init start","data":{},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        self.sr = sr
        self.hop_length = hop_length
        
        # Initialize all modules (lazy initialization for speed)
        #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"smart_mixer.py:35","message":"Initializing modules","data":{},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        self.transition_finder = SmartTransitionFinder(sr=sr, hop_length=hop_length)
        self.beat_aligner = BeatAligner(sr=sr, hop_length=hop_length)
        
        #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"smart_mixer.py:39","message":"Psychoacoustics init","data":{},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        self.psychoacoustics = PsychoacousticAnalyzer(sr=sr)
        self.harmonic_analyzer = HarmonicAnalyzer()
        
        #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"smart_mixer.py:43","message":"Structure analyzer init","data":{},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        self.structure_analyzer = StructureAnalyzer(sr=sr, hop_length=hop_length)
        self.advanced_beatmatcher = AdvancedBeatMatcher(sr=sr, hop_length=hop_length)
        self.dynamic_processor = DynamicProcessor(sr=sr)
        self.transition_strategist = TransitionStrategist()
        self.crossfade_engine = CrossfadeEngine(sr=sr)
        self.quality_assessor = QualityAssessor(sr=sr)
        self.technique_executor = TechniqueExecutor(sr=sr)
        
        # Load stem separation config
        try:
            import yaml
            config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                stem_config = config.get('analysis', {}).get('stem_separation', {})
                self.stem_separation_enabled = stem_config.get('enabled', True)
                self.stem_model = stem_config.get('model', 'htdemucs')
                self.separate_stems = stem_config.get('separate_stems', ['drums', 'bass'])
                self.drums_fade_ratio = stem_config.get('drums_fade_ratio', 0.25)
                self.bass_fade_ratio = stem_config.get('bass_fade_ratio', 0.5)
            else:
                # Defaults
                self.stem_separation_enabled = True
                self.stem_model = 'htdemucs'
                self.separate_stems = ['drums', 'bass']
                self.drums_fade_ratio = 0.25
                self.bass_fade_ratio = 0.5
        except Exception as e:
            # Fallback to defaults
            self.stem_separation_enabled = True
            self.stem_model = 'htdemucs'
            self.separate_stems = ['drums', 'bass']
            self.drums_fade_ratio = 0.25
            self.bass_fade_ratio = 0.5
        
        # Initialize stem separator (lazy - only if enabled)
        self.stem_separator = None
        if self.stem_separation_enabled:
            try:
                self.stem_separator = StemSeparator(model_name=self.stem_model)
            except Exception as e:
                print(f"  ⚠ Stem separation not available: {e}")
                self.stem_separation_enabled = False
        
        # Initialize Superhuman DJ Engine (advanced features 2-6)
        self.superhuman_engine = None
        self.superhuman_enabled = True
        try:
            from src.superhuman_engine import SuperhumanDJEngine
            self.superhuman_engine = SuperhumanDJEngine(sr=sr)
            # Apply base defaults
            self.superhuman_engine.configure(
                enable_micro_timing=True,
                enable_spectral_intelligence=True,
                enable_hybrid_techniques=True,
                enable_stem_orchestration=True,
                enable_montecarlo_optimization=True,
                creativity_level=0.6
            )
            # Apply config overrides if available
            try:
                if 'config' in locals():
                    super_cfg = config.get('superhuman', {})
                else:
                    import yaml
                    cfg_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
                    super_cfg = {}
                    if cfg_path.exists():
                        with open(cfg_path) as f:
                            full_cfg = yaml.safe_load(f)
                        super_cfg = full_cfg.get('superhuman', {}) or {}
                if super_cfg:
                    self.superhuman_engine.configure(
                        vocal_bed_swap_enabled=super_cfg.get('vocal_bed_swap_enabled', True),
                        max_tempo_shift_pct=super_cfg.get('max_tempo_shift_pct', 0.06),
                        allow_simultaneous_vocals=super_cfg.get('allow_simultaneous_vocals', 'rare'),
                        mix_at_level=super_cfg.get('mix_at_level', True),
                        bpm_matching_always=super_cfg.get('bpm_matching_always', True),
                        bpm_matching_min_diff=float(super_cfg.get('bpm_matching_min_diff', 1.0)),
                        technique_execution_ratio=float(super_cfg.get('technique_execution_ratio', 0.62)),
                        avoid_consecutive_stem_orchestration=bool(super_cfg.get('avoid_consecutive_stem_orchestration', True)),
                        technique_diversity_lookback=int(super_cfg.get('technique_diversity_lookback', 4)),
                        technique_diversity_attempts=int(super_cfg.get('technique_diversity_attempts', 4)),
                        avoid_high_risk_vocal_techniques=bool(super_cfg.get('avoid_high_risk_vocal_techniques', True)),
                        vocal_overlap_soft_guard=float(super_cfg.get('vocal_overlap_soft_guard', 0.55)),
                        vocal_overlap_hard_guard=float(super_cfg.get('vocal_overlap_hard_guard', 0.75)),
                        key_modulation_enabled=super_cfg.get('key_modulation_enabled', True),
                        key_modulation_max_semitones=int(super_cfg.get('key_modulation_max_semitones', 2)),
                        key_modulation_only_when_incompatible=super_cfg.get('key_modulation_only_when_incompatible', True),
                    )
            except Exception:
                pass
        except Exception as e:
            print(f"  ⚠ Superhuman engine not available: {e}")
            self.superhuman_enabled = False
        
        #region agent log
        init_time = time.time() - init_start
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"smart_mixer.py:51","message":"SmartMixer init complete","data":{"time_sec":init_time},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
    
    def create_smooth_mix(self,
                         song_a_path: str,
                         song_b_path: str,
                         transition_duration: Optional[float] = None,
                         song_a_analysis: Optional[Dict] = None,
                         song_b_analysis: Optional[Dict] = None,
                         ai_transition_data: Optional[Dict] = None,
                         return_metadata: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Create a human-level smooth mix using all advanced modules.
        """
        print("\n" + "="*60)
        print("HUMAN-LEVEL MIXING SYSTEM")
        print("="*60)
        
        # Load audio
        import time
        log_path = '/Users/saksham/untitled folder 7/.cursor/debug.log'
        os.makedirs(os.path.dirname(log_path), exist_ok=True)  # Ensure directory exists
        start_time = time.time()
        
        #region agent log
        with open(log_path, 'a') as f:
            import json
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"smart_mixer.py:59","message":"Starting audio load","data":{"time":start_time},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        y_a, sr_a = librosa.load(song_a_path, sr=self.sr)
        
        #region agent log
        load_a_time = time.time() - start_time
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"smart_mixer.py:65","message":"Song A loaded","data":{"duration_sec":len(y_a)/self.sr,"load_time_sec":load_a_time},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        y_b, sr_b = librosa.load(song_b_path, sr=self.sr)
        
        #region agent log
        load_b_time = time.time() - start_time - load_a_time
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"smart_mixer.py:69","message":"Song B loaded","data":{"duration_sec":len(y_b)/self.sr,"load_time_sec":load_b_time},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Analyze both songs comprehensively (lightweight version for speed)
        print("\n[1/8] Analyzing Song A...")
        analysis_start = time.time()
        analysis_a = self._analyze_song_fast(y_a, song_a_analysis)
        
        #region agent log
        analysis_a_time = time.time() - analysis_start
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"smart_mixer.py:75","message":"Song A analysis complete","data":{"time_sec":analysis_a_time},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        print("[2/8] Analyzing Song B...")
        analysis_start = time.time()
        analysis_b = self._analyze_song_fast(y_b, song_b_analysis)
        
        #region agent log
        analysis_b_time = time.time() - analysis_start
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"smart_mixer.py:81","message":"Song B analysis complete","data":{"time_sec":analysis_b_time},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Find optimal transition points using structure and harmonics
        print("[3/8] Finding optimal transition points...")
        transition_pair = self._find_optimal_transition_points(
            y_a, y_b, analysis_a, analysis_b
        )
        
        # Advanced beat matching (only analyze segments around transition points, not full songs)
        print("[4/8] Beat matching and phase alignment...")
        
        #region agent log
        beat_start = time.time()
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"smart_mixer.py:81","message":"Starting beat matching","data":{"point_a":transition_pair.song_a_point.time_sec,"point_b":transition_pair.song_b_point.time_sec},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Extract 30-second segments around transition points for beat matching
        window_sec = 15
        point_a = transition_pair.song_a_point.time_sec
        point_b = transition_pair.song_b_point.time_sec
        
        start_a = max(0, int((point_a - window_sec) * self.sr))
        end_a = min(len(y_a), int((point_a + window_sec) * self.sr))
        y_a_seg = y_a[start_a:end_a]
        
        start_b = max(0, int((point_b - window_sec) * self.sr))
        end_b = min(len(y_b), int((point_b + window_sec) * self.sr))
        y_b_seg = y_b[start_b:end_b]
        
        beat_match = self.advanced_beatmatcher.match_beats(
            y_a_seg, y_b_seg,
            window_sec,  # Adjusted point (in segment)
            window_sec   # Adjusted point
        )
        
        # Adjust beat match times back to full song coordinates
        beat_match['aligned_point_a_sec'] = point_a + (beat_match['aligned_point_a_sec'] - window_sec)
        beat_match['aligned_point_b_sec'] = point_b + (beat_match['aligned_point_b_sec'] - window_sec)
        
        #region agent log
        beat_time = time.time() - beat_start
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"smart_mixer.py:105","message":"Beat matching complete","data":{"time_sec":beat_time,"raw_aligned_a":beat_match.get('aligned_point_a_sec'),"raw_aligned_b":beat_match.get('aligned_point_b_sec')},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Clamp to valid ranges (beat matching can return negative values or out of bounds)
        aligned_a = max(0.0, min(beat_match['aligned_point_a_sec'], len(y_a) / self.sr))
        aligned_b = max(0.0, min(beat_match['aligned_point_b_sec'], len(y_b) / self.sr))
        
        #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"smart_mixer.py:185","message":"Aligned points clamped","data":{"aligned_a_sec":aligned_a,"aligned_b_sec":aligned_b,"y_a_duration":len(y_a)/self.sr,"y_b_duration":len(y_b)/self.sr},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Select transition technique
        print("[5/8] Selecting transition technique...")
        
        #region agent log
        clash_start = time.time()
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"smart_mixer.py:88","message":"Starting frequency clash analysis","data":{"y_a_len":len(y_a),"y_b_len":len(y_b)},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Use samples for clash analysis (5 seconds max each - optimized)
        sample_len = min(5 * self.sr, len(y_a), len(y_b))
        y_a_clash = y_a[:sample_len]
        y_b_clash = y_b[:sample_len]
        clash_analysis = self.psychoacoustics.predict_frequency_clash(y_a_clash, y_b_clash)
        
        #region agent log
        clash_time = time.time() - clash_start
        clash_score = clash_analysis.get('clash_score', 0.5)
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"smart_mixer.py:92","message":"Frequency clash analysis complete","data":{"time_sec":clash_time,"clash_score":clash_score},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        technique = self.transition_strategist.select_technique(
            analysis_a['key'],
            analysis_b['key'],
            analysis_a['tempo'],
            analysis_b['tempo'],
            transition_pair.song_a_point.structural_label,
            transition_pair.song_b_point.structural_label,
            transition_pair.song_a_point.energy,
            transition_pair.song_b_point.energy,
            clash_score
        )
        
        # Get technique-specific parameters
        technique_params = self.transition_strategist.get_technique_parameters(
            technique['technique_name'],
            {'duration_sec': transition_duration or technique['duration_sec']}
        )
        
        # Determine transition duration
        if transition_duration is None:
            transition_duration = technique['duration_sec']
        
        print(f"  Technique: {technique['technique_name']}")
        print(f"  Duration: {transition_duration:.1f}s ({technique['duration_bars']} bars)")
        
        # Extract segments
        print("[6/8] Extracting and processing segments...")
        
        #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"smart_mixer.py:224","message":"Before segment extraction","data":{"aligned_a_sec":aligned_a,"aligned_b_sec":aligned_b,"transition_duration":transition_duration,"y_a_len":len(y_a),"y_b_len":len(y_b)},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        seg_a, seg_b = self._extract_segments(y_a, y_b, aligned_a, aligned_b, transition_duration)
        
        #region agent log
        seg_a_rms = float(np.sqrt(np.mean(seg_a**2)))
        seg_b_rms = float(np.sqrt(np.mean(seg_b**2)))
        seg_a_max = float(np.max(np.abs(seg_a)))
        seg_b_max = float(np.max(np.abs(seg_b)))
        seg_a_zeros = int(np.sum(np.abs(seg_a) < 0.001))
        seg_b_zeros = int(np.sum(np.abs(seg_b) < 0.001))
        #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"smart_mixer.py:230","message":"Segments extracted","data":{"seg_a_len":len(seg_a),"seg_b_len":len(seg_b),"seg_a_shape":str(seg_a.shape),"seg_b_shape":str(seg_b.shape),"seg_a_rms":seg_a_rms,"seg_b_rms":seg_b_rms,"seg_a_max":seg_a_max,"seg_b_max":seg_b_max,"seg_a_zeros":seg_a_zeros,"seg_b_zeros":seg_b_zeros},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # ========== BEAT ALIGNMENT (Time Stretching) ==========
        # Prevent beat drift by matching tempos during the transition
        try:
            tempo_a = analysis_a['tempo']
            tempo_b = analysis_b['tempo']
            if abs(tempo_a - tempo_b) > 1.0:
                print(f"  ⚡ Aligning tempos: {tempo_a:.1f} vs {tempo_b:.1f} BPM")
                
                # Stretch the incoming track (seg_b) to match outgoing (seg_a)
                # Or stretch both to average? Let's stretch B to match A for smoother handoff
                rate = tempo_b / tempo_a
                
                # Check if rate is reasonable (don't stretch more than +/- 20%)
                if 0.8 <= rate <= 1.2:
                    # Time stretch seg_b
                    # Handle stereo if needed
                    if seg_b.ndim == 2:
                        # Process channels independently
                        seg_b_stretched_l = librosa.effects.time_stretch(seg_b[:, 0], rate=rate)
                        seg_b_stretched_r = librosa.effects.time_stretch(seg_b[:, 1], rate=rate)
                        # Ensure equal length
                        min_len = min(len(seg_b_stretched_l), len(seg_b_stretched_r))
                        seg_b = np.column_stack([seg_b_stretched_l[:min_len], seg_b_stretched_r[:min_len]])
                    else:
                        seg_b = librosa.effects.time_stretch(seg_b, rate=rate)
                    
                    # Truncate or pad to match seg_a length for perfect alignment
                    target_len = len(seg_a)
                    if len(seg_b) > target_len:
                        seg_b = seg_b[:target_len]
                    elif len(seg_b) < target_len:
                        pad_width = target_len - len(seg_b)
                        if seg_b.ndim == 2:
                            seg_b = np.pad(seg_b, ((0, pad_width), (0, 0)))
                        else:
                            seg_b = np.pad(seg_b, (0, pad_width))
                            
                    print(f"  ✓ Song B time-stretched to match Song A (rate={rate:.3f})")
                else:
                    print(f"  ⚠ Tempo difference too large for stretching ({rate:.2f}x), skipping alignment")
        except Exception as e:
            print(f"  ⚠ Beat alignment failed: {e}")
        
        # ========== LOUDNESS MATCHING (LUFS) ==========
        # Match perceived loudness of both segments so transitions don't have volume dips/bumps.
        # This is the single most important step for professional-sounding mixes.
        try:
            seg_a_mono_lufs = np.mean(seg_a, axis=1) if seg_a.ndim > 1 else seg_a
            seg_b_mono_lufs = np.mean(seg_b, axis=1) if seg_b.ndim > 1 else seg_b
            lufs_a = self.psychoacoustics.analyze_loudness_lufs(seg_a_mono_lufs)
            lufs_b = self.psychoacoustics.analyze_loudness_lufs(seg_b_mono_lufs)
            lufs_diff = lufs_a['integrated_lufs'] - lufs_b['integrated_lufs']
            # Clamp gain adjustment to ±6dB to avoid extreme corrections
            gain_db = max(-6.0, min(6.0, lufs_diff))
            if abs(gain_db) > 0.5:  # Only adjust if difference is audible
                gain_linear = 10 ** (gain_db / 20.0)
                seg_b = seg_b * gain_linear
                print(f"  ✓ Loudness matched: Song B adjusted by {gain_db:+.1f} dB (A={lufs_a['integrated_lufs']:.1f} LUFS, B={lufs_b['integrated_lufs']:.1f} LUFS)")
            else:
                print(f"  ✓ Loudness already matched (diff={lufs_diff:.1f} dB)")
        except Exception as e:
            print(f"  ⚠ Loudness matching failed: {e}")

        # Initialize stems variables (for techniques that need them)
        seg_a_stems = None
        seg_b_stems = None
        
        # Stem separation for smooth transitions (prevent drums/bass from carrying over)
        if self.stem_separation_enabled and self.stem_separator is not None:
            print("  Separating stems to fade out heavy instruments...")
            try:
                #region agent log
                stem_start = time.time()
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"STEM","location":"smart_mixer.py:312","message":"Starting stem separation","data":{"seg_a_len":len(seg_a)},"timestamp":int(time.time()*1000)}) + '\n')
                #endregion
                
                # Separate stems from song A (outgoing track)
                seg_a_stems = self.stem_separator.separate_segment(seg_a, self.sr)
                
                # Also separate stems from song B to detect when vocals start
                seg_b_stems = self.stem_separator.separate_segment(seg_b, self.sr)
                
                #region agent log
                stem_time = time.time() - stem_start
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"STEM","location":"smart_mixer.py:318","message":"Stem separation complete","data":{"time_sec":stem_time,"stems_available":list(seg_a_stems.keys())},"timestamp":int(time.time()*1000)}) + '\n')
                #endregion
                
                # Detect when Song B's vocals start
                vocal_start_time_ratio = self._detect_vocal_start_time(
                    seg_b_stems.get('vocals', None), 
                    len(seg_b)
                )
                
                #region agent log
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"VOCAL","location":"smart_mixer.py:325","message":"Vocal start detected","data":{"vocal_start_ratio":vocal_start_time_ratio},"timestamp":int(time.time()*1000)}) + '\n')
                #endregion
                
                # Create fast fade curves for drums and bass
                n_samples = len(seg_a)
                drums_fade = self.crossfade_engine.create_fast_fade(n_samples, fade_out_ratio=self.drums_fade_ratio)
                bass_fade = self.crossfade_engine.create_fast_fade(n_samples, fade_out_ratio=self.bass_fade_ratio)
                
                # Moderate vocal fade: complete by when Song B vocals start, ~25% fade duration (gentler than aggressive)
                fade_complete_by = max(0.25, vocal_start_time_ratio)
                vocal_fade = self.crossfade_engine.create_moderate_vocal_fade(
                    n_samples,
                    fade_complete_by_ratio=fade_complete_by,
                    fade_duration_ratio=0.25
                )
                
                # Apply fades to stems
                seg_a_processed = np.zeros_like(seg_a)
                
                # Process drums with fast fade
                if 'drums' in seg_a_stems and 'drums' in self.separate_stems:
                    # Ensure fade curve matches audio shape
                    if drums_fade.ndim == 1:
                        drums_fade_2d = drums_fade[:, np.newaxis]  # [samples, 1]
                    else:
                        drums_fade_2d = drums_fade
                    # Ensure shapes match
                    if seg_a_stems['drums'].shape[1] == drums_fade_2d.shape[1] or drums_fade_2d.shape[1] == 1:
                        drums_faded = seg_a_stems['drums'] * drums_fade_2d
                    else:
                        drums_faded = seg_a_stems['drums'] * drums_fade[:, np.newaxis]
                    seg_a_processed += drums_faded
                
                # Process bass with fade
                if 'bass' in seg_a_stems and 'bass' in self.separate_stems:
                    # Ensure fade curve matches audio shape
                    if bass_fade.ndim == 1:
                        bass_fade_2d = bass_fade[:, np.newaxis]  # [samples, 1]
                    else:
                        bass_fade_2d = bass_fade
                    # Ensure shapes match
                    if seg_a_stems['bass'].shape[1] == bass_fade_2d.shape[1] or bass_fade_2d.shape[1] == 1:
                        bass_faded = seg_a_stems['bass'] * bass_fade_2d
                    else:
                        bass_faded = seg_a_stems['bass'] * bass_fade[:, np.newaxis]
                    seg_a_processed += bass_faded
                
                # Process vocals with moderate fade (phrase-friendly, not sudden)
                if 'vocals' in seg_a_stems:
                    if vocal_fade.ndim == 1:
                        vocal_fade_2d = vocal_fade[:, np.newaxis]
                    else:
                        vocal_fade_2d = vocal_fade
                    # Ensure shapes match
                    if seg_a_stems['vocals'].shape[1] == vocal_fade_2d.shape[1] or vocal_fade_2d.shape[1] == 1:
                        vocals_faded = seg_a_stems['vocals'] * vocal_fade_2d
                    else:
                        vocals_faded = seg_a_stems['vocals'] * vocal_fade[:, np.newaxis]
                    seg_a_processed += vocals_faded
                    print(f"  ✓ Vocals faded moderately (Song B vocals start at {vocal_start_time_ratio*100:.1f}% of transition)")
                
                # Keep other elements with normal processing (no special fade)
                if 'other' in seg_a_stems:
                    seg_a_processed += seg_a_stems['other']
                
                # Replace seg_a with processed version
                seg_a = seg_a_processed
                
                #region agent log
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"STEM","location":"smart_mixer.py:340","message":"Stems processed and recombined","data":{"seg_a_rms_after":float(np.sqrt(np.mean(seg_a**2)))},"timestamp":int(time.time()*1000)}) + '\n')
                #endregion
                
                print("  ✓ Stems separated and drums/bass faded out")
            except Exception as e:
                print(f"  ⚠ Stem separation failed: {e}")
                print("  → Continuing with original audio")
                # Continue with original seg_a if separation fails
                seg_a_stems = None
                seg_b_stems = None
        
        # Apply dynamic EQ if needed
        if clash_score > 0.3:
            print("  Applying dynamic EQ to prevent frequency clashes...")
            
            #region agent log
            eq_start = time.time()
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"smart_mixer.py:116","message":"Starting EQ clash analysis","data":{"seg_a_len":len(seg_a),"seg_b_len":len(seg_b)},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            eq_analysis = self.dynamic_processor.analyze_frequency_clash(seg_a, seg_b)
            
            #region agent log
            eq_time = time.time() - eq_start
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"smart_mixer.py:120","message":"EQ clash analysis complete","data":{"time_sec":eq_time},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            if 'bass_swap' in eq_analysis['recommendations']:
                seg_a, seg_b = self._apply_bass_swap(seg_a, seg_b, transition_duration)
        
        # Execute selected technique using TechniqueExecutor
        print(f"[7/8] Executing {technique['technique_name']} technique...")
        
        # Check if technique needs stems but we don't have them
        stem_required_techniques = ['staggered_stem_mix', 'partial_stem_separation', 'vocal_layering']
        if technique['technique_name'] in stem_required_techniques and (seg_a_stems is None or seg_b_stems is None):
            # Try to separate stems if needed
            if self.stem_separation_enabled and self.stem_separator is not None:
                try:
                    print(f"  Separating stems for {technique['technique_name']} technique...")
                    seg_a_stems = self.stem_separator.separate_segment(seg_a, self.sr)
                    seg_b_stems = self.stem_separator.separate_segment(seg_b, self.sr)
                except Exception as e:
                    print(f"  ⚠ Stem separation failed for technique: {e}")
                    print(f"  → Falling back to long_blend")
                    technique['technique_name'] = 'long_blend'
                    technique_params = self.transition_strategist.get_technique_parameters('long_blend', {'duration_sec': transition_duration})
        
        #region agent log
        exec_start = time.time()
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"TECHNIQUE","location":"smart_mixer.py:442","message":"Starting technique execution","data":{"technique":technique["technique_name"],"needs_stems":technique["technique_name"] in stem_required_techniques,"has_stems":seg_a_stems is not None and seg_b_stems is not None},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Execute technique
        mixed = self.technique_executor.execute(
            technique['technique_name'],
            seg_a,
            seg_b,
            technique_params,
            seg_a_stems=seg_a_stems,
            seg_b_stems=seg_b_stems
        )
        
        #region agent log
        exec_time = time.time() - exec_start
        mixed_rms = float(np.sqrt(np.mean(mixed**2))) if len(mixed) > 0 else 0
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"TECHNIQUE","location":"smart_mixer.py:465","message":"Technique execution complete","data":{"technique":technique["technique_name"],"time_sec":exec_time,"mixed_len":len(mixed),"mixed_rms":mixed_rms},"timestamp":int(time.time()*1000)}) + "\n")
        #endregion
        
        print(f"[8/8] Mixing with {technique['technique_name']}...")
        
        # Post-transition quality validation
        #region agent log
        quality_start = time.time()
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"QUALITY","location":"smart_mixer.py:496","message":"Starting quality validation","data":{"technique":technique["technique_name"],"mixed_len":len(mixed),"mixed_shape":str(mixed.shape) if hasattr(mixed, 'shape') else 'unknown',"seg_a_len":len(seg_a) if seg_a is not None else 0,"seg_b_len":len(seg_b) if seg_b is not None else 0},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        try:
            #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"QUALITY","location":"smart_mixer.py:506","message":"Calling assess_transition_quality","data":{"key_a":analysis_a.get('key'),"key_b":analysis_b.get('key')},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            # Convert to mono if stereo for quality assessment
            mixed_mono = mixed
            if mixed.ndim > 1:
                mixed_mono = np.mean(mixed, axis=1)
            seg_a_mono = seg_a
            if seg_a is not None and seg_a.ndim > 1:
                seg_a_mono = np.mean(seg_a, axis=1)
            seg_b_mono = seg_b
            if seg_b is not None and seg_b.ndim > 1:
                seg_b_mono = np.mean(seg_b, axis=1)
            
            quality_assessment = self.quality_assessor.assess_transition_quality(
                mixed_mono,
                y_a=seg_a_mono,
                y_b=seg_b_mono,
                key_a=analysis_a.get('key'),
                key_b=analysis_b.get('key')
            )
        except Exception as e:
            #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"QUALITY","location":"smart_mixer.py:525","message":"Quality assessment failed","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            print(f"  ⚠ Quality assessment failed: {e}")
            quality_assessment = {
                'overall_score': 0.5,
                'smoothness': {'score': 0.5, 'analysis': 'unknown'},
                'clarity': {'score': 0.5, 'analysis': 'unknown'},
                'harmonic_tension': 0.5,
                'frequency_balance': {'score': 0.5, 'analysis': 'unknown'},
                'energy_continuity': {'score': 0.5, 'analysis': 'unknown'},
                'quality_rating': 'unknown'
            }
        
        #region agent log
        quality_time = time.time() - quality_start
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"QUALITY","location":"smart_mixer.py:505","message":"Quality assessment complete","data":{"overall_score":quality_assessment["overall_score"],"smoothness":quality_assessment["smoothness"]["score"],"clarity":quality_assessment["clarity"]["score"],"harmonic_tension":quality_assessment["harmonic_tension"],"frequency_balance":quality_assessment["frequency_balance"]["score"],"energy_continuity":quality_assessment["energy_continuity"]["score"],"quality_rating":quality_assessment["quality_rating"],"time_sec":quality_time},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        quality_threshold = 0.6  # Minimum acceptable quality score
        if quality_assessment['overall_score'] < quality_threshold:
            print(f"  ⚠ Quality score {quality_assessment['overall_score']:.2f} below threshold {quality_threshold}")
            print(f"     Issues: {quality_assessment['quality_rating']}")
            #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"QUALITY","location":"smart_mixer.py:516","message":"Quality below threshold - potential retry needed","data":{"overall_score":quality_assessment["overall_score"],"threshold":quality_threshold,"quality_rating":quality_assessment["quality_rating"]},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
        else:
            print(f"  ✓ Quality score: {quality_assessment['overall_score']:.2f} ({quality_assessment['quality_rating']})")
        
        print("\n✓ Mix complete")
        
        # Build final mix with context
        # FIXED: Extract context correctly - ensure NO overlap with mixed segment
        context_before = int(10 * self.sr)
        context_after = int(10 * self.sr)
        
        # Context A: 10 seconds BEFORE the START of seg_a (not before transition point)
        # seg_a starts at: aligned_a - transition_duration
        seg_a_start_in_song_a = int(aligned_a * self.sr) - len(seg_a)
        ctx_a_start = max(0, seg_a_start_in_song_a - context_before)
        ctx_a_end = seg_a_start_in_song_a  # End exactly where seg_a starts (no overlap)
        ctx_a = y_a[ctx_a_start:ctx_a_end]
        
        #region agent log
        # Test hypothesis A and C: Check for overlap between ctx_a and seg_a
        seg_a_start_in_song_a = int(aligned_a * self.sr) - len(seg_a)
        seg_a_end_in_song_a = int(aligned_a * self.sr)
        overlap_a = max(0, min(ctx_a_end, seg_a_end_in_song_a) - max(ctx_a_start, seg_a_start_in_song_a))
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"smart_mixer.py:357","message":"Context A extracted - checking overlap","data":{"ctx_a_start":ctx_a_start,"ctx_a_end":ctx_a_end,"ctx_a_len":len(ctx_a),"seg_a_start_in_song_a":seg_a_start_in_song_a,"seg_a_end_in_song_a":seg_a_end_in_song_a,"seg_a_len":len(seg_a),"overlap_samples":overlap_a,"overlap_sec":overlap_a/self.sr,"aligned_a_sec":aligned_a},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Context B: Starts IMMEDIATELY after mixed segment ends
        # The mixed segment contains: end of A + beginning of B overlapping for transition_duration seconds
        # seg_b was extracted from song B starting at aligned_b, for transition_duration seconds
        # So after the mix, we continue song B from: aligned_b + transition_duration
        transition_samples = int(transition_duration * self.sr)
        seg_b_start_in_song_b = int(aligned_b * self.sr)
        seg_b_end_in_song_b = seg_b_start_in_song_b + len(seg_b)
        ctx_b_start = seg_b_end_in_song_b
        ctx_b_end = min(len(y_b), ctx_b_start + context_after)
        ctx_b = y_b[ctx_b_start:ctx_b_end]
        
        #region agent log
        # Test hypothesis E: Check for overlap between ctx_b and seg_b
        overlap_b = max(0, min(ctx_b_end, seg_b_end_in_song_b) - max(ctx_b_start, seg_b_start_in_song_b))
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"smart_mixer.py:375","message":"Context B extracted - checking overlap","data":{"ctx_b_start":ctx_b_start,"ctx_b_end":ctx_b_end,"ctx_b_len":len(ctx_b),"seg_b_start_in_song_b":seg_b_start_in_song_b,"seg_b_end_in_song_b":seg_b_end_in_song_b,"seg_b_len":len(seg_b),"overlap_samples":overlap_b,"overlap_sec":overlap_b/self.sr,"aligned_b_sec":aligned_b,"transition_duration":transition_duration},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Ensure all segments are stereo
        if ctx_a.ndim == 1:
            ctx_a = np.column_stack([ctx_a, ctx_a])
        if ctx_b.ndim == 1:
            ctx_b = np.column_stack([ctx_b, ctx_b])
        if mixed.ndim == 1:
            mixed = np.column_stack([mixed, mixed])
        
        #region agent log
        final_start = time.time()
        # Test all hypotheses: Log final concatenation details
        ctx_a_rms = float(np.sqrt(np.mean(ctx_a**2))) if len(ctx_a) > 0 else 0
        mixed_rms_final = float(np.sqrt(np.mean(mixed**2))) if len(mixed) > 0 else 0
        ctx_b_rms = float(np.sqrt(np.mean(ctx_b**2))) if len(ctx_b) > 0 else 0
        ctx_a_max = float(np.max(np.abs(ctx_a))) if len(ctx_a) > 0 else 0
        mixed_max_final = float(np.max(np.abs(mixed))) if len(mixed) > 0 else 0
        ctx_b_max = float(np.max(np.abs(ctx_b))) if len(ctx_b) > 0 else 0
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"smart_mixer.py:395","message":"Before concatenation - final check","data":{"ctx_a_len":len(ctx_a),"mixed_len":len(mixed),"ctx_b_len":len(ctx_b),"ctx_a_rms":ctx_a_rms,"mixed_rms":mixed_rms_final,"ctx_b_rms":ctx_b_rms,"ctx_a_max":ctx_a_max,"mixed_max":mixed_max_final,"ctx_b_max":ctx_b_max,"ctx_a_duration_sec":len(ctx_a)/self.sr,"mixed_duration_sec":len(mixed)/self.sr,"ctx_b_duration_sec":len(ctx_b)/self.sr},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion

        # ========== BOUNDARY CROSSFADES (eliminate clicks at splice points) ==========
        # Instead of hard-splicing, apply short 50ms crossfades at the junctions
        xfade_samples = min(int(0.05 * self.sr), len(ctx_a) // 2, len(mixed) // 2, len(ctx_b) // 2)  # 50ms
        
        if xfade_samples > 10 and len(ctx_a) > 0 and len(mixed) > 0 and len(ctx_b) > 0:
            # Build crossfade curves
            fade_out = np.linspace(1.0, 0.0, xfade_samples)
            fade_in = np.linspace(0.0, 1.0, xfade_samples)
            if ctx_a.ndim == 2:
                fade_out_2d = fade_out[:, np.newaxis]
                fade_in_2d = fade_in[:, np.newaxis]
            else:
                fade_out_2d = fade_out
                fade_in_2d = fade_in
            
            # Junction 1: ctx_a → mixed
            # Overlap last xfade_samples of ctx_a with first xfade_samples of mixed
            overlap_1 = ctx_a[-xfade_samples:] * fade_out_2d + mixed[:xfade_samples] * fade_in_2d
            
            # Junction 2: mixed → ctx_b
            overlap_2 = mixed[-xfade_samples:] * fade_out_2d + ctx_b[:xfade_samples] * fade_in_2d
            
            # Assemble: ctx_a (minus overlap tail) + overlap_1 + mixed (minus both ends) + overlap_2 + ctx_b (minus overlap head)
            final = np.concatenate([
                ctx_a[:-xfade_samples],
                overlap_1,
                mixed[xfade_samples:-xfade_samples] if len(mixed) > 2 * xfade_samples else np.empty((0, ctx_a.shape[1]) if ctx_a.ndim == 2 else (0,)),
                overlap_2,
                ctx_b[xfade_samples:]
            ], axis=0)
        else:
            final = np.concatenate([ctx_a, mixed, ctx_b], axis=0)

        # Compute metadata for resume point in Song B (relative to y_b)
        mix_metadata: Dict = {
            # Start of transition content in Song A snippet coordinates.
            "a_transition_start_samples": int(ctx_a_start),
            "a_transition_start_sec": float(ctx_a_start / self.sr),
            # End of consumed Song B content in Song B snippet coordinates.
            "b_resume_offset_samples": int(ctx_b_end),
            "b_resume_offset_sec": float(ctx_b_end / self.sr),
        }

        #region agent log
        final_time = time.time() - final_start
        final_rms = float(np.sqrt(np.mean(final**2))) if len(final) > 0 else 0
        final_max = float(np.max(np.abs(final))) if len(final) > 0 else 0
        # Check for potential double-layering: look at RMS values at transition boundaries
        transition_boundary_1_start = len(ctx_a) - min(1000, len(ctx_a))
        transition_boundary_1_end = len(ctx_a) + min(1000, len(mixed))
        transition_boundary_2_start = len(ctx_a) + len(mixed) - min(1000, len(mixed))
        transition_boundary_2_end = len(ctx_a) + len(mixed) + min(1000, len(ctx_b))
        boundary_1_rms = float(np.sqrt(np.mean(final[transition_boundary_1_start:transition_boundary_1_end]**2))) if transition_boundary_1_end > transition_boundary_1_start else 0
        boundary_2_rms = float(np.sqrt(np.mean(final[transition_boundary_2_start:transition_boundary_2_end]**2))) if transition_boundary_2_end > transition_boundary_2_start else 0
        with open(log_path, 'a') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "ALL",
                "location": "smart_mixer.py:407",
                "message": "Concatenation complete - final analysis",
                "data": {
                    "final_len": len(final),
                    "final_duration_sec": len(final) / self.sr,
                    "time_sec": final_time,
                    "final_rms": final_rms,
                    "final_max": final_max,
                    "boundary_1_rms": boundary_1_rms,
                    "boundary_2_rms": boundary_2_rms,
                    "boundary_1_max": float(np.max(np.abs(final[transition_boundary_1_start:transition_boundary_1_end]))) if transition_boundary_1_end > transition_boundary_1_start else 0,
                    "boundary_2_max": float(np.max(np.abs(final[transition_boundary_2_start:transition_boundary_2_end]))) if transition_boundary_2_end > transition_boundary_2_start else 0,
                    "a_transition_start_samples": mix_metadata["a_transition_start_samples"],
                    "a_transition_start_sec": mix_metadata["a_transition_start_sec"],
                    "b_resume_offset_samples": mix_metadata["b_resume_offset_samples"],
                    "b_resume_offset_sec": mix_metadata["b_resume_offset_sec"],
                },
                "timestamp": int(time.time() * 1000)
            }) + '\n')
        #endregion

        if return_metadata:
            return final, mix_metadata
        return final
    
    def _analyze_song_fast(self, y: np.ndarray, existing_analysis: Optional[Dict] = None) -> Dict:
        """Fast song analysis - only what we need for transitions."""
        import time, json
        log_path = '/Users/saksham/untitled folder 7/.cursor/debug.log'
        
        #region agent log
        analyze_start = time.time()
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"smart_mixer.py:160","message":"_analyze_song_fast start","data":{"y_len":len(y),"duration_sec":len(y)/self.sr},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Use sample of song for faster analysis
        # For Song B (incoming): analyze more of the song to find good transition-IN points anywhere
        # For Song A (outgoing): sample beginning and end to find good mix-out points
        duration = len(y) / self.sr
        if duration > 120:
            # For long songs, use strategic sampling:
            # - First 60s: for early mix-in points (intros, builds)
            # - Middle sections: for later mix-in points (drops, choruses)
            # - Last 60s: for mix-out points
            sample_length = 60 * self.sr
            y_sample_start = y[:sample_length]
            # Sample middle sections for Song B (incoming) to find later transition points
            mid_point = len(y) // 2
            y_sample_mid = y[mid_point:mid_point + sample_length] if mid_point + sample_length < len(y) else y[mid_point:]
            y_sample_end = y[-sample_length:]
            # Use start for most analysis (key, tempo) but structure analyzer will use full song
            y_sample = y_sample_start  # For key/tempo detection
        else:
            y_sample = y  # For short songs, analyze full song
        
        if existing_analysis:
            # Use existing analysis but enhance with new modules
            key = existing_analysis.get('harmony', {}).get('key', 'C')
            tempo = existing_analysis.get('tempo', {}).get('bpm', 120)
        else:
            # Quick key and tempo detection
            print("    Detecting key...")
            key_info = self.harmonic_analyzer.detect_key_camelot(y_sample)
            key = key_info['key']
            
            print("    Detecting tempo...")
            tempo, _ = librosa.beat.beat_track(y=y_sample, sr=self.sr, hop_length=self.hop_length)
            tempo = float(tempo)
        
        # Quick structure analysis (simplified) - skip if too slow
        print("    Analyzing structure...")
        
        #region agent log
        struct_start = time.time()
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"smart_mixer.py:177","message":"Starting structure analysis","data":{"sample_len":len(y_sample)},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        try:
            # For structure analysis: analyze MORE of the song for Song B to find good transition points
            # Analyze up to 60 seconds (or full song if shorter) instead of just 30 seconds
            # This helps find good transition-IN points later in Song B
            max_struct_sec = 60 if duration > 60 else duration  # Analyze up to 60s or full song
            struct_sample = y[:min(int(max_struct_sec * self.sr), len(y))]
            structure = self.structure_analyzer.analyze_structure(struct_sample)
            #region agent log
            struct_time = time.time() - struct_start
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"smart_mixer.py:184","message":"Structure analysis complete","data":{"time_sec":struct_time},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            
            # Scale structure times to full song if we used a sample
            if len(y) > len(struct_sample):
                scale_factor = len(y) / len(struct_sample)
                for phrase in structure.get('phrases', []):
                    phrase['start_sec'] *= scale_factor
                    phrase['end_sec'] *= scale_factor
                for section in structure.get('sections', []):
                    section['start_sec'] *= scale_factor
                    section['end_sec'] *= scale_factor
                # Convert numpy arrays to lists and scale
                mix_in_points = structure.get('best_mix_in_points', [])
                if isinstance(mix_in_points, np.ndarray):
                    mix_in_points = mix_in_points.tolist()
                structure['best_mix_in_points'] = [float(p) * scale_factor for p in mix_in_points]
                
                mix_out_points = structure.get('best_mix_out_points', [])
                if isinstance(mix_out_points, np.ndarray):
                    mix_out_points = mix_out_points.tolist()
                structure['best_mix_out_points'] = [float(p) * scale_factor for p in mix_out_points]
        except Exception as e:
            #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"smart_mixer.py:200","message":"Structure analysis failed","data":{"error":str(e)},"timestamp":int(time.time()*1000)}) + '\n')
            #endregion
            print(f"    Warning: Structure analysis failed: {e}")
            # Fallback structure
            duration = len(y) / self.sr
            structure = {
                'best_mix_out_points': [max(0, duration - 30)],
                'best_mix_in_points': [min(30, duration * 0.1)],
                'sections': []
            }
        
        # Quick energy estimate (sample only)
        energy_sample = y[:min(10 * self.sr, len(y))]  # First 10 seconds
        energy = np.mean(np.abs(energy_sample) ** 2)
        
        #region agent log
        analyze_time = time.time() - analyze_start
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"smart_mixer.py:214","message":"_analyze_song_fast complete","data":{"time_sec":analyze_time},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        return {
            'key': key,
            'tempo': tempo,
            'structure': structure,
            'energy': float(energy)
        }
    
    def _find_optimal_transition_points(self,
                                       y_a: np.ndarray,
                                       y_b: np.ndarray,
                                       analysis_a: Dict,
                                       analysis_b: Dict) -> TransitionPair:
        """
        Find optimal transition points using intelligent multi-candidate evaluation.
        Uses quality prediction to find the best transition points, while still
        favoring end-of-A and start-of-B as high-priority candidates.
        """
        import tempfile
        import os
        
        # Save temporary audio files for transition finder
        temp_dir = tempfile.mkdtemp()
        temp_a = os.path.join(temp_dir, 'temp_song_a.wav')
        temp_b = os.path.join(temp_dir, 'temp_song_b.wav')
        
        try:
            # Ensure stereo for soundfile
            if y_a.ndim == 1:
                y_a_stereo = np.column_stack([y_a, y_a])
            else:
                y_a_stereo = y_a
            if y_b.ndim == 1:
                y_b_stereo = np.column_stack([y_b, y_b])
            else:
                y_b_stereo = y_b
            
            sf.write(temp_a, y_a_stereo, self.sr)
            sf.write(temp_b, y_b_stereo, self.sr)
            
            # Use intelligent transition finder with quality prediction
            transition_pair = self.transition_finder.find_best_transition_pair_intelligent(
                temp_a,
                temp_b,
                song_a_analysis=analysis_a,
                song_b_analysis=analysis_b
            )
            
            return transition_pair
            
        finally:
            # Cleanup temporary files
            try:
                if os.path.exists(temp_a):
                    os.remove(temp_a)
                if os.path.exists(temp_b):
                    os.remove(temp_b)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except:
                pass  # Silently ignore cleanup errors
    
    def _get_section_at_time(self, time_sec: float, structure: Dict) -> str:
        """Get section type at a given time."""
        sections = structure.get('sections', [])
        for section in sections:
            if section['start_sec'] <= time_sec <= section['end_sec']:
                return section['type']
        return 'unknown'
    
    def _extract_segments(self,
                         y_a: np.ndarray,
                         y_b: np.ndarray,
                         point_a: float,
                         point_b: float,
                         duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """Extract transition segments.
        
        Args:
            point_a: Transition point in song A (where to start fading out)
            point_b: Transition point in song B (where to start fading in)
            duration: Transition duration in seconds
        
        Returns:
            seg_a: Segment from song A (end portion for fade out)
            seg_b: Segment from song B (beginning portion for fade in)
        """
        n_samples = int(duration * self.sr)
        
        # Clamp transition points to valid ranges
        point_a = max(0.0, min(point_a, len(y_a) / self.sr))
        point_b = max(0.0, min(point_b, len(y_b) / self.sr))
        
        # Extract from song A: point_a is where transition ENDS, so extract BACKWARDS
        # We want duration seconds BEFORE point_a (or up to point_a if near start)
        end_a = int(point_a * self.sr)
        start_a = max(0, end_a - n_samples)
        seg_a = y_a[start_a:end_a]
        
        #region agent log
        import json
        import time
        log_path = '/Users/saksham/untitled folder 7/.cursor/debug.log'
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"smart_mixer.py:590","message":"Extracting seg_a","data":{"point_a_sec":point_a,"start_a":start_a,"end_a":end_a,"n_samples":n_samples,"seg_a_len":len(seg_a),"extracted_duration_sec":len(seg_a)/self.sr},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Extract from song B: starting at point_b, extract duration seconds
        # Ensure point_b is valid (non-negative)
        start_b = max(0, int(point_b * self.sr))
        end_b = min(start_b + n_samples, len(y_b))
        seg_b = y_b[start_b:end_b]
        
        #region agent log
        if start_b < 0 or len(seg_b) == 0:
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"smart_mixer.py:610","message":"WARNING: Invalid seg_b extraction","data":{"point_b_sec":point_b,"start_b":start_b,"end_b":end_b,"seg_b_len":len(seg_b)},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"smart_mixer.py:525","message":"Extracting seg_b","data":{"point_b_sec":point_b,"start_b":start_b,"end_b":end_b,"n_samples":n_samples,"seg_b_len":len(seg_b)},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Ensure same length (trim to shortest)
        min_len = min(len(seg_a), len(seg_b))
        if len(seg_a) > min_len:
            seg_a = seg_a[:min_len]
        if len(seg_b) > min_len:
            seg_b = seg_b[:min_len]
        
        # Pad if shorter than requested duration
        if len(seg_a) < n_samples:
            pad = n_samples - len(seg_a)
            if seg_a.ndim == 1:
                seg_a = np.pad(seg_a, (0, pad), mode='constant')
            else:
                seg_a = np.pad(seg_a, ((0, pad), (0, 0)), mode='constant')
        
        if len(seg_b) < n_samples:
            pad = n_samples - len(seg_b)
            if seg_b.ndim == 1:
                seg_b = np.pad(seg_b, (0, pad), mode='constant')
            else:
                seg_b = np.pad(seg_b, ((0, pad), (0, 0)), mode='constant')
        
        # Ensure stereo
        if seg_a.ndim == 1:
            seg_a = np.column_stack([seg_a, seg_a])
        if seg_b.ndim == 1:
            seg_b = np.column_stack([seg_b, seg_b])
        
        #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"smart_mixer.py:554","message":"Segments prepared","data":{"final_seg_a_len":len(seg_a),"final_seg_b_len":len(seg_b),"seg_a_shape":str(seg_a.shape),"seg_b_shape":str(seg_b.shape)},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        return seg_a, seg_b
    
    def _apply_bass_swap(self, seg_a: np.ndarray, seg_b: np.ndarray, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply bass swap technique."""
        automation = self.dynamic_processor.create_bass_swap_automation(duration)
        return self.dynamic_processor.apply_bass_swap(seg_a, seg_b, automation)
    
    def _apply_preventive_eq(self, seg_a: np.ndarray, seg_b: np.ndarray, 
                            vol_a: np.ndarray, vol_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply preventive EQ to prevent frequency clashes and instrument conflicts.
        
        This prevents drums/instruments from fighting for space by:
        1. Gradually reducing bass in outgoing track
        2. Gradually reducing conflicting mid frequencies  
        3. Keeping transitions smooth
        """
        from scipy import signal
        
        sr = self.sr
        n_samples = len(seg_a)
        
        # Create time-varying EQ curves
        t = np.linspace(0, 1, n_samples)
        
        # For outgoing track: reduce bass as it fades out (prevent low-end conflict)
        # For incoming track: keep bass low initially, let it rise gradually
        bass_cut_a = 0.4 + 0.6 * t  # Gradually cut more bass (0.4 -> 1.0)
        bass_cut_b = 0.8 - 0.5 * t  # Start with cut, then restore (0.8 -> 0.3)
        
        seg_a_eq = seg_a.copy()
        seg_b_eq = seg_b.copy()
        
        # Process in chunks to apply time-varying EQ
        chunk_size = int(0.5 * sr)  # 0.5 second chunks
        nyq = sr / 2
        low_cutoff = 250 / nyq  # Bass cutoff at 250Hz
        
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            
            # Get average EQ value for this chunk
            bass_cut_a_chunk = np.mean(bass_cut_a[i:end_idx])
            bass_cut_b_chunk = np.mean(bass_cut_b[i:end_idx])
            
            # Apply bass reduction using high-pass filter
            # Only apply if significant reduction needed
            if bass_cut_a_chunk > 0.5:  # Need at least 50% reduction
                for ch in range(seg_a_eq.shape[1] if seg_a_eq.ndim == 2 else 1):
                    if seg_a_eq.ndim == 2:
                        chunk = seg_a_eq[i:end_idx, ch]
                        # Gentle high-pass to reduce bass
                        b, a = signal.butter(2, low_cutoff * bass_cut_a_chunk, btype='high')
                        seg_a_eq[i:end_idx, ch] = signal.filtfilt(b, a, chunk) * (0.85 + 0.15 * (1 - bass_cut_a_chunk))
                    else:
                        chunk = seg_a_eq[i:end_idx]
                        b, a = signal.butter(2, low_cutoff * bass_cut_a_chunk, btype='high')
                        seg_a_eq[i:end_idx] = signal.filtfilt(b, a, chunk) * (0.85 + 0.15 * (1 - bass_cut_a_chunk))
            
            if bass_cut_b_chunk > 0.4:
                for ch in range(seg_b_eq.shape[1] if seg_b_eq.ndim == 2 else 1):
                    if seg_b_eq.ndim == 2:
                        chunk = seg_b_eq[i:end_idx, ch]
                        b, a = signal.butter(2, low_cutoff * bass_cut_b_chunk, btype='high')
                        seg_b_eq[i:end_idx, ch] = signal.filtfilt(b, a, chunk) * (0.85 + 0.15 * (1 - bass_cut_b_chunk))
                    else:
                        chunk = seg_b_eq[i:end_idx]
                        b, a = signal.butter(2, low_cutoff * bass_cut_b_chunk, btype='high')
                        seg_b_eq[i:end_idx] = signal.filtfilt(b, a, chunk) * (0.85 + 0.15 * (1 - bass_cut_b_chunk))
        
        return seg_a_eq, seg_b_eq
    
    def _detect_vocal_start_time(self, vocal_stem: Optional[np.ndarray], total_samples: int) -> float:
        """
        Detect when vocals actually start in Song B.
        
        Args:
            vocal_stem: Separated vocal stem from Song B (can be None)
            total_samples: Total number of samples in transition segment
        
        Returns:
            Ratio (0-1) indicating when vocals start (0.5 = 50% through transition)
        """
        if vocal_stem is None or len(vocal_stem) == 0:
            return 0.5  # Default: 50% through transition
        
        # Convert to mono if stereo
        if vocal_stem.ndim > 1:
            vocal_stem = np.mean(vocal_stem, axis=1)
        
        # Use energy-based detection
        frame_size = int(self.sr * 0.5)  # 0.5 second frames
        n_frames = len(vocal_stem) // frame_size
        if n_frames == 0:
            return 0.5
        
        frame_energies = []
        for i in range(n_frames):
            start = i * frame_size
            end = min(start + frame_size, len(vocal_stem))
            energy = np.mean(vocal_stem[start:end] ** 2)
            frame_energies.append(energy)
        
        if len(frame_energies) == 0:
            return 0.5
        
        # Find first frame with significant vocal energy
        max_energy = np.max(frame_energies)
        threshold = max_energy * 0.3  # 30% of max energy
        
        for i, energy in enumerate(frame_energies):
            if energy > threshold:
                # Convert frame index to ratio
                frame_ratio = (i * frame_size) / total_samples
                return float(np.clip(frame_ratio, 0.0, 1.0))
        
        # If no significant energy found, assume vocals start halfway
        return 0.5
    
    def create_superhuman_mix(self,
                              song_a_path: str,
                              song_b_path: str,
                              transition_duration: Optional[float] = None,
                              song_a_analysis: Optional[Dict] = None,
                              song_b_analysis: Optional[Dict] = None,
                              creativity_level: float = 0.6,
                              optimize_quality: bool = True,
                              force_stem_orchestration: bool = False,
                              conversation_type_override: Optional[str] = None,
                              return_metadata: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Create a SUPERHUMAN-quality mix using all advanced AI/DSP capabilities.
        
        This method uses features that exceed what human DJs can achieve:
        - Micro-timing: Sub-millisecond groove and transient alignment
        - Spectral Intelligence: Surgical frequency slot negotiation
        - Hybrid Techniques: Creative blending of multiple techniques
        - Stem Orchestration: Musical stem conversations
        - Monte Carlo Optimization: Quality simulation and prediction
        
        Args:
            song_a_path: Path to outgoing song
            song_b_path: Path to incoming song
            transition_duration: Optional transition duration in seconds
            song_a_analysis: Optional pre-computed analysis for song A
            song_b_analysis: Optional pre-computed analysis for song B
            creativity_level: 0.0 (conservative) to 1.0 (experimental)
            optimize_quality: Whether to run Monte Carlo optimization
        
        Returns:
            Mixed audio with context (stereo, 44100Hz)
        """
        if not self.superhuman_enabled or self.superhuman_engine is None:
            print("  ⚠ Superhuman engine not available, falling back to standard mix")
            ai_data = None
            try:
                config_root = Path(__file__).resolve().parent.parent
                with open(config_root / "youtube_transition.json") as f:
                    trans = json.load(f)
                    if "curves" in trans:
                        ai_data = trans
            except Exception:
                pass
            return self.create_smooth_mix(
                song_a_path,
                song_b_path,
                transition_duration,
                song_a_analysis,
                song_b_analysis,
                ai_transition_data=ai_data,
                return_metadata=return_metadata,
            )
        
        print("\n" + "="*60)
        print("🚀 SUPERHUMAN MIXING ENGINE")
        print("="*60)
        
        import time
        start_time = time.time()
        try:
            # Configure engine with creativity level
            self.superhuman_engine.configure(creativity_level=creativity_level)
            
            # Load audio
            print("\n[1/4] Loading audio...")
            y_a, _ = librosa.load(song_a_path, sr=self.sr)
            y_b, _ = librosa.load(song_b_path, sr=self.sr)
            
            # Analyze songs
            print("[2/4] Analyzing songs...")
            analysis_a = self._analyze_song_fast(y_a, song_a_analysis)
            analysis_b = self._analyze_song_fast(y_b, song_b_analysis)
            
            tempo_a = analysis_a.get('tempo', 120)
            tempo_b = analysis_b.get('tempo', 120)
            key_a = analysis_a.get('key', 'C')
            key_b = analysis_b.get('key', 'C')
            
            # Find optimal transition points
            print("[3/4] Finding optimal transition points...")
            transition_pair = self._find_optimal_transition_points(y_a, y_b, analysis_a, analysis_b)
            
            point_a = transition_pair.song_a_point.time_sec
            point_b = transition_pair.song_b_point.time_sec
            
            # Determine transition duration
            if transition_duration is None:
                bars = 16 if tempo_a > 130 else 24
                transition_duration = (bars * 4 * 60) / tempo_a
            
            print(f"  → Transition: {transition_duration:.1f}s")
            print(f"  → Song A exit: {point_a:.1f}s")
            print(f"  → Song B entry: {point_b:.1f}s")
            
            seg_a, seg_b = self._extract_segments(y_a, y_b, point_a, point_b, transition_duration)
            
            stems_a = None
            stems_b = None
            if self.stem_separation_enabled and self.stem_separator is not None:
                print("  Separating stems for advanced processing...")
                try:
                    stems_a = self.stem_separator.separate_segment(seg_a, self.sr)
                    stems_b = self.stem_separator.separate_segment(seg_b, self.sr)
                except Exception as e:
                    print(f"  ⚠ Stem separation failed: {e}")
            
            print("[4/4] 🎛️ Running Superhuman Mix...")
            result = self.superhuman_engine.create_superhuman_mix(
                seg_a, seg_b,
                tempo_a, tempo_b,
                key_a, key_b,
                stems_a, stems_b,
                force_stem_orchestration=force_stem_orchestration,
                conversation_type_override=conversation_type_override
            )
            
            mixed = result['mixed']
            analysis = result['analysis']
            quality = result['quality']
            
            print("\n" + "-"*40)
            print("📊 SUPERHUMAN MIX RESULTS:")
            print(f"  Quality Score: {quality['overall_score']:.2f}")
            print(f"  Confidence: {quality.get('confidence', 0):.2f}")
            print(f"  Recommendation: {quality.get('recommendation', 'unknown')}")
            
            if analysis.get('mix_method') == 'stem_orchestration':
                print(f"  Method: Stem orchestration (One Kiss beat → Hell of a Life handoff)")
            elif 'technique' in analysis:
                tech = analysis['technique']
                if tech.get('type') == 'hybrid':
                    print(f"  Technique: {tech.get('name')} (hybrid)")
                    print(f"    Components: {', '.join(tech.get('techniques', []))}")
                else:
                    print(f"  Technique: {tech.get('name')}")
            
            if 'micro_timing' in analysis:
                mt = analysis['micro_timing']
                print(f"  Groove Compatibility: {mt.get('groove_compatibility', 0):.2f}")
                print(f"  Rhythmic Compatibility: {mt.get('rhythmic_compatibility', 0):.2f}")
            
            if 'spectral' in analysis:
                sp = analysis['spectral']
                print(f"  Frequency Conflict: {sp.get('overall_conflict', 0):.2f}")
                print(f"  Resonance Strength: {sp.get('resonance_strength', 0):.2f}")
            
            print("-"*40)
            
            context_before = int(10 * self.sr)
            context_after = int(10 * self.sr)
            
            seg_a_start = int(point_a * self.sr) - len(seg_a)
            ctx_a_start = max(0, seg_a_start - context_before)
            ctx_a_end = seg_a_start
            ctx_a = y_a[ctx_a_start:ctx_a_end]
            
            seg_b_start = int(point_b * self.sr)
            seg_b_end = seg_b_start + len(seg_b)
            ctx_b_start = seg_b_end
            ctx_b_end = min(len(y_b), ctx_b_start + context_after)
            ctx_b = y_b[ctx_b_start:ctx_b_end]
            
            if ctx_a.ndim == 1:
                ctx_a = np.column_stack([ctx_a, ctx_a])
            if ctx_b.ndim == 1:
                ctx_b = np.column_stack([ctx_b, ctx_b])
            if mixed.ndim == 1:
                mixed = np.column_stack([mixed, mixed])
            
            # ========== BOUNDARY CROSSFADES (eliminate clicks at splice points) ==========
            xfade_samples = min(int(0.05 * self.sr), len(ctx_a) // 2, len(mixed) // 2, len(ctx_b) // 2)
            
            if xfade_samples > 10 and len(ctx_a) > 0 and len(mixed) > 0 and len(ctx_b) > 0:
                fade_out = np.linspace(1.0, 0.0, xfade_samples)[:, np.newaxis]
                fade_in = np.linspace(0.0, 1.0, xfade_samples)[:, np.newaxis]
                
                overlap_1 = ctx_a[-xfade_samples:] * fade_out + mixed[:xfade_samples] * fade_in
                overlap_2 = mixed[-xfade_samples:] * fade_out + ctx_b[:xfade_samples] * fade_in
                
                final = np.concatenate([
                    ctx_a[:-xfade_samples],
                    overlap_1,
                    mixed[xfade_samples:-xfade_samples] if len(mixed) > 2 * xfade_samples else np.empty((0, mixed.shape[1]) if mixed.ndim == 2 else (0,)),
                    overlap_2,
                    ctx_b[xfade_samples:]
                ], axis=0)
            else:
                final = np.concatenate([ctx_a, mixed, ctx_b], axis=0)

            technique_meta = analysis.get('technique', {}) if isinstance(analysis, dict) else {}
            if technique_meta.get('type') == 'hybrid':
                technique_name = technique_meta.get('name') or '+'.join(technique_meta.get('techniques', []))
            else:
                technique_name = technique_meta.get('name')

            # Metadata for where to resume Song B (relative to y_b)
            mix_metadata: Dict = {
                # Start of transition content in Song A snippet coordinates.
                "a_transition_start_samples": int(ctx_a_start),
                "a_transition_start_sec": float(ctx_a_start / self.sr),
                # End of consumed Song B content in Song B snippet coordinates.
                "b_resume_offset_samples": int(ctx_b_end),
                "b_resume_offset_sec": float(ctx_b_end / self.sr),
                # Transition diagnostics
                "mix_method": analysis.get('mix_method', 'unknown'),
                "technique_name": technique_name,
                "vocal_overlap_risk_estimate": analysis.get('vocal_overlap_risk_estimate'),
                "vocal_guard_note": technique_meta.get('vocal_guard_note'),
            }
            
            total_time = time.time() - start_time
            print(f"\n✅ Superhuman mix complete in {total_time:.1f}s")
            print(f"   Output duration: {len(final)/self.sr:.1f}s")
            
            if return_metadata:
                return final, mix_metadata
            return final
        except Exception as e:
            print(f"  ⚠ Superhuman path failed: {e}")
            print("  Falling back to standard smooth mix with AI curves...")
            ai_data = None
            try:
                config_root = Path(__file__).resolve().parent.parent
                with open(config_root / "youtube_transition.json") as f:
                    trans = json.load(f)
                    if "curves" in trans:
                        ai_data = trans
            except Exception:
                pass
            return self.create_smooth_mix(
                song_a_path,
                song_b_path,
                transition_duration,
                song_a_analysis,
                song_b_analysis,
                ai_transition_data=ai_data,
                return_metadata=return_metadata,
            )        
