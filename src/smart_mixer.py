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
from typing import Dict, Optional, Tuple

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

# Debug logging helper
DEBUG_LOG_PATH = '/Users/saksham/untitled folder 7/.cursor/debug.log'
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
        log_path = '/Users/saksham/untitled folder 7/.cursor/debug.log'
        
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
                         ai_transition_data: Optional[Dict] = None) -> np.ndarray:
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
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"smart_mixer.py:230","message":"Segments extracted","data":{"seg_a_len":len(seg_a),"seg_b_len":len(seg_b),"seg_a_shape":str(seg_a.shape),"seg_b_shape":str(seg_b.shape),"seg_a_rms":seg_a_rms,"seg_b_rms":seg_b_rms,"seg_a_max":seg_a_max,"seg_b_max":seg_b_max,"seg_a_zeros":seg_a_zeros,"seg_b_zeros":seg_b_zeros},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
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
        
        # Create smooth, gradual crossfade curves
        print("[7/8] Creating smooth crossfade curves...")
        
        # Use simple, smooth equal-power crossfade with extra smoothing
        vol_a, vol_b = self.crossfade_engine.create_gradual_crossfade(
            len(seg_a), overlap_ratio=0.75
        )
        
        #region agent log
        vol_a_sample_start = vol_a[:min(100, len(vol_a))].tolist() if len(vol_a) > 0 else []
        vol_a_sample_end = vol_a[-min(100, len(vol_a)):].tolist() if len(vol_a) > 0 else []
        vol_b_sample_start = vol_b[:min(100, len(vol_b))].tolist() if len(vol_b) > 0 else []
        vol_b_sample_end = vol_b[-min(100, len(vol_b)):].tolist() if len(vol_b) > 0 else []
        debug_log("Volume curves created", {
            "vol_a_len": len(vol_a), "vol_b_len": len(vol_b),
            "vol_a_min": float(np.min(vol_a)), "vol_a_max": float(np.max(vol_a)),
            "vol_b_min": float(np.min(vol_b)), "vol_b_max": float(np.max(vol_b)),
            "vol_a_start_sample": vol_a_sample_start[:10], "vol_a_end_sample": vol_a_sample_end[-10:],
            "vol_b_start_sample": vol_b_sample_start[:10], "vol_b_end_sample": vol_b_sample_end[-10:]
        }, "B", "smart_mixer.py:277")
        #endregion
        
        # Simple bass management: apply gentle high-pass to outgoing track as it fades
        # This prevents bass conflicts without complex processing
        from scipy import signal
        sr = self.sr
        n_samples = len(seg_a)
        
        # Create time-varying bass reduction for outgoing track
        t = np.linspace(0, 1, n_samples)
        # Gradually reduce bass in outgoing track (prevent low-end conflict)
        bass_reduction = t ** 2  # Starts at 0, ends at 1
        
        # Apply gentle high-pass filter that increases over time
        seg_a_final = seg_a.copy()
        if seg_a_final.ndim == 2:
            for ch in range(seg_a_final.shape[1]):
                # Process in chunks for efficiency
                chunk_size = int(2.0 * sr)  # 2 second chunks
                for i in range(0, n_samples, chunk_size):
                    end_idx = min(i + chunk_size, n_samples)
                    chunk = seg_a_final[i:end_idx, ch]
                    
                    # Apply bass reduction based on fade progress
                    avg_reduction = np.mean(bass_reduction[i:end_idx])
                    if avg_reduction > 0.3:  # Only filter if significant reduction needed
                        nyq = sr / 2
                        cutoff = (200 + 100 * avg_reduction) / nyq  # 200Hz -> 300Hz
                        b, a = signal.butter(2, cutoff, btype='high')
                        seg_a_final[i:end_idx, ch] = signal.filtfilt(b, a, chunk)
        else:
            # Mono handling
            chunk_size = int(2.0 * sr)
            for i in range(0, n_samples, chunk_size):
                end_idx = min(i + chunk_size, n_samples)
                chunk = seg_a_final[i:end_idx]
                avg_reduction = np.mean(bass_reduction[i:end_idx])
                if avg_reduction > 0.3:
                    nyq = sr / 2
                    cutoff = (200 + 100 * avg_reduction) / nyq
                    b, a = signal.butter(2, cutoff, btype='high')
                    seg_a_final[i:end_idx] = signal.filtfilt(b, a, chunk)
        
        seg_b_final = seg_b  # Incoming track doesn't need filtering (bass will naturally fade in)
        
        # Mix
        print("[8/8] Mixing with smooth crossfade...")
        
        #region agent log
        mix_start = time.time()
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"smart_mixer.py:268","message":"Before crossfade","data":{"vol_a_len":len(vol_a),"vol_b_len":len(vol_b),"seg_a_rms":float(np.sqrt(np.mean(seg_a_final**2))),"seg_b_rms":float(np.sqrt(np.mean(seg_b_final**2)))},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        #region agent log
        # Log BEFORE crossfade to test hypothesis D
        seg_a_final_rms = float(np.sqrt(np.mean(seg_a_final**2)))
        seg_b_final_rms = float(np.sqrt(np.mean(seg_b_final**2)))
        seg_a_final_max = float(np.max(np.abs(seg_a_final)))
        seg_b_final_max = float(np.max(np.abs(seg_b_final)))
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"smart_mixer.py:330","message":"Before apply_crossfade","data":{"seg_a_final_rms":seg_a_final_rms,"seg_b_final_rms":seg_b_final_rms,"seg_a_final_max":seg_a_final_max,"seg_b_final_max":seg_b_final_max,"vol_a_len":len(vol_a),"vol_b_len":len(vol_b)},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        mixed = self.crossfade_engine.apply_crossfade(seg_a_final, seg_b_final, vol_a, vol_b)
        
        #region agent log
        mix_time = time.time() - mix_start
        seg_a_rms = float(np.sqrt(np.mean(seg_a**2)))
        seg_b_rms = float(np.sqrt(np.mean(seg_b**2)))
        mixed_rms = float(np.sqrt(np.mean(mixed**2)))
        mixed_max = float(np.max(np.abs(mixed)))
        vol_a_min = float(np.min(vol_a))
        vol_a_max = float(np.max(vol_a))
        vol_b_min = float(np.min(vol_b))
        vol_b_max = float(np.max(vol_b))
        # Test hypothesis D: Check if crossfade properly reduces volume
        mixed_sample_start = mixed[:min(1000, len(mixed))].tolist() if len(mixed) > 0 else []
        mixed_sample_mid = mixed[len(mixed)//2:len(mixed)//2+min(1000, len(mixed))].tolist() if len(mixed) > 0 else []
        mixed_sample_end = mixed[-min(1000, len(mixed)):].tolist() if len(mixed) > 0 else []
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"smart_mixer.py:343","message":"After apply_crossfade","data":{"mixed_len":len(mixed),"mixed_shape":str(mixed.shape),"time_sec":mix_time,"seg_a_rms":seg_a_rms,"seg_b_rms":seg_b_rms,"mixed_rms":mixed_rms,"mixed_max":mixed_max,"vol_a_range":f"{vol_a_min:.3f}-{vol_a_max:.3f}","vol_b_range":f"{vol_b_min:.3f}-{vol_b_max:.3f}","mixed_sample_start_max":float(np.max(np.abs(mixed[:min(1000, len(mixed))]))) if len(mixed) > 0 else 0,"mixed_sample_mid_max":float(np.max(np.abs(mixed[len(mixed)//2:len(mixed)//2+min(1000, len(mixed))]))) if len(mixed) > 0 else 0,"mixed_sample_end_max":float(np.max(np.abs(mixed[-min(1000, len(mixed)):]))) if len(mixed) > 0 else 0},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        # Skip quality assessment for speed (optional, can add back later)
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
        
        final = np.concatenate([ctx_a, mixed, ctx_b], axis=0)
        
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
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"ALL","location":"smart_mixer.py:407","message":"Concatenation complete - final analysis","data":{"final_len":len(final),"final_duration_sec":len(final)/self.sr,"time_sec":final_time,"final_rms":final_rms,"final_max":final_max,"boundary_1_rms":boundary_1_rms,"boundary_2_rms":boundary_2_rms,"boundary_1_max":float(np.max(np.abs(final[transition_boundary_1_start:transition_boundary_1_end]))) if transition_boundary_1_end > transition_boundary_1_start else 0,"boundary_2_max":float(np.max(np.abs(final[transition_boundary_2_start:transition_boundary_2_end]))) if transition_boundary_2_end > transition_boundary_2_start else 0},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
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
        
        # Use sample of song for faster analysis (first 60 seconds OR last 60 seconds for mix-out points)
        # For mix-out, we need the END of the song
        duration = len(y) / self.sr
        if duration > 120:
            # Sample both beginning (for mix-in) and end (for mix-out)
            sample_length = 60 * self.sr
            y_sample_start = y[:sample_length]
            y_sample_end = y[-sample_length:]
            # Combine for analysis but remember which is which
            y_sample = y_sample_start  # Use start for most analysis
        else:
            y_sample = y
        
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
            # Limit structure analysis to 30 seconds max to prevent hanging
            struct_sample = y_sample[:min(30 * self.sr, len(y_sample))]
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
        """Find optimal transition points - STRICTLY end of song A and start of song B."""
        from src.smart_transition_finder import TransitionPoint
        
        # STRICT: Use end of song A and start of song B
        duration_a = len(y_a) / self.sr
        duration_b = len(y_b) / self.sr
        
        # Song A: Use the very end (last 30 seconds max, but prefer as close to end as possible)
        # For transition, we want the transition to END at the end of song A
        # So point_a should be close to duration_a
        point_a = duration_a  # Exact end of song A
        
        # Song B: Use the very start (first few seconds, but aligned to beat)
        # Find first beat after a short skip (to avoid silence/intro noise)
        import librosa
        tempo_b, beat_frames = librosa.beat.beat_track(y=y_b[:min(30 * self.sr, len(y_b))], 
                                                       sr=self.sr, 
                                                       hop_length=self.hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr, hop_length=self.hop_length)
        
        # Use first beat that's after 2 seconds (skip intro silence)
        point_b = 0.0
        for bt in beat_times:
            if bt >= 2.0:  # Start after 2 seconds to skip intro silence
                point_b = bt
                break
        
        # If no beat found after 2 seconds, use 2 seconds
        if point_b == 0.0:
            point_b = 2.0
        
        # Ensure point_a is valid (within song bounds)
        # For transition, we extract BACKWARDS from point_a, so point_a should be at or near the end
        point_a = min(point_a, duration_a)
        
        # Align point_a to nearest beat at the end of song A
        tempo_a, beat_frames_a = librosa.beat.beat_track(y=y_a[max(0, int((duration_a - 30) * self.sr)):], 
                                                         sr=self.sr, 
                                                         hop_length=self.hop_length)
        beat_times_a = librosa.frames_to_time(beat_frames_a, sr=self.sr, hop_length=self.hop_length)
        # Adjust beat times to absolute time in song
        beat_times_a = beat_times_a + max(0, duration_a - 30)
        
        # Find last beat before or at point_a
        if len(beat_times_a) > 0:
            # Use the last beat
            aligned_point_a = beat_times_a[-1]
            # But ensure we don't go beyond song end
            point_a = min(aligned_point_a, duration_a)
        else:
            # Fallback: use last 5 seconds
            point_a = max(0, duration_a - 5)
        
        # Calculate harmonic compatibility for scoring
        harmonic = self.harmonic_analyzer.score_transition_harmonics(
            analysis_a['key'], analysis_b['key'],
            analysis_a['tempo'], analysis_b['tempo']
        )
        
        # Create transition pair
        best_pair = TransitionPair(
            song_a_point=TransitionPoint(
                time_sec=point_a,
                beat_aligned=True,
                energy=0.5,
                energy_trend='falling',
                structural_label='outro',
                score=1.0,  # High score since we're using optimal end/start
                beat_position=0
            ),
            song_b_point=TransitionPoint(
                time_sec=point_b,
                beat_aligned=True,
                energy=0.5,
                energy_trend='rising',
                structural_label='intro',
                score=1.0,
                beat_position=0
            ),
            compatibility_score=harmonic['overall_score'],
            tempo_match=abs(analysis_a['tempo'] - analysis_b['tempo']) < 5,
            key_match=harmonic['key_compatibility']['compatible'],
            beat_aligned=True
        )
        
        return best_pair
    
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
