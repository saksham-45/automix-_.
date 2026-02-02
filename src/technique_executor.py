"""
Technique Executor Module

Executes specific DJ transition techniques based on selection.
Handles all transition techniques: long_blend, quick_cut, bass_swap, filter_sweep,
echo_out, drop_mix, staggered_stem_mix, partial_stem_separation, vocal_layering.
"""
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import signal
import librosa


class TechniqueExecutor:
    """
    Executes DJ transition techniques based on selection.
    Each technique has its own implementation method.
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
    
    def execute(self,
                technique_name: str,
                seg_a: np.ndarray,
                seg_b: np.ndarray,
                params: Dict,
                seg_a_stems: Optional[Dict] = None,
                seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """
        Route to appropriate technique execution method.
        
        Args:
            technique_name: Name of technique to execute
            seg_a: Audio segment from outgoing song
            seg_b: Audio segment from incoming song
            params: Technique-specific parameters
            seg_a_stems: Optional separated stems for song A
            seg_b_stems: Optional separated stems for song B
        
        Returns:
            Mixed audio array
        """
        # Route to technique-specific method
        method_map = {
            'long_blend': self._execute_long_blend,
            'quick_cut': self._execute_quick_cut,
            'bass_swap': self._execute_bass_swap,
            'filter_sweep': self._execute_filter_sweep,
            'echo_out': self._execute_echo_out,
            'drop_mix': self._execute_drop_mix,
            'staggered_stem_mix': self._execute_staggered_stem_mix,
            'partial_stem_separation': self._execute_partial_stem_separation,
            'vocal_layering': self._execute_vocal_layering,
            'phrase_match': self._execute_phrase_match,
            'backspin': self._execute_backspin,
            'double_drop': self._execute_double_drop,
            'acapella_overlay': self._execute_acapella_overlay,
            'modulation': self._execute_modulation,
            'energy_build': self._execute_energy_build,
            'loop_transition': self._execute_loop_transition,
            'breakdown_to_build': self._execute_breakdown_to_build,
            'drop_on_the_one': self._execute_drop_on_the_one,
            'back_and_forth': self._execute_back_and_forth,
            'drum_roll': self._execute_drum_roll,
            'thematic_handoff': self._execute_thematic_handoff
        }
        
        if technique_name in method_map:
            return method_map[technique_name](seg_a, seg_b, params, seg_a_stems, seg_b_stems)
        else:
            # Fallback to long_blend
            print(f"  ⚠ Unknown technique '{technique_name}', using long_blend")
            return self._execute_long_blend(seg_a, seg_b, params, seg_a_stems, seg_b_stems)
    
    def _execute_long_blend(self,
                           seg_a: np.ndarray,
                           seg_b: np.ndarray,
                           params: Dict,
                           seg_a_stems: Optional[Dict] = None,
                           seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """
        Execute long blend: smooth, gradual equal-power crossfade.
        """
        from src.crossfade_engine import CrossfadeEngine
        crossfade_engine = CrossfadeEngine(sr=self.sr)
        
        n_samples = min(len(seg_a), len(seg_b))
        vol_a, vol_b = crossfade_engine.create_equal_power_crossfade(
            n_samples, curve_shape=params.get('curve_shape', 'smooth')
        )
        
        # Ensure stereo
        if seg_a.ndim == 1:
            seg_a = np.column_stack([seg_a, seg_a])
        if seg_b.ndim == 1:
            seg_b = np.column_stack([seg_b, seg_b])
        
        # Apply volumes
        if vol_a.ndim == 1:
            vol_a = vol_a[:, np.newaxis]
        if vol_b.ndim == 1:
            vol_b = vol_b[:, np.newaxis]
        
        mixed = seg_a[:n_samples] * vol_a[:n_samples] + seg_b[:n_samples] * vol_b[:n_samples]
        return mixed
    
    def _execute_quick_cut(self,
                          seg_a: np.ndarray,
                          seg_b: np.ndarray,
                          params: Dict,
                          seg_a_stems: Optional[Dict] = None,
                          seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """
        Execute quick cut: fast cut at beat boundary with minimal overlap.
        """
        n_samples = min(len(seg_a), len(seg_b))
        cut_point = params.get('cut_point', 'downbeat')
        fade_out_ms = params.get('fade_out_ms', 100)
        fade_in_ms = params.get('fade_in_ms', 100)
        
        fade_out_samples = int(fade_out_ms * self.sr / 1000)
        fade_in_samples = int(fade_in_ms * self.sr / 1000)
        
        # Create quick fade curves
        fade_out = np.ones(n_samples)
        if fade_out_samples > 0:
            fade_out[-fade_out_samples:] = np.linspace(1.0, 0.0, fade_out_samples)
        
        fade_in = np.zeros(n_samples)
        if fade_in_samples > 0:
            fade_in[:fade_in_samples] = np.linspace(0.0, 1.0, fade_in_samples)
        
        # Ensure stereo
        if seg_a.ndim == 1:
            seg_a = np.column_stack([seg_a, seg_a])
        if seg_b.ndim == 1:
            seg_b = np.column_stack([seg_b, seg_b])
        
        if fade_out.ndim == 1:
            fade_out = fade_out[:, np.newaxis]
        if fade_in.ndim == 1:
            fade_in = fade_in[:, np.newaxis]
        
        mixed = seg_a[:n_samples] * fade_out[:n_samples] + seg_b[:n_samples] * fade_in[:n_samples]
        return mixed
    
    def _execute_bass_swap(self,
                          seg_a: np.ndarray,
                          seg_b: np.ndarray,
                          params: Dict,
                          seg_a_stems: Optional[Dict] = None,
                          seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """
        Execute bass swap: swap bass frequencies at midpoint.
        """
        from src.crossfade_engine import CrossfadeEngine
        crossfade_engine = CrossfadeEngine(sr=self.sr)
        
        n_samples = min(len(seg_a), len(seg_b))
        swap_point_ratio = params.get('swap_point_ratio', 0.5)
        bass_cut_db = params.get('bass_cut_db', -12)
        bass_boost_db = params.get('bass_boost_db', 6)
        
        swap_point = int(n_samples * swap_point_ratio)
        
        # Create bass swap curves
        bass_a_fade = np.ones(n_samples)
        bass_a_fade[swap_point:] = 10 ** (bass_cut_db / 20)  # Cut bass after swap
        
        bass_b_fade = np.zeros(n_samples)
        bass_b_fade[:swap_point] = 10 ** (bass_cut_db / 20)  # Keep bass low before swap
        bass_b_fade[swap_point:] = 10 ** (bass_boost_db / 20)  # Boost bass after swap
        
        # Regular crossfade for mid/high frequencies
        vol_a, vol_b = crossfade_engine.create_equal_power_crossfade(n_samples, 'smooth')
        
        # Apply bass swap using high-pass filters
        seg_a_swapped = seg_a.copy()
        seg_b_swapped = seg_b.copy()
        
        # Ensure stereo
        if seg_a_swapped.ndim == 1:
            seg_a_swapped = np.column_stack([seg_a_swapped, seg_a_swapped])
        if seg_b_swapped.ndim == 1:
            seg_b_swapped = np.column_stack([seg_b_swapped, seg_b_swapped])
        
        # Apply bass swap in chunks
        chunk_size = int(1.0 * self.sr)  # 1 second chunks
        nyq = self.sr / 2
        low_cutoff = 250 / nyq  # 250 Hz cutoff for bass
        
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            
            # Get average bass gain for this chunk
            avg_bass_a = np.mean(bass_a_fade[i:end_idx])
            avg_bass_b = np.mean(bass_b_fade[i:end_idx])
            
            for ch in range(seg_a_swapped.shape[1]):
                chunk_a = seg_a_swapped[i:end_idx, ch]
                chunk_b = seg_b_swapped[i:end_idx, ch]
                
                # High-pass filter to isolate bass
                if avg_bass_a < 1.0:
                    # Reduce bass on seg_a
                    b, a = signal.butter(2, low_cutoff, btype='high')
                    bass_content = signal.filtfilt(b, a, chunk_a)
                    seg_a_swapped[i:end_idx, ch] = chunk_a * avg_bass_a + bass_content * (1 - avg_bass_a)
                
                if avg_bass_b > 1.0:
                    # Boost bass on seg_b
                    b, a = signal.butter(2, low_cutoff, btype='high')
                    bass_content = signal.filtfilt(b, a, chunk_b)
                    seg_b_swapped[i:end_idx, ch] = chunk_b + bass_content * (avg_bass_b - 1.0)
        
        # Apply regular crossfade
        if vol_a.ndim == 1:
            vol_a = vol_a[:, np.newaxis]
        if vol_b.ndim == 1:
            vol_b = vol_b[:, np.newaxis]
        
        mixed = seg_a_swapped[:n_samples] * vol_a[:n_samples] + seg_b_swapped[:n_samples] * vol_b[:n_samples]
        return mixed
    
    def _execute_filter_sweep(self,
                             seg_a: np.ndarray,
                             seg_b: np.ndarray,
                             params: Dict,
                             seg_a_stems: Optional[Dict] = None,
                             seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """
        Execute filter sweep: high/low-pass filter automation during transition.
        """
        from src.crossfade_engine import CrossfadeEngine
        crossfade_engine = CrossfadeEngine(sr=self.sr)
        
        n_samples = min(len(seg_a), len(seg_b))
        filter_type = params.get('filter_type', 'high_pass')
        filter_start_hz = params.get('filter_start_hz', 20)
        filter_end_hz = params.get('filter_end_hz', 10000)
        
        # Ensure stereo
        if seg_a.ndim == 1:
            seg_a = np.column_stack([seg_a, seg_a])
        if seg_b.ndim == 1:
            seg_b = np.column_stack([seg_b, seg_b])
        
        nyq = self.sr / 2
        seg_a_filtered = seg_a.copy()
        
        # Apply filter sweep to outgoing track
        chunk_size = int(0.5 * self.sr)  # 0.5 second chunks
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            progress = i / n_samples
            
            # Calculate cutoff for this chunk
            if filter_type == 'high_pass':
                cutoff_hz = filter_start_hz + (filter_end_hz - filter_start_hz) * progress
                cutoff = min(cutoff_hz / nyq, 0.99)
            else:  # low_pass
                cutoff_hz = filter_end_hz - (filter_end_hz - filter_start_hz) * progress
                cutoff = max(cutoff_hz / nyq, 0.01)
            
            # Apply filter (scipy expects 'highpass'/'lowpass', not 'high_pass'/'low_pass')
            btype = 'highpass' if filter_type == 'high_pass' else 'lowpass'
            b, a = signal.butter(2, cutoff, btype=btype)
            
            for ch in range(seg_a_filtered.shape[1]):
                chunk = seg_a_filtered[i:end_idx, ch]
                seg_a_filtered[i:end_idx, ch] = signal.filtfilt(b, a, chunk)
        
        # Regular crossfade
        vol_a, vol_b = crossfade_engine.create_equal_power_crossfade(n_samples, 'smooth')
        if vol_a.ndim == 1:
            vol_a = vol_a[:, np.newaxis]
        if vol_b.ndim == 1:
            vol_b = vol_b[:, np.newaxis]
        
        mixed = seg_a_filtered[:n_samples] * vol_a[:n_samples] + seg_b[:n_samples] * vol_b[:n_samples]
        return mixed
    
    def _execute_echo_out(self,
                         seg_a: np.ndarray,
                         seg_b: np.ndarray,
                         params: Dict,
                         seg_a_stems: Optional[Dict] = None,
                         seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """
        Execute echo out: add delay/reverb to outgoing track.
        """
        from src.crossfade_engine import CrossfadeEngine
        crossfade_engine = CrossfadeEngine(sr=self.sr)
        
        n_samples = min(len(seg_a), len(seg_b))
        delay_time_ms = params.get('delay_time_ms', 500)
        feedback = params.get('feedback', 0.4)
        wet_mix = params.get('wet_mix', 0.6)
        
        # Ensure stereo
        if seg_a.ndim == 1:
            seg_a = np.column_stack([seg_a, seg_a])
        if seg_b.ndim == 1:
            seg_b = np.column_stack([seg_b, seg_b])
        
        # Simple delay effect
        delay_samples = int(delay_time_ms * self.sr / 1000)
        seg_a_echo = seg_a.copy()
        
        # Apply delay
        for ch in range(seg_a_echo.shape[1]):
            dry = seg_a_echo[:, ch]
            delayed = np.zeros_like(dry)
            
            # Copy delayed signal
            if delay_samples < len(dry):
                delayed[delay_samples:] = dry[:-delay_samples] * feedback
                seg_a_echo[:, ch] = dry * (1 - wet_mix) + delayed * wet_mix
        
        # Regular crossfade with echo
        vol_a, vol_b = crossfade_engine.create_equal_power_crossfade(n_samples, 'smooth')
        if vol_a.ndim == 1:
            vol_a = vol_a[:, np.newaxis]
        if vol_b.ndim == 1:
            vol_b = vol_b[:, np.newaxis]
        
        mixed = seg_a_echo[:n_samples] * vol_a[:n_samples] + seg_b[:n_samples] * vol_b[:n_samples]
        return mixed
    
    def _execute_drop_mix(self,
                         seg_a: np.ndarray,
                         seg_b: np.ndarray,
                         params: Dict,
                         seg_a_stems: Optional[Dict] = None,
                         seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """
        Execute drop mix: energy dip before transition, then build.
        """
        from src.crossfade_engine import CrossfadeEngine
        crossfade_engine = CrossfadeEngine(sr=self.sr)
        
        n_samples = min(len(seg_a), len(seg_b))
        dip_ratio = params.get('dip_ratio', 0.3)  # Energy dip in first 30%
        
        # Create energy dip curve for seg_a
        dip_samples = int(n_samples * dip_ratio)
        energy_dip = np.ones(n_samples)
        
        # Dip in first portion
        if dip_samples > 0:
            dip_curve = 1.0 - 0.6 * np.sin(np.pi * np.linspace(0, 1, dip_samples))  # Dip to 0.4
            energy_dip[:dip_samples] = dip_curve
            # Fade out after dip
            energy_dip[dip_samples:] = np.linspace(energy_dip[dip_samples-1], 0.0, n_samples - dip_samples)
        
        # Build curve for seg_b
        energy_build = np.zeros(n_samples)
        energy_build[dip_samples:] = np.linspace(0.0, 1.0, n_samples - dip_samples)
        
        # Ensure stereo
        if seg_a.ndim == 1:
            seg_a = np.column_stack([seg_a, seg_a])
        if seg_b.ndim == 1:
            seg_b = np.column_stack([seg_b, seg_b])
        
        if energy_dip.ndim == 1:
            energy_dip = energy_dip[:, np.newaxis]
        if energy_build.ndim == 1:
            energy_build = energy_build[:, np.newaxis]
        
        mixed = seg_a[:n_samples] * energy_dip[:n_samples] + seg_b[:n_samples] * energy_build[:n_samples]
        return mixed
    
    def _execute_staggered_stem_mix(self,
                                   seg_a: np.ndarray,
                                   seg_b: np.ndarray,
                                   params: Dict,
                                   seg_a_stems: Optional[Dict] = None,
                                   seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """
        Execute staggered stem mix: beat first, vocals later.
        """
        if seg_a_stems is None or seg_b_stems is None:
            # Fallback to long_blend if stems not available
            return self._execute_long_blend(seg_a, seg_b, params, seg_a_stems, seg_b_stems)
        
        from src.crossfade_engine import CrossfadeEngine
        crossfade_engine = CrossfadeEngine(sr=self.sr)
        
        # Get reference length
        n_samples = len(seg_a_stems.get('drums', seg_a))
        
        # Stage timing ratios
        beat_mix_ratio = 0.2  # Song B beat starts at 20%
        vocal_a_fade_start = 0.3  # Song A vocals start fading at 30%
        vocal_a_fade_end = 0.7  # Song A vocals out by 70%
        vocal_b_start = 0.5  # Song B vocals start at 50%
        
        # Create multi-stage curves
        beat_a_fade = crossfade_engine.create_multi_stage_curve(n_samples, [
            {'start': 0.0, 'end': beat_mix_ratio, 'fade_type': 'hold', 'value': 1.0},
            {'start': beat_mix_ratio, 'end': 1.0, 'fade_type': 'exponential', 'end_value': 0.0}
        ])
        
        beat_b_fade = crossfade_engine.create_multi_stage_curve(n_samples, [
            {'start': 0.0, 'end': beat_mix_ratio, 'fade_type': 'hold', 'value': 0.0},
            {'start': beat_mix_ratio, 'end': 1.0, 'fade_type': 'smooth', 'end_value': 1.0}
        ])
        
        vocal_a_fade = crossfade_engine.create_multi_stage_curve(n_samples, [
            {'start': 0.0, 'end': vocal_a_fade_start, 'fade_type': 'hold', 'value': 1.0},
            {'start': vocal_a_fade_start, 'end': vocal_a_fade_end, 'fade_type': 'aggressive', 'end_value': 0.0},
            {'start': vocal_a_fade_end, 'end': 1.0, 'fade_type': 'hold', 'value': 0.0}
        ])
        
        vocal_b_fade = crossfade_engine.create_multi_stage_curve(n_samples, [
            {'start': 0.0, 'end': vocal_b_start, 'fade_type': 'hold', 'value': 0.0},
            {'start': vocal_b_start, 'end': 1.0, 'fade_type': 'smooth', 'end_value': 1.0}
        ])
        
        # Initialize mixed output
        mixed = np.zeros_like(seg_a_stems.get('drums', seg_a))
        if mixed.ndim == 1:
            mixed = np.column_stack([mixed, mixed])
        
        # Mix stems with appropriate curves
        def ensure_stereo(audio):
            if audio.ndim == 1:
                return np.column_stack([audio, audio])
            return audio
        
        def apply_fade_curve(audio, curve):
            audio = ensure_stereo(audio)
            if curve.ndim == 1:
                curve = curve[:, np.newaxis]
            return audio * curve[:len(audio)]
        
        # Drums
        if 'drums' in seg_a_stems:
            mixed += apply_fade_curve(seg_a_stems['drums'][:n_samples], beat_a_fade)
        if 'drums' in seg_b_stems:
            mixed += apply_fade_curve(seg_b_stems['drums'][:n_samples], beat_b_fade)
        
        # Vocals
        if 'vocals' in seg_a_stems:
            mixed += apply_fade_curve(seg_a_stems['vocals'][:n_samples], vocal_a_fade)
        if 'vocals' in seg_b_stems:
            mixed += apply_fade_curve(seg_b_stems['vocals'][:n_samples], vocal_b_fade)
        
        # Bass
        if 'bass' in seg_a_stems:
            mixed += apply_fade_curve(seg_a_stems['bass'][:n_samples], beat_a_fade)
        if 'bass' in seg_b_stems:
            mixed += apply_fade_curve(seg_b_stems['bass'][:n_samples], beat_b_fade)
        
        # Other instruments
        if 'other' in seg_a_stems:
            mixed += apply_fade_curve(seg_a_stems['other'][:n_samples], beat_a_fade)
        if 'other' in seg_b_stems:
            mixed += apply_fade_curve(seg_b_stems['other'][:n_samples], beat_b_fade)
        
        return mixed[:n_samples]
    
    def _execute_partial_stem_separation(self,
                                        seg_a: np.ndarray,
                                        seg_b: np.ndarray,
                                        params: Dict,
                                        seg_a_stems: Optional[Dict] = None,
                                        seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """
        Execute partial stem separation: different stems transition at different times.
        """
        if seg_a_stems is None or seg_b_stems is None:
            return self._execute_long_blend(seg_a, seg_b, params, seg_a_stems, seg_b_stems)
        
        from src.crossfade_engine import CrossfadeEngine
        crossfade_engine = CrossfadeEngine(sr=self.sr)
        
        n_samples = len(seg_a_stems.get('drums', seg_a))
        
        # Different transition times for different stems
        drums_transition_ratio = 0.2
        bass_transition_ratio = 0.4
        # Vocals/other: longer fade window (35%) and moderate (cosine) curve for gentler transition
        vocals_transition_ratio = 0.35
        other_transition_ratio = 0.40
        
        def create_stem_fade(start_ratio: float, n_samp: int, use_moderate: bool = False):
            if use_moderate:
                return crossfade_engine.create_multi_stage_curve(n_samp, [
                    {'start': 0.0, 'end': start_ratio, 'fade_type': 'hold', 'value': 1.0},
                    {'start': start_ratio, 'end': 1.0, 'fade_type': 'moderate', 'end_value': 0.0}
                ])
            return crossfade_engine.create_multi_stage_curve(n_samp, [
                {'start': 0.0, 'end': start_ratio, 'fade_type': 'hold', 'value': 1.0},
                {'start': start_ratio, 'end': 1.0, 'fade_type': 'smooth', 'end_value': 0.0}
            ])
        
        drums_fade_a = create_stem_fade(drums_transition_ratio, n_samples, use_moderate=False)
        bass_fade_a = create_stem_fade(bass_transition_ratio, n_samples, use_moderate=False)
        vocals_fade_a = create_stem_fade(vocals_transition_ratio, n_samples, use_moderate=True)
        other_fade_a = create_stem_fade(other_transition_ratio, n_samples, use_moderate=True)
        
        # Incoming fades (inverse)
        drums_fade_b = 1 - drums_fade_a
        bass_fade_b = 1 - bass_fade_a
        vocals_fade_b = 1 - vocals_fade_a
        other_fade_b = 1 - other_fade_a
        
        mixed = np.zeros_like(seg_a_stems.get('drums', seg_a))
        if mixed.ndim == 1:
            mixed = np.column_stack([mixed, mixed])
        
        def apply_fade(audio, curve):
            audio = audio.copy()
            if audio.ndim == 1:
                audio = np.column_stack([audio, audio])
            if curve.ndim == 1:
                curve = curve[:, np.newaxis]
            return audio[:n_samples] * curve[:n_samples]
        
        # Mix each stem at its own transition time
        if 'drums' in seg_a_stems:
            mixed += apply_fade(seg_a_stems['drums'], drums_fade_a)
        if 'drums' in seg_b_stems:
            mixed += apply_fade(seg_b_stems['drums'], drums_fade_b)
        
        if 'bass' in seg_a_stems:
            mixed += apply_fade(seg_a_stems['bass'], bass_fade_a)
        if 'bass' in seg_b_stems:
            mixed += apply_fade(seg_b_stems['bass'], bass_fade_b)
        
        if 'vocals' in seg_a_stems:
            mixed += apply_fade(seg_a_stems['vocals'], vocals_fade_a)
        if 'vocals' in seg_b_stems:
            mixed += apply_fade(seg_b_stems['vocals'], vocals_fade_b)
        
        if 'other' in seg_a_stems:
            mixed += apply_fade(seg_a_stems['other'], other_fade_a)
        if 'other' in seg_b_stems:
            mixed += apply_fade(seg_b_stems['other'], other_fade_b)
        
        return mixed[:n_samples]
    
    def _execute_vocal_layering(self,
                               seg_a: np.ndarray,
                               seg_b: np.ndarray,
                               params: Dict,
                               seg_a_stems: Optional[Dict] = None,
                               seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """
        Execute vocal layering: keep Song A vocals with Song B beat, transition vocals later.
        """
        if seg_a_stems is None or seg_b_stems is None:
            return self._execute_long_blend(seg_a, seg_b, params, seg_a_stems, seg_b_stems)
        
        from src.crossfade_engine import CrossfadeEngine
        crossfade_engine = CrossfadeEngine(sr=self.sr)
        
        n_samples = len(seg_a_stems.get('drums', seg_a))
        
        # Song B beat comes in first, Song A vocals continue
        beat_transition_ratio = 0.3
        vocal_transition_ratio = 0.6
        
        # Beat fades
        beat_a_fade = crossfade_engine.create_multi_stage_curve(n_samples, [
            {'start': 0.0, 'end': beat_transition_ratio, 'fade_type': 'hold', 'value': 1.0},
            {'start': beat_transition_ratio, 'end': 1.0, 'fade_type': 'smooth', 'end_value': 0.0}
        ])
        
        beat_b_fade = 1 - beat_a_fade
        
        # Vocals fade later
        vocal_a_fade = crossfade_engine.create_multi_stage_curve(n_samples, [
            {'start': 0.0, 'end': vocal_transition_ratio, 'fade_type': 'hold', 'value': 1.0},
            {'start': vocal_transition_ratio, 'end': 1.0, 'fade_type': 'aggressive', 'end_value': 0.0}
        ])
        
        vocal_b_fade = crossfade_engine.create_multi_stage_curve(n_samples, [
            {'start': 0.0, 'end': vocal_transition_ratio, 'fade_type': 'hold', 'value': 0.0},
            {'start': vocal_transition_ratio, 'end': 1.0, 'fade_type': 'smooth', 'end_value': 1.0}
        ])
        
        mixed = np.zeros_like(seg_a_stems.get('drums', seg_a))
        if mixed.ndim == 1:
            mixed = np.column_stack([mixed, mixed])
        
        def apply_fade(audio, curve):
            if audio.ndim == 1:
                audio = np.column_stack([audio, audio])
            if curve.ndim == 1:
                curve = curve[:, np.newaxis]
            return audio[:n_samples] * curve[:n_samples]
        
        # Song B beat with Song A vocals
        if 'drums' in seg_b_stems:
            mixed += apply_fade(seg_b_stems['drums'], beat_b_fade)
        if 'bass' in seg_b_stems:
            mixed += apply_fade(seg_b_stems['bass'], beat_b_fade)
        
        if 'vocals' in seg_a_stems:
            mixed += apply_fade(seg_a_stems['vocals'], vocal_a_fade)
        if 'vocals' in seg_b_stems:
            mixed += apply_fade(seg_b_stems['vocals'], vocal_b_fade)
        
        # Other instruments
        if 'other' in seg_a_stems:
            mixed += apply_fade(seg_a_stems['other'], beat_a_fade)
        if 'other' in seg_b_stems:
            mixed += apply_fade(seg_b_stems['other'], beat_b_fade)
        
        return mixed[:n_samples]

    def _execute_phrase_match(self,
                              seg_a: np.ndarray,
                              seg_b: np.ndarray,
                              params: Dict,
                              seg_a_stems: Optional[Dict] = None,
                              seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """Execute phrase match: align transition to phrase boundaries."""
        # Phrase matching is mostly about timing (handled upstream)
        return self._execute_long_blend(seg_a, seg_b, params, seg_a_stems, seg_b_stems)
    
    def _execute_backspin(self,
                         seg_a: np.ndarray,
                         seg_b: np.ndarray,
                         params: Dict,
                         seg_a_stems: Optional[Dict] = None,
                         seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """Execute backspin: spin out outgoing track (reverse/tape stop effect).
        Song A plays the backspin effect, then is cut to zero immediately when the effect
        ends (no tail) so only Song B continues."""
        from src.crossfade_engine import CrossfadeEngine
        crossfade_engine = CrossfadeEngine(sr=self.sr)
        
        n_samples = min(len(seg_a), len(seg_b))
        spin_duration_ratio = params.get('spin_duration_ratio', 0.6)
        tape_stop = params.get('tape_stop', True)
        # Short ramp (ms) at effect end to avoid click when cutting A to zero
        cut_ramp_ms = params.get('backspin_cut_ramp_ms', 50)
        cut_ramp_samples = min(int(self.sr * cut_ramp_ms / 1000), n_samples // 10)
        
        if seg_a.ndim == 1:
            seg_a = np.column_stack([seg_a, seg_a])
        if seg_b.ndim == 1:
            seg_b = np.column_stack([seg_b, seg_b])
        
        seg_a_spin = seg_a.copy()
        spin_start_idx = int(n_samples * (1 - spin_duration_ratio))
        
        if tape_stop:
            for ch in range(seg_a_spin.shape[1]):
                spin_section = seg_a_spin[spin_start_idx:, ch]
                t = np.linspace(1.0, 0.3, len(spin_section))
                indices = np.cumsum(t)
                indices = (indices / indices[-1] * (len(spin_section) - 1)).astype(int)
                indices = np.clip(indices, 0, len(spin_section) - 1)
                seg_a_spin[spin_start_idx:, ch] = spin_section[indices]
        
        # Song A: full level during backspin; when effect ends, cut to zero (short ramp to avoid click)
        fade_out = np.ones(n_samples)
        effect_end = n_samples - cut_ramp_samples  # effect "done" here; then ramp A to 0 over 50ms
        if effect_end > 0 and cut_ramp_samples > 0:
            fade_out[effect_end:] = np.linspace(1.0, 0.0, n_samples - effect_end)
        
        # Song B: fade in during backspin, full after A is cut
        fade_in = np.zeros(n_samples)
        fade_in[spin_start_idx:] = np.linspace(0.0, 1.0, n_samples - spin_start_idx)
        fade_in[effect_end:] = 1.0
        
        if fade_out.ndim == 1:
            fade_out = fade_out[:, np.newaxis]
        if fade_in.ndim == 1:
            fade_in = fade_in[:, np.newaxis]
        
        mixed = seg_a_spin[:n_samples] * fade_out[:n_samples] + seg_b[:n_samples] * fade_in[:n_samples]
        return mixed
    
    def _execute_double_drop(self,
                            seg_a: np.ndarray,
                            seg_b: np.ndarray,
                            params: Dict,
                            seg_a_stems: Optional[Dict] = None,
                            seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """Execute double drop: time two drops together for maximum energy peak."""
        from src.crossfade_engine import CrossfadeEngine
        crossfade_engine = CrossfadeEngine(sr=self.sr)
        
        n_samples = min(len(seg_a), len(seg_b))
        sync_point_ratio = params.get('sync_point_ratio', 0.5)
        energy_boost = params.get('energy_boost', 1.2)
        
        if seg_a.ndim == 1:
            seg_a = np.column_stack([seg_a, seg_a])
        if seg_b.ndim == 1:
            seg_b = np.column_stack([seg_b, seg_b])
        
        sync_point = int(n_samples * sync_point_ratio)
        
        vol_a = np.ones(n_samples)
        vol_a[sync_point:] = np.linspace(1.0, 0.0, n_samples - sync_point)
        
        vol_b = np.zeros(n_samples)
        vol_b[:sync_point] = np.linspace(0.0, 0.8, sync_point)
        boost_samples = int(self.sr * 0.5)
        boost_end = min(sync_point + boost_samples, n_samples)
        vol_b[sync_point:boost_end] = energy_boost
        vol_b[boost_end:] = np.linspace(energy_boost, 1.0, n_samples - boost_end)
        
        if vol_a.ndim == 1:
            vol_a = vol_a[:, np.newaxis]
        if vol_b.ndim == 1:
            vol_b = vol_b[:, np.newaxis]
        
        mixed = seg_a[:n_samples] * vol_a[:n_samples] + seg_b[:n_samples] * vol_b[:n_samples]
        return mixed
    
    def _execute_acapella_overlay(self,
                                  seg_a: np.ndarray,
                                  seg_b: np.ndarray,
                                  params: Dict,
                                  seg_a_stems: Optional[Dict] = None,
                                  seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """Execute acapella overlay: layer acapella from Track A over instrumental of Track B."""
        if seg_a_stems is None or seg_b_stems is None:
            return self._execute_long_blend(seg_a, seg_b, params, seg_a_stems, seg_b_stems)
        
        from src.crossfade_engine import CrossfadeEngine
        crossfade_engine = CrossfadeEngine(sr=self.sr)
        
        n_samples = len(seg_a_stems.get('drums', seg_a))
        overlay_start_ratio = params.get('overlay_start_ratio', 0.3)
        vocal_fade_out_ratio = params.get('vocal_fade_out_ratio', 0.7)
        
        vocal_a_fade = crossfade_engine.create_multi_stage_curve(n_samples, [
            {'start': 0.0, 'end': overlay_start_ratio, 'fade_type': 'hold', 'value': 0.0},
            {'start': overlay_start_ratio, 'end': vocal_fade_out_ratio, 'fade_type': 'smooth', 'end_value': 1.0},
            {'start': vocal_fade_out_ratio, 'end': 1.0, 'fade_type': 'smooth', 'end_value': 0.0}
        ])
        
        beat_b_fade = crossfade_engine.create_multi_stage_curve(n_samples, [
            {'start': 0.0, 'end': overlay_start_ratio, 'fade_type': 'hold', 'value': 0.0},
            {'start': overlay_start_ratio, 'end': 1.0, 'fade_type': 'smooth', 'end_value': 1.0}
        ])
        
        vocal_b_fade = crossfade_engine.create_multi_stage_curve(n_samples, [
            {'start': 0.0, 'end': vocal_fade_out_ratio, 'fade_type': 'hold', 'value': 0.0},
            {'start': vocal_fade_out_ratio, 'end': 1.0, 'fade_type': 'smooth', 'end_value': 1.0}
        ])
        
        mixed = np.zeros_like(seg_a_stems.get('drums', seg_a))
        if mixed.ndim == 1:
            mixed = np.column_stack([mixed, mixed])
        
        def apply_fade(audio, curve):
            if audio.ndim == 1:
                audio = np.column_stack([audio, audio])
            if curve.ndim == 1:
                curve = curve[:, np.newaxis]
            return audio[:n_samples] * curve[:n_samples]
        
        if 'vocals' in seg_a_stems:
            mixed += apply_fade(seg_a_stems['vocals'], vocal_a_fade)
        if 'drums' in seg_b_stems:
            mixed += apply_fade(seg_b_stems['drums'], beat_b_fade)
        if 'bass' in seg_b_stems:
            mixed += apply_fade(seg_b_stems['bass'], beat_b_fade)
        if 'other' in seg_b_stems:
            mixed += apply_fade(seg_b_stems['other'], beat_b_fade)
        if 'vocals' in seg_b_stems:
            mixed += apply_fade(seg_b_stems['vocals'], vocal_b_fade)
        
        return mixed[:n_samples]
    
    def _execute_modulation(self,
                           seg_a: np.ndarray,
                           seg_b: np.ndarray,
                           params: Dict,
                           seg_a_stems: Optional[Dict] = None,
                           seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """Execute modulation: smooth key change during transition (pitch-shift then crossfade)."""
        from src.crossfade_engine import CrossfadeEngine
        import librosa
        crossfade_engine = CrossfadeEngine(sr=self.sr)
        
        shift_a = params.get('shift_a_semitones', 0) or 0
        shift_b = params.get('shift_b_semitones', 0) or 0
        
        def _pitch_shift(y: np.ndarray, n_steps: float) -> np.ndarray:
            if abs(n_steps) < 0.01:
                return y
            if y.ndim == 1:
                return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
            out = np.zeros_like(y)
            for ch in range(y.shape[1]):
                out[:, ch] = librosa.effects.pitch_shift(y[:, ch], sr=self.sr, n_steps=n_steps)
            return out
        
        if shift_a != 0:
            seg_a = _pitch_shift(seg_a.copy(), float(shift_a))
        if shift_b != 0:
            seg_b = _pitch_shift(seg_b.copy(), float(shift_b))
        
        n_samples = min(len(seg_a), len(seg_b))
        vol_a, vol_b = crossfade_engine.create_equal_power_crossfade(n_samples, 'smooth')
        
        if seg_a.ndim == 1:
            seg_a = np.column_stack([seg_a, seg_a])
        if seg_b.ndim == 1:
            seg_b = np.column_stack([seg_b, seg_b])
        
        if vol_a.ndim == 1:
            vol_a = vol_a[:, np.newaxis]
        if vol_b.ndim == 1:
            vol_b = vol_b[:, np.newaxis]
        
        mixed = seg_a[:n_samples] * vol_a[:n_samples] + seg_b[:n_samples] * vol_b[:n_samples]
        return mixed
    
    def _execute_energy_build(self,
                             seg_a: np.ndarray,
                             seg_b: np.ndarray,
                             params: Dict,
                             seg_a_stems: Optional[Dict] = None,
                             seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """Execute energy build: progressive energy increase during transition."""
        from src.crossfade_engine import CrossfadeEngine
        from scipy import signal
        
        n_samples = min(len(seg_a), len(seg_b))
        build_curve = params.get('build_curve', 'exponential')
        filter_sweep = params.get('filter_sweep', True)
        eq_boost = params.get('eq_boost', True)
        
        if seg_a.ndim == 1:
            seg_a = np.column_stack([seg_a, seg_a])
        if seg_b.ndim == 1:
            seg_b = np.column_stack([seg_b, seg_b])
        
        seg_a_filtered = seg_a.copy()
        seg_b_boosted = seg_b.copy()
        
        if filter_sweep:
            nyq = self.sr / 2
            chunk_size = int(0.5 * self.sr)
            for i in range(0, n_samples, chunk_size):
                end_idx = min(i + chunk_size, n_samples)
                progress = i / n_samples
                cutoff = (200 + 200 * progress) / nyq
                cutoff = min(cutoff, 0.99)
                b, a = signal.butter(2, cutoff, btype='high')
                for ch in range(seg_a_filtered.shape[1]):
                    seg_a_filtered[i:end_idx, ch] = signal.filtfilt(b, a, seg_a_filtered[i:end_idx, ch])
        
        if eq_boost:
            nyq = self.sr / 2
            chunk_size = int(0.5 * self.sr)
            for i in range(0, n_samples, chunk_size):
                end_idx = min(i + chunk_size, n_samples)
                progress = i / n_samples
                boost = 1.0 + 0.3 * progress
                cutoff = 4000 / nyq
                b, a = signal.butter(2, cutoff, btype='high')
                for ch in range(seg_b_boosted.shape[1]):
                    highs = signal.filtfilt(b, a, seg_b_boosted[i:end_idx, ch])
                    seg_b_boosted[i:end_idx, ch] = seg_b_boosted[i:end_idx, ch] + highs * (boost - 1.0)
        
        t = np.linspace(0, 1, n_samples)
        if build_curve == 'exponential':
            vol_a = np.exp(-3 * t)
            vol_b = 1 - np.exp(-3 * (1 - t))
        elif build_curve == 'linear':
            vol_a = 1 - t
            vol_b = t
        else:
            vol_a = 1 - np.log(1 + t) / np.log(2)
            vol_b = np.log(1 + t) / np.log(2)
        
        if vol_a.ndim == 1:
            vol_a = vol_a[:, np.newaxis]
        if vol_b.ndim == 1:
            vol_b = vol_b[:, np.newaxis]
        
        mixed = seg_a_filtered[:n_samples] * vol_a[:n_samples] + seg_b_boosted[:n_samples] * vol_b[:n_samples]
        return mixed
    
    def _execute_loop_transition(self,
                                seg_a: np.ndarray,
                                seg_b: np.ndarray,
                                params: Dict,
                                seg_a_stems: Optional[Dict] = None,
                                seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """Execute loop transition: repeat a short loop from seg_a, then crossfade to seg_b."""
        from src.crossfade_engine import CrossfadeEngine
        crossfade_engine = CrossfadeEngine(sr=self.sr)
        n_samples = min(len(seg_a), len(seg_b))
        loop_length_bars = params.get('loop_length_bars', 4)
        loop_repeats = params.get('loop_repeats', 4)
        # Assume ~120 BPM: 4 beats/bar, so 1 bar = 2 sec
        bar_sec = 4.0 * (60.0 / 120.0)  # 2 sec per bar
        loop_sec = loop_length_bars * bar_sec
        loop_samples = min(int(loop_sec * self.sr), len(seg_a) // 2)
        loop_samples = max(loop_samples, int(0.5 * self.sr))  # at least 0.5 sec
        loop_portion_ratio = params.get('loop_portion_ratio', 0.4)
        loop_portion = int(n_samples * loop_portion_ratio)
        if seg_a.ndim == 1:
            seg_a = np.column_stack([seg_a, seg_a])
        if seg_b.ndim == 1:
            seg_b = np.column_stack([seg_b, seg_b])
        loop_slice = seg_a[-loop_samples:].copy()
        n_repeats = max(1, loop_portion // len(loop_slice) + 1)
        repeated = np.tile(loop_slice, (n_repeats, 1))[:loop_portion]
        if len(repeated) < loop_portion:
            pad = np.zeros((loop_portion - len(repeated), repeated.shape[1]), dtype=repeated.dtype)
            repeated = np.vstack([repeated, pad])
        seg_a_looped = np.zeros((n_samples, seg_a.shape[1]), dtype=seg_a.dtype)
        seg_a_looped[:loop_portion] = repeated[:loop_portion]
        crossfade_start = loop_portion
        crossfade_len = n_samples - crossfade_start
        vol_a = np.ones(n_samples)
        vol_a[crossfade_start:] = np.linspace(1.0, 0.0, crossfade_len)
        vol_b = np.zeros(n_samples)
        vol_b[:crossfade_start] = 0.0
        vol_b[crossfade_start:] = np.linspace(0.0, 1.0, crossfade_len)
        seg_a_looped[crossfade_start:] = seg_a[crossfade_start:n_samples]
        if vol_a.ndim == 1:
            vol_a = vol_a[:, np.newaxis]
        if vol_b.ndim == 1:
            vol_b = vol_b[:, np.newaxis]
        mixed = seg_a_looped * vol_a + seg_b[:n_samples] * vol_b
        return mixed
    
    def _execute_breakdown_to_build(self,
                                    seg_a: np.ndarray,
                                    seg_b: np.ndarray,
                                    params: Dict,
                                    seg_a_stems: Optional[Dict] = None,
                                    seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """Execute breakdown-to-build: transition from breakdown to build section."""
        n_samples = min(len(seg_a), len(seg_b))
        breakdown_ratio = params.get('breakdown_ratio', 0.4)
        
        breakdown_samples = int(n_samples * breakdown_ratio)
        
        vol_a = np.ones(n_samples)
        vol_a[breakdown_samples:] = np.linspace(1.0, 0.0, n_samples - breakdown_samples)
        
        vol_b = np.zeros(n_samples)
        vol_b[breakdown_samples:] = (np.linspace(0.0, 1.0, n_samples - breakdown_samples) ** 2)
        
        if seg_a.ndim == 1:
            seg_a = np.column_stack([seg_a, seg_a])
        if seg_b.ndim == 1:
            seg_b = np.column_stack([seg_b, seg_b])
        
        if vol_a.ndim == 1:
            vol_a = vol_a[:, np.newaxis]
        if vol_b.ndim == 1:
            vol_b = vol_b[:, np.newaxis]
        
        mixed = seg_a[:n_samples] * vol_a[:n_samples] + seg_b[:n_samples] * vol_b[:n_samples]
        return mixed

    def _execute_drop_on_the_one(self,
                                 seg_a: np.ndarray,
                                 seg_b: np.ndarray,
                                 params: Dict,
                                 seg_a_stems: Optional[Dict] = None,
                                 seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """Execute drop on the one: instant cut on downbeat with minimal crossfade."""
        n_samples = min(len(seg_a), len(seg_b))
        fade_ms = params.get('fade_ms', 50)
        fade_samples = min(int(fade_ms * self.sr / 1000), n_samples // 4)
        fade_out = np.ones(n_samples)
        if fade_samples > 0:
            fade_out[-fade_samples:] = np.linspace(1.0, 0.0, fade_samples)
        fade_in = np.zeros(n_samples)
        if fade_samples > 0:
            fade_in[:fade_samples] = np.linspace(0.0, 1.0, fade_samples)
        if seg_a.ndim == 1:
            seg_a = np.column_stack([seg_a, seg_a])
        if seg_b.ndim == 1:
            seg_b = np.column_stack([seg_b, seg_b])
        if fade_out.ndim == 1:
            fade_out = fade_out[:, np.newaxis]
        if fade_in.ndim == 1:
            fade_in = fade_in[:, np.newaxis]
        mixed = seg_a[:n_samples] * fade_out + seg_b[:n_samples] * fade_in
        return mixed

    def _execute_back_and_forth(self,
                               seg_a: np.ndarray,
                               seg_b: np.ndarray,
                               params: Dict,
                               seg_a_stems: Optional[Dict] = None,
                               seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """Execute back-and-forth: alternate A and B every N bars, then settle on B."""
        n_samples = min(len(seg_a), len(seg_b))
        switch_interval_bars = params.get('switch_interval_bars', 8)
        num_switches = params.get('num_switches', 2)
        bar_sec = 4.0 * (60.0 / 120.0)
        switch_samples = int(switch_interval_bars * bar_sec * self.sr)
        switch_samples = min(switch_samples, n_samples // (num_switches + 1))
        if switch_samples < 1:
            switch_samples = n_samples // (num_switches + 1)
        cf_samples = min(int(0.1 * self.sr), switch_samples // 4)
        if seg_a.ndim == 1:
            seg_a = np.column_stack([seg_a, seg_a])
        if seg_b.ndim == 1:
            seg_b = np.column_stack([seg_b, seg_b])
        mixed = np.zeros((n_samples, seg_a.shape[1]), dtype=seg_a.dtype)
        pos = 0
        use_a = True
        for _ in range(num_switches + 1):
            end = min(pos + switch_samples, n_samples)
            chunk = end - pos
            if chunk <= 0:
                break
            if use_a:
                mixed[pos:end] = seg_a[pos:end]
            else:
                mixed[pos:end] = seg_b[pos:end]
            pos = end
            use_a = not use_a
        if pos < n_samples:
            mixed[pos:] = seg_b[pos:n_samples]
        return mixed

    def _execute_drum_roll(self,
                           seg_a: np.ndarray,
                           seg_b: np.ndarray,
                           params: Dict,
                           seg_a_stems: Optional[Dict] = None,
                           seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """Execute drum roll: high-pass sweep on A + repeated slice for roll, then cut to B."""
        n_samples = min(len(seg_a), len(seg_b))
        roll_ratio = params.get('roll_duration_ratio', 0.5)
        roll_samples = int(n_samples * roll_ratio)
        slice_sec = 0.5
        slice_samples = min(int(slice_sec * self.sr), len(seg_a) // 4)
        slice_samples = max(slice_samples, int(0.1 * self.sr))
        if seg_a.ndim == 1:
            seg_a = np.column_stack([seg_a, seg_a])
        if seg_b.ndim == 1:
            seg_b = np.column_stack([seg_b, seg_b])
        seg_a_roll = np.zeros((n_samples, seg_a.shape[1]), dtype=seg_a.dtype)
        loop_slice = seg_a[-slice_samples:].copy()
        n_repeats = max(2, roll_samples // len(loop_slice) + 1)
        repeated = np.tile(loop_slice, (n_repeats, 1))[:roll_samples]
        if len(repeated) < roll_samples:
            pad = np.zeros((roll_samples - len(repeated), repeated.shape[1]), dtype=repeated.dtype)
            repeated = np.vstack([repeated, pad])
        seg_a_roll[:roll_samples] = repeated[:roll_samples]
        nyq = self.sr / 2
        chunk_size = int(0.25 * self.sr)
        for i in range(0, roll_samples, chunk_size):
            end_idx = min(i + chunk_size, roll_samples)
            progress = i / max(roll_samples, 1)
            cutoff_hz = 200 + (2000 - 200) * progress
            cutoff = min(cutoff_hz / nyq, 0.99)
            b, a = signal.butter(2, cutoff, btype='highpass')
            for ch in range(seg_a_roll.shape[1]):
                chunk = seg_a_roll[i:end_idx, ch]
                seg_a_roll[i:end_idx, ch] = signal.filtfilt(b, a, chunk)
        vol_ramp = np.linspace(0.7, 1.0, roll_samples)
        if vol_ramp.ndim == 1:
            vol_ramp = vol_ramp[:, np.newaxis]
        seg_a_roll[:roll_samples] *= vol_ramp
        crossfade_len = n_samples - roll_samples
        vol_a = np.ones(n_samples)
        vol_a[roll_samples:] = np.linspace(1.0, 0.0, crossfade_len)
        vol_b = np.zeros(n_samples)
        vol_b[roll_samples:] = np.linspace(0.0, 1.0, crossfade_len)
        seg_a_roll[roll_samples:] = seg_a[roll_samples:n_samples]
        if vol_a.ndim == 1:
            vol_a = vol_a[:, np.newaxis]
        if vol_b.ndim == 1:
            vol_b = vol_b[:, np.newaxis]
        mixed = seg_a_roll * vol_a + seg_b[:n_samples] * vol_b
        return mixed

    def _execute_thematic_handoff(self,
                                  seg_a: np.ndarray,
                                  seg_b: np.ndarray,
                                  params: Dict,
                                  seg_a_stems: Optional[Dict] = None,
                                  seg_b_stems: Optional[Dict] = None) -> np.ndarray:
        """Execute thematic handoff: same as phrase_match (phrase-aligned blend)."""
        return self._execute_phrase_match(seg_a, seg_b, params, seg_a_stems, seg_b_stems)
