"""
Stem Intelligence Orchestrator

Advanced stem-level processing that creates musical conversations:
- Intelligent stem orchestration (musical stem dialogues)
- Stem-level effect processing
- Vocal phrase awareness (never overlap phrases)
- Counter-melody creation (Song A melody over Song B)

This module treats stems as musical voices and orchestrates them together.
"""
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.ndimage import gaussian_filter1d


class StemOrchestrator:
    """
    Intelligent stem orchestration for creative mixing.
    
    Human DJs fade stems. This engine creates musical conversations
    between stems of different songs.
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        
        # Stem categories and their characteristics
        self.stem_profiles = {
            'drums': {
                'frequency_range': (40, 15000),
                'rhythmic': True,
                'harmonic': False,
                'priority': 1,  # Most important for groove
                'fade_sensitivity': 'low'  # Can fade quickly
            },
            'bass': {
                'frequency_range': (30, 300),
                'rhythmic': True,
                'harmonic': True,
                'priority': 2,
                'fade_sensitivity': 'medium'
            },
            'vocals': {
                'frequency_range': (100, 8000),
                'rhythmic': False,
                'harmonic': True,
                'priority': 3,  # Important for identity
                'fade_sensitivity': 'high'  # Needs careful fading
            },
            'other': {
                'frequency_range': (100, 16000),
                'rhythmic': False,
                'harmonic': True,
                'priority': 4,
                'fade_sensitivity': 'medium'
            }
        }
    
    # ==================== CONVERSATION CREATION ====================
    
    def create_stem_conversation(self,
                                 stems_a: Dict[str, np.ndarray],
                                 stems_b: Dict[str, np.ndarray],
                                 conversation_type: str = 'call_response') -> Dict:
        """
        Create a musical conversation between stems of two songs.
        
        Instead of simply fading, this creates interplay:
        - Call-response: Song A stems, then Song B responds
        - Interweave: Alternating stems create dialogue
        - Layered reveal: Gradually reveal Song B stem by stem
        - Counter melody: Song A melody over Song B rhythm
        
        Args:
            stems_a: Dict with keys 'drums', 'bass', 'vocals', 'other'
            stems_b: Dict with keys 'drums', 'bass', 'vocals', 'other'
            conversation_type: Type of conversation to create
        
        Returns:
            Conversation specification with timing and volume curves
        """
        available_stems = set(stems_a.keys()) & set(stems_b.keys())
        
        if conversation_type == 'call_response':
            return self._create_call_response(stems_a, stems_b, available_stems)
        elif conversation_type == 'interweave':
            return self._create_interweave(stems_a, stems_b, available_stems)
        elif conversation_type == 'layered_reveal':
            return self._create_layered_reveal(stems_a, stems_b, available_stems)
        elif conversation_type == 'counter_melody':
            return self._create_counter_melody(stems_a, stems_b, available_stems)
        else:
            return self._create_layered_reveal(stems_a, stems_b, available_stems)
    
    def _create_call_response(self, 
                              stems_a: Dict, 
                              stems_b: Dict,
                              available_stems: set) -> Dict:
        """
        Create call-response pattern:
        - Song A drums play, then Song B drums
        - Creates conversational rhythm
        """
        # Find reference length
        ref_stem = list(stems_a.values())[0]
        n_samples = len(ref_stem)
        
        # Call-response pattern: 8-16 bars alternating
        bar_samples = int(2.0 * self.sr)  # Approximate 2 seconds per bar at 120 BPM
        phrase_samples = bar_samples * 4  # 4 bars per phrase
        
        curves = {stem: {} for stem in available_stems}
        
        for stem in available_stems:
            curve_a = np.zeros(n_samples)
            curve_b = np.zeros(n_samples)
            
            # Alternate phrases
            phrase_idx = 0
            for start in range(0, n_samples, phrase_samples):
                end = min(start + phrase_samples, n_samples)
                
                if phrase_idx % 2 == 0:
                    # Song A phrase
                    curve_a[start:end] = 1.0
                    # Song B very quiet underneath
                    curve_b[start:end] = 0.15
                else:
                    # Song B phrase
                    curve_b[start:end] = 1.0
                    # Song A quiet underneath
                    curve_a[start:end] = 0.15
                
                phrase_idx += 1
            
            # Smooth transitions
            curve_a = gaussian_filter1d(curve_a, sigma=int(0.1 * self.sr))
            curve_b = gaussian_filter1d(curve_b, sigma=int(0.1 * self.sr))
            
            curves[stem] = {
                'a': curve_a.tolist(),
                'b': curve_b.tolist()
            }
        
        return {
            'type': 'call_response',
            'curves': curves,
            'description': 'Alternating phrases create musical dialogue'
        }
    
    def _create_interweave(self,
                           stems_a: Dict,
                           stems_b: Dict,
                           available_stems: set) -> Dict:
        """
        Create interweave pattern:
        - Different stems fade at different times
        - Creates complex texture
        """
        ref_stem = list(stems_a.values())[0]
        n_samples = len(ref_stem)
        
        # Staggered transition times
        transition_order = ['drums', 'bass', 'other', 'vocals']
        transition_ratios = [0.15, 0.35, 0.50, 0.65]
        
        curves = {}
        
        for stem in available_stems:
            if stem in transition_order:
                idx = transition_order.index(stem)
                ratio = transition_ratios[idx]
            else:
                ratio = 0.5
            
            transition_point = int(n_samples * ratio)
            transition_length = int(n_samples * 0.2)  # 20% of transition
            
            curve_a = np.ones(n_samples)
            curve_b = np.zeros(n_samples)
            
            # Create crossfade at transition point
            start = max(0, transition_point - transition_length // 2)
            end = min(n_samples, transition_point + transition_length // 2)
            
            t = np.linspace(0, 1, end - start)
            fade_out = np.sqrt(1 - t)  # Equal power
            fade_in = np.sqrt(t)
            
            curve_a[start:end] = fade_out
            curve_a[end:] = 0.0
            
            curve_b[:start] = 0.0
            curve_b[start:end] = fade_in
            curve_b[end:] = 1.0
            
            # Smooth
            curve_a = gaussian_filter1d(curve_a, sigma=1000)
            curve_b = gaussian_filter1d(curve_b, sigma=1000)
            
            curves[stem] = {
                'a': curve_a.tolist(),
                'b': curve_b.tolist()
            }
        
        return {
            'type': 'interweave',
            'curves': curves,
            'description': 'Stems transition at different times for complex texture'
        }
    
    def _create_layered_reveal(self,
                               stems_a: Dict,
                               stems_b: Dict,
                               available_stems: set) -> Dict:
        """
        Create layered reveal pattern:
        - Gradually introduce Song B stems one by one
        - Professional DJ technique
        """
        ref_stem = list(stems_a.values())[0]
        n_samples = len(ref_stem)
        
        # Reveal order: drums first (groove), then bass, other, finally vocals
        reveal_order = ['drums', 'bass', 'other', 'vocals']
        reveal_points = [0.10, 0.30, 0.50, 0.70]  # When each stem starts
        
        curves = {}
        
        for stem in available_stems:
            if stem in reveal_order:
                idx = reveal_order.index(stem)
                start_ratio = reveal_points[idx]
            else:
                start_ratio = 0.5
            
            start_sample = int(n_samples * start_ratio)
            fade_length = int(n_samples * 0.25)
            
            curve_a = np.ones(n_samples)
            curve_b = np.zeros(n_samples)
            
            # Song A stems fade out based on reveal timing
            a_fade_start = int(n_samples * (start_ratio + 0.1))
            a_fade_end = min(n_samples, a_fade_start + fade_length)
            
            if a_fade_end > a_fade_start:
                t = np.linspace(0, 1, a_fade_end - a_fade_start)
                curve_a[a_fade_start:a_fade_end] = np.sqrt(1 - t)
                curve_a[a_fade_end:] = 0.0
            
            # Song B stems fade in
            b_fade_end = min(start_sample + fade_length, n_samples)
            if b_fade_end > start_sample:
                t = np.linspace(0, 1, b_fade_end - start_sample)
                curve_b[start_sample:b_fade_end] = np.sqrt(t)
                curve_b[b_fade_end:] = 1.0
            
            # Smooth
            curve_a = gaussian_filter1d(curve_a, sigma=500)
            curve_b = gaussian_filter1d(curve_b, sigma=500)
            
            curves[stem] = {
                'a': curve_a.tolist(),
                'b': curve_b.tolist()
            }
        
        return {
            'type': 'layered_reveal',
            'curves': curves,
            'description': 'Gradually reveal Song B stems one by one'
        }
    
    def _create_counter_melody(self,
                               stems_a: Dict,
                               stems_b: Dict,
                               available_stems: set) -> Dict:
        """
        Create counter-melody effect:
        - Song A melody/vocals over Song B rhythm section
        - Creates unique hybrid
        """
        ref_stem = list(stems_a.values())[0]
        n_samples = len(ref_stem)
        
        curves = {}
        
        for stem in available_stems:
            curve_a = np.zeros(n_samples)
            curve_b = np.zeros(n_samples)
            
            if stem in ['vocals', 'other']:
                # Melodic stems: keep Song A longer, blend with Song B
                # Counter-melody phase: 20% to 70%
                counter_start = int(n_samples * 0.20)
                counter_end = int(n_samples * 0.70)
                
                # Song A melodic stems stay full, then fade
                curve_a[:counter_end] = 1.0
                fade_len = n_samples - counter_end
                if fade_len > 0:
                    curve_a[counter_end:] = np.linspace(1.0, 0.0, fade_len)
                
                # Song B melodic stems fade in at end
                curve_b[:counter_start] = 0.0
                if counter_end > counter_start:
                    t = np.linspace(0, 0.3, counter_end - counter_start)
                    curve_b[counter_start:counter_end] = t  # Quiet underneath
                curve_b[counter_end:] = np.linspace(0.3, 1.0, n_samples - counter_end)
                
            elif stem in ['drums', 'bass']:
                # Rhythmic stems: Song B takes over earlier
                transition_ratio = 0.15 if stem == 'drums' else 0.25
                transition_point = int(n_samples * transition_ratio)
                fade_length = int(n_samples * 0.15)
                
                fade_end = min(transition_point + fade_length, n_samples)
                
                curve_a[:transition_point] = 1.0
                if fade_end > transition_point:
                    t = np.linspace(0, 1, fade_end - transition_point)
                    curve_a[transition_point:fade_end] = 1.0 - t
                curve_a[fade_end:] = 0.0
                
                curve_b[:transition_point] = 0.0
                if fade_end > transition_point:
                    curve_b[transition_point:fade_end] = t
                curve_b[fade_end:] = 1.0
            
            # Smooth
            curve_a = gaussian_filter1d(curve_a, sigma=500)
            curve_b = gaussian_filter1d(curve_b, sigma=500)
            
            curves[stem] = {
                'a': curve_a.tolist(),
                'b': curve_b.tolist()
            }
        
        return {
            'type': 'counter_melody',
            'curves': curves,
            'description': 'Song A melody plays over Song B rhythm'
        }
    
    # ==================== VOCAL PHRASE DETECTION ====================
    
    def detect_vocal_phrases(self, vocal_stem: np.ndarray) -> Dict:
        """
        Detect vocal phrases to avoid cutting in the middle of words.
        
        Human speech has natural pauses between phrases.
        We detect these to place transitions.
        """
        if len(vocal_stem) == 0:
            return {'phrases': [], 'phrase_boundaries': []}
        
        # Convert to mono if stereo
        if vocal_stem.ndim > 1:
            vocal_mono = np.mean(vocal_stem, axis=1)
        else:
            vocal_mono = vocal_stem
        
        # Calculate RMS energy with small window
        frame_length = int(0.05 * self.sr)  # 50ms frames
        hop_length = frame_length // 2
        
        rms = librosa.feature.rms(
            y=vocal_mono,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Smooth RMS
        rms_smooth = gaussian_filter1d(rms, sigma=3)
        
        # Find silence/low energy regions (phrase boundaries)
        threshold = np.mean(rms_smooth) * 0.3
        is_silence = rms_smooth < threshold
        
        # Find boundaries (transitions from sound to silence)
        boundaries = []
        in_phrase = False
        phrase_start = 0
        
        for i, silent in enumerate(is_silence):
            if not silent and not in_phrase:
                # Start of phrase
                phrase_start = i
                in_phrase = True
            elif silent and in_phrase:
                # End of phrase
                phrase_end = i
                boundaries.append({
                    'start_frame': phrase_start,
                    'end_frame': phrase_end,
                    'start_sec': float(phrase_start * hop_length / self.sr),
                    'end_sec': float(phrase_end * hop_length / self.sr),
                    'duration_sec': float((phrase_end - phrase_start) * hop_length / self.sr)
                })
                in_phrase = False
        
        # Handle phrase extending to end
        if in_phrase:
            boundaries.append({
                'start_frame': phrase_start,
                'end_frame': len(is_silence) - 1,
                'start_sec': float(phrase_start * hop_length / self.sr),
                'end_sec': float(len(vocal_mono) / self.sr),
                'duration_sec': float((len(is_silence) - 1 - phrase_start) * hop_length / self.sr)
            })
        
        # Find best transition points (silence gaps)
        silence_regions = []
        silence_start = None
        
        for i, silent in enumerate(is_silence):
            if silent and silence_start is None:
                silence_start = i
            elif not silent and silence_start is not None:
                silence_duration = i - silence_start
                if silence_duration > 5:  # At least 5 frames (~125ms)
                    silence_regions.append({
                        'center_sec': float((silence_start + i) / 2 * hop_length / self.sr),
                        'duration_sec': float(silence_duration * hop_length / self.sr)
                    })
                silence_start = None
        
        return {
            'phrases': boundaries,
            'phrase_boundaries': silence_regions,
            'phrase_count': len(boundaries),
            'safe_transition_points': [s['center_sec'] for s in silence_regions]
        }
    
    def find_safe_vocal_transition_point(self,
                                         current_point_sec: float,
                                         phrase_data: Dict,
                                         search_window_sec: float = 2.0) -> float:
        """
        Find a safe transition point that doesn't cut vocals mid-phrase.
        """
        safe_points = phrase_data.get('safe_transition_points', [])
        
        if len(safe_points) == 0:
            return current_point_sec
        
        # Find closest safe point within window
        candidates = [
            p for p in safe_points
            if abs(p - current_point_sec) <= search_window_sec
        ]
        
        if len(candidates) == 0:
            return current_point_sec
        
        # Return closest
        return min(candidates, key=lambda x: abs(x - current_point_sec))
    
    # ==================== STEM-LEVEL EFFECTS ====================
    
    def apply_stem_effect(self,
                          stem: np.ndarray,
                          effect_type: str,
                          params: Dict,
                          progress: float = 0.0) -> np.ndarray:
        """
        Apply effect to individual stem.
        
        Args:
            stem: Audio stem
            effect_type: 'filter', 'reverb', 'delay', 'pitch_shift'
            params: Effect-specific parameters
            progress: Transition progress (0-1)
        """
        if effect_type == 'filter':
            return self._apply_filter(stem, params, progress)
        elif effect_type == 'reverb':
            return self._apply_reverb(stem, params, progress)
        elif effect_type == 'delay':
            return self._apply_delay(stem, params, progress)
        elif effect_type == 'pitch_shift':
            return self._apply_pitch_shift(stem, params, progress)
        else:
            return stem
    
    def _apply_filter(self, 
                      stem: np.ndarray,
                      params: Dict,
                      progress: float) -> np.ndarray:
        """Apply dynamic filter to stem."""
        filter_type = params.get('type', 'lowpass')
        start_freq = params.get('start_freq', 20000)
        end_freq = params.get('end_freq', 200)
        
        # Calculate current frequency
        current_freq = start_freq + (end_freq - start_freq) * progress
        current_freq = max(20, min(20000, current_freq))
        
        nyq = self.sr / 2
        normalized_freq = current_freq / nyq
        normalized_freq = max(0.001, min(0.999, normalized_freq))
        
        try:
            if filter_type == 'lowpass':
                sos = signal.butter(4, normalized_freq, btype='low', output='sos')
            else:
                sos = signal.butter(4, normalized_freq, btype='high', output='sos')
            
            return signal.sosfilt(sos, stem)
        except:
            return stem
    
    def _apply_reverb(self,
                      stem: np.ndarray,
                      params: Dict,
                      progress: float) -> np.ndarray:
        """Apply simple reverb (convolution with impulse response)."""
        wet_mix = params.get('wet_mix', 0.3) * progress
        decay = params.get('decay', 0.5)
        
        # Create simple impulse response
        ir_length = int(decay * self.sr)
        t = np.linspace(0, 1, ir_length)
        ir = np.exp(-5 * t) * np.random.randn(ir_length) * 0.1
        
        # Convolve
        wet = signal.fftconvolve(stem, ir, mode='same')
        
        # Mix
        return stem * (1 - wet_mix) + wet * wet_mix
    
    def _apply_delay(self,
                     stem: np.ndarray,
                     params: Dict,
                     progress: float) -> np.ndarray:
        """Apply delay effect."""
        delay_ms = params.get('delay_ms', 500)
        feedback = params.get('feedback', 0.4)
        wet_mix = params.get('wet_mix', 0.5) * progress
        
        delay_samples = int(delay_ms * self.sr / 1000)
        
        if stem.ndim == 1:
            delayed = np.zeros_like(stem)
            if delay_samples < len(stem):
                delayed[delay_samples:] = stem[:-delay_samples] * feedback
            return stem * (1 - wet_mix) + delayed * wet_mix
        else:
            result = stem.copy()
            for ch in range(stem.shape[1]):
                delayed = np.zeros(len(stem))
                if delay_samples < len(stem):
                    delayed[delay_samples:] = stem[:-delay_samples, ch] * feedback
                result[:, ch] = stem[:, ch] * (1 - wet_mix) + delayed * wet_mix
            return result
    
    def _apply_pitch_shift(self,
                           stem: np.ndarray,
                           params: Dict,
                           progress: float) -> np.ndarray:
        """Apply pitch shift (semitones)."""
        semitones = params.get('semitones', 0)
        
        if semitones == 0:
            return stem
        
        try:
            return librosa.effects.pitch_shift(
                stem, sr=self.sr, n_steps=semitones * progress
            )
        except:
            return stem
    
    # ==================== ORCHESTRATED MIXING ====================
    
    def orchestrate_mix(self,
                        stems_a: Dict[str, np.ndarray],
                        stems_b: Dict[str, np.ndarray],
                        conversation: Dict,
                        apply_effects: bool = True,
                        effect_params: Optional[Dict] = None) -> np.ndarray:
        """
        Create the final mix from stem conversation specification.
        
        Args:
            stems_a: Dict of stems from Song A
            stems_b: Dict of stems from Song B
            conversation: Conversation spec from create_stem_conversation
            apply_effects: Whether to apply stem effects
            effect_params: Effect parameters per stem
        
        Returns:
            Mixed audio
        """
        curves = conversation.get('curves', {})
        
        # Get reference length
        ref_stem = next((s for s in stems_a.values() if len(s) > 0), None)
        if ref_stem is None:
            ref_stem = next((s for s in stems_b.values() if len(s) > 0), None)
        
        if ref_stem is None:
            return np.zeros(44100)  # 1 second of silence
        
        n_samples = len(ref_stem)
        
        # Determine output shape
        if ref_stem.ndim == 1:
            mixed = np.zeros(n_samples)
        else:
            mixed = np.zeros((n_samples, ref_stem.shape[1]))
        
        common_stems = set(stems_a.keys()) & set(stems_b.keys()) & set(curves.keys())
        
        for stem_name in common_stems:
            stem_a = stems_a.get(stem_name, np.zeros(n_samples))
            stem_b = stems_b.get(stem_name, np.zeros(n_samples))
            
            # Get curves
            curve_spec = curves.get(stem_name, {})
            curve_a = np.array(curve_spec.get('a', [1.0] * n_samples))
            curve_b = np.array(curve_spec.get('b', [0.0] * n_samples))
            
            # Interpolate curves to match audio length
            if len(curve_a) != n_samples:
                curve_a = np.interp(
                    np.linspace(0, len(curve_a) - 1, n_samples),
                    np.arange(len(curve_a)),
                    curve_a
                )
            if len(curve_b) != n_samples:
                curve_b = np.interp(
                    np.linspace(0, len(curve_b) - 1, n_samples),
                    np.arange(len(curve_b)),
                    curve_b
                )
            
            # Apply effects if requested
            if apply_effects and effect_params:
                stem_effects = effect_params.get(stem_name, {})
                for effect_type, eparams in stem_effects.items():
                    if stem_name in ['drums', 'bass']:
                        # Apply filter to outgoing
                        for i in range(10):
                            progress = i / 10
                            chunk_start = int(n_samples * i / 10)
                            chunk_end = int(n_samples * (i + 1) / 10)
                            stem_a[chunk_start:chunk_end] = self.apply_stem_effect(
                                stem_a[chunk_start:chunk_end],
                                effect_type, eparams, progress
                            )
            
            # Apply curves
            samples = min(len(stem_a), len(stem_b), n_samples)
            
            if stem_a.ndim == 1:
                contribution_a = stem_a[:samples] * curve_a[:samples]
                contribution_b = stem_b[:samples] * curve_b[:samples]
            else:
                contribution_a = stem_a[:samples] * curve_a[:samples, np.newaxis]
                contribution_b = stem_b[:samples] * curve_b[:samples, np.newaxis]
            
            mixed[:samples] += contribution_a + contribution_b
        
        # Normalize
        max_val = np.max(np.abs(mixed))
        if max_val > 0.95:
            mixed = mixed * (0.95 / max_val)
        
        return mixed
    
    def analyze_stems_for_orchestration(self,
                                        stems_a: Dict[str, np.ndarray],
                                        stems_b: Dict[str, np.ndarray]) -> Dict:
        """
        Analyze stems to recommend orchestration strategy.
        """
        analysis = {'stems_a': {}, 'stems_b': {}}
        
        for name, stem in stems_a.items():
            if len(stem) > 0:
                energy = float(np.sqrt(np.mean(stem ** 2)))
                analysis['stems_a'][name] = {
                    'energy': energy,
                    'has_content': energy > 0.01
                }
        
        for name, stem in stems_b.items():
            if len(stem) > 0:
                energy = float(np.sqrt(np.mean(stem ** 2)))
                analysis['stems_b'][name] = {
                    'energy': energy,
                    'has_content': energy > 0.01
                }
        
        # Recommend conversation type
        has_vocals_a = analysis['stems_a'].get('vocals', {}).get('has_content', False)
        has_vocals_b = analysis['stems_b'].get('vocals', {}).get('has_content', False)
        
        if has_vocals_a and has_vocals_b:
            # Both have vocals - use careful layering
            recommended = 'interweave'
        elif has_vocals_a and not has_vocals_b:
            # Song A vocals can play over Song B
            recommended = 'counter_melody'
        elif has_vocals_b and not has_vocals_a:
            # Reveal Song B vocals last
            recommended = 'layered_reveal'
        else:
            # Instrumental - call and response works well
            recommended = 'call_response'
        
        analysis['recommended_conversation'] = recommended
        analysis['reasoning'] = f"Based on vocal content: A has vocals={has_vocals_a}, B has vocals={has_vocals_b}"
        
        return analysis
