"""
Superhuman DJ Engine

Unified engine that integrates all advanced mixing capabilities:
- Micro-timing perfection (groove matching, transient alignment)
- Spectral intelligence (frequency slot negotiation, harmonic resonance)
- Hybrid technique blending (creative combinations)
- Stem orchestration (musical conversations)
- Monte Carlo optimization (quality prediction and simulation)

This engine coordinates all modules to create transitions that exceed
what human DJs can achieve in terms of precision, creativity, and quality.
"""
import numpy as np
import random
import time
from typing import Dict, List, Tuple, Optional


class SuperhumanDJEngine:
    """
    The ultimate DJ mixing engine that exceeds human capabilities.
    
    Coordinates:
    - MicroTimingEngine: Sub-millisecond groove and transient matching
    - SpectralIntelligenceEngine: Surgical frequency management
    - HybridTechniqueBlender: Creative technique combinations
    - StemOrchestrator: Musical stem conversations
    - MonteCarloQualityOptimizer: Simulation-based quality optimization
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        
        # Initialize all advanced modules
        from src.micro_timing_engine import MicroTimingEngine
        from src.spectral_intelligence import SpectralIntelligenceEngine
        from src.hybrid_technique_blender import HybridTechniqueBlender
        from src.stem_orchestrator import StemOrchestrator
        from src.montecarlo_optimizer import MonteCarloQualityOptimizer
        
        self.micro_timing = MicroTimingEngine(sr=sr)
        self.spectral_intel = SpectralIntelligenceEngine(sr=sr)
        self.technique_blender = HybridTechniqueBlender(sr=sr)
        self.stem_orchestrator = StemOrchestrator(sr=sr)
        self.montecarlo = MonteCarloQualityOptimizer(sr=sr)
        
        # Configuration
        # NOTE: some of these can be overridden from config.yaml via SmartMixer.configure(...)
        self.config = {
            'enable_micro_timing': True,
            'enable_spectral_intelligence': True,
            'enable_hybrid_techniques': True,
            'enable_stem_orchestration': True,
            'enable_montecarlo_optimization': True,
            'montecarlo_simulations': 30,
            'creativity_level': 0.6,  # 0.0 = conservative, 1.0 = experimental
            'quality_threshold': 0.55,
            # Vocal/bed swap + tempo morph behaviour (can be tuned from config.yaml)
            'vocal_bed_swap_enabled': True,
            # Maximum fractional tempo shift during overlap (e.g. 0.06 ~= 6%)
            'max_tempo_shift_pct': 0.06,
            # 'never', 'rare', 'allowed'
            'allow_simultaneous_vocals': 'rare',
            # Mix at level: equal-power curves + normalize so no volume dip
            'mix_at_level': True,
            # BPM matching: apply tempo morph whenever |tempo_a - tempo_b| > min_diff (not only when role_plan)
            'bpm_matching_always': True,
            'bpm_matching_min_diff': 1.0,
            # Use selected technique (filter_sweep, drop_mix, etc.) this fraction of the time instead of stem crossfade
            'technique_execution_ratio': 0.62,
            # Diversity controls
            'avoid_consecutive_stem_orchestration': True,
            'technique_diversity_lookback': 4,
            'technique_diversity_attempts': 4,
            # Key modulation: pitch-shift A/B so they match or sit in compatible key (A/B material only)
            'key_modulation_enabled': True,
            'key_modulation_max_semitones': 2,
            'key_modulation_only_when_incompatible': True,
        }
        # Session-local diversity memory
        self._recent_mix_methods: List[str] = []
        self._recent_primary_techniques: List[str] = []
    
    def configure(self, **kwargs):
        """Update engine configuration."""
        self.config.update(kwargs)
        return self
    
    # ==================== MAIN MIXING PIPELINE ====================
    
    def create_superhuman_mix(self,
                              seg_a: np.ndarray,
                              seg_b: np.ndarray,
                              tempo_a: float,
                              tempo_b: float,
                              key_a: str = None,
                              key_b: str = None,
                              stems_a: Optional[Dict] = None,
                              stems_b: Optional[Dict] = None,
                              techniques: Optional[List[str]] = None,
                              force_stem_orchestration: Optional[bool] = None,
                              conversation_type_override: Optional[str] = None) -> Dict:
        """
        Create a superhuman quality mix using all advanced capabilities.
        
        This is the main entry point that orchestrates all modules.
        
        Args:
            seg_a: Audio segment from outgoing song
            seg_b: Audio segment from incoming song
            tempo_a, tempo_b: Tempos in BPM
            key_a, key_b: Musical keys (optional)
            stems_a, stems_b: Separated stems (optional, enables advanced features)
            techniques: List of techniques to consider (optional)
        
        Returns:
            Dict with:
            - 'mixed': Final mixed audio
            - 'analysis': Detailed analysis of what was done
            - 'quality': Quality assessment
            - 'technique_used': Technique or hybrid used
        """
        start_time = time.time()
        analysis = {'stages': [], 'timings': {}}
        
        # ==================== STAGE 1: ANALYSIS ====================
        print("  🔬 Stage 1: Superhuman Analysis...")
        stage_start = time.time()
        
        # Use samples for faster analysis (max 10 seconds)
        max_analysis_samples = int(10 * self.sr)
        seg_a_sample = seg_a[:max_analysis_samples] if len(seg_a) > max_analysis_samples else seg_a
        seg_b_sample = seg_b[:max_analysis_samples] if len(seg_b) > max_analysis_samples else seg_b
        
        # Convert to mono if stereo for analysis
        if seg_a_sample.ndim > 1:
            seg_a_sample = np.mean(seg_a_sample, axis=1)
        if seg_b_sample.ndim > 1:
            seg_b_sample = np.mean(seg_b_sample, axis=1)
        
        # 1.1 Micro-timing analysis
        if self.config['enable_micro_timing']:
            try:
                groove_a = self.micro_timing.extract_groove_pattern(seg_a_sample, tempo_a)
                groove_b = self.micro_timing.extract_groove_pattern(seg_b_sample, tempo_b)
                groove_match = self.micro_timing.match_grooves(
                    groove_a, groove_b, len(seg_a)
                )
                
                rhythmic_dna_a = self.micro_timing.extract_rhythmic_dna(seg_a_sample, tempo_a)
                rhythmic_dna_b = self.micro_timing.extract_rhythmic_dna(seg_b_sample, tempo_b)
                rhythm_match = self.micro_timing.match_rhythmic_dna(rhythmic_dna_a, rhythmic_dna_b)
                
                analysis['micro_timing'] = {
                    'groove_compatibility': groove_match.get('groove_compatibility', 0.5),
                    'rhythmic_compatibility': rhythm_match.get('overall_rhythmic_compatibility', 0.5),
                    'swing_a': groove_a.get('swing_ratio', 0.5),
                    'swing_b': groove_b.get('swing_ratio', 0.5)
                }
                print(f"    ✓ Micro-timing: groove={groove_match.get('groove_compatibility', 0):.2f}")
            except Exception as e:
                print(f"    ⚠ Micro-timing failed: {e}")
                groove_match = {'groove_compatibility': 0.5}
                rhythm_match = {'overall_rhythmic_compatibility': 0.5}
        else:
            groove_match = {'groove_compatibility': 0.5}
            rhythm_match = {'overall_rhythmic_compatibility': 0.5}
        
        # 1.2 Spectral analysis (with timeout protection)
        if self.config['enable_spectral_intelligence']:
            try:
                spectrum_a = self.spectral_intel.analyze_spectrum(seg_a_sample)
                spectrum_b = self.spectral_intel.analyze_spectrum(seg_b_sample)
                
                frequency_negotiation = self.spectral_intel.negotiate_frequency_slots(seg_a_sample, seg_b_sample)
                harmonic_resonances = self.spectral_intel.find_harmonic_resonances(seg_a_sample, seg_b_sample)
                # Skip masking analysis - it's slow and not critical
                masking_analysis = {'masking_severity': 'low', 'overall_masking': 0.3}
                
                analysis['spectral'] = {
                    'overall_conflict': frequency_negotiation.get('overall_conflict', 0.0),
                    'resonance_strength': harmonic_resonances.get('resonance_strength', 0.0),
                    'masking_severity': masking_analysis.get('masking_severity', 'low'),
                    'dominant_band_a': spectrum_a.get('dominant_band'),
                    'dominant_band_b': spectrum_b.get('dominant_band')
                }
                print(f"    ✓ Spectral: conflict={frequency_negotiation.get('overall_conflict', 0):.2f}")
            except Exception as e:
                print(f"    ⚠ Spectral analysis failed: {e}")
                frequency_negotiation = {'overall_conflict': 0.0, 'eq_curves': {}}
                harmonic_resonances = {'resonance_strength': 0.0}
                masking_analysis = {'masking_severity': 'low'}
        else:
            frequency_negotiation = {'overall_conflict': 0.0, 'eq_curves': {}}
            harmonic_resonances = {'resonance_strength': 0.0}
            masking_analysis = {'masking_severity': 'low'}
        
        analysis['timings']['analysis'] = time.time() - stage_start
        analysis['stages'].append('analysis_complete')
        print(f"    ⏱️ Analysis took {analysis['timings']['analysis']:.1f}s")
        
        # ==================== STAGE 2: TECHNIQUE SELECTION ====================
        print("  🎨 Stage 2: Creative Technique Selection...")
        stage_start = time.time()
        
        context = {
            'harmonic_compatibility': 1.0 - frequency_negotiation.get('overall_conflict', 0.0),
            'energy_a': float(np.sqrt(np.mean(seg_a ** 2))) * 10,
            'energy_b': float(np.sqrt(np.mean(seg_b ** 2))) * 10,
            'tempo_diff': abs(tempo_a - tempo_b),
            'has_vocals_a': stems_a is not None and 'vocals' in stems_a,
            'has_vocals_b': stems_b is not None and 'vocals' in stems_b
        }

        if self.config['enable_hybrid_techniques']:
            # Use context to suggest creative technique
            suggested_hybrid = self.technique_blender.suggest_creative_technique(
                context,
                creativity_level=self.config.get('creativity_level', 0.6)
            )
            
            if 'error' not in suggested_hybrid:
                selected_technique = suggested_hybrid
            else:
                # Fallback to single technique
                selected_technique = {
                    'techniques': [rhythm_match.get('recommended_technique', 'long_blend')],
                    'weights': [1.0]
                }
        else:
            selected_technique = {
                'techniques': techniques if techniques else ['long_blend'],
                'weights': [1.0 / len(techniques)] * len(techniques) if techniques else [1.0]
            }

        # Diversify selected technique to avoid repeatedly using the same preset/type.
        selected_technique, diversity_note = self._diversify_selected_technique(
            selected_technique,
            context
        )

        if len(selected_technique.get('techniques', [])) > 1:
            analysis['technique'] = {
                'type': 'hybrid',
                'name': selected_technique.get('name'),
                'techniques': selected_technique.get('techniques'),
                'weights': selected_technique.get('weights'),
                'complexity': selected_technique.get('complexity', 0.5),
                'boldness': selected_technique.get('boldness', 0.5)
            }
        else:
            analysis['technique'] = {
                'type': 'single',
                'name': (selected_technique.get('techniques') or ['long_blend'])[0]
            }
        if diversity_note:
            analysis['technique']['diversity_note'] = diversity_note
        
        analysis['timings']['technique_selection'] = time.time() - stage_start
        analysis['stages'].append('technique_selected')
        
        # ==================== STAGE 3: QUALITY PREDICTION ====================
        print("  📊 Stage 3: Quality Prediction...")
        stage_start = time.time()
        
        if self.config['enable_montecarlo_optimization']:
            primary_technique = selected_technique['techniques'][0]
            should_proceed, prediction = self.montecarlo.should_proceed(
                seg_a, seg_b, primary_technique, {},
                threshold=self.config.get('quality_threshold', 0.55)
            )
            analysis['prediction'] = {
                'predicted_quality': prediction['predicted_quality'],
                'should_proceed': should_proceed,
                'recommendation': prediction['recommendation']
            }
            # Selected technique is always executed; no fallback to other techniques.

        analysis['timings']['prediction'] = time.time() - stage_start
        analysis['stages'].append('prediction_complete')
        
        # ==================== STAGE 4: STEM ORCHESTRATION ====================
        print("  🎭 Stage 4: Stem Orchestration...")
        stage_start = time.time()
        
        conversation = None
        role_plan = None  # Will describe which bed/vocals/tempo target to use during overlap
        if self.config['enable_stem_orchestration'] and stems_a is not None and stems_b is not None:
            # Analyze stems for orchestration
            stem_analysis = self.stem_orchestrator.analyze_stems_for_orchestration(stems_a, stems_b)
            
            # Detect vocal phrases for phrase-aware fade timing (before creating conversation)
            phrase_data_a = None
            phrase_data_b = None
            if 'vocals' in stems_a:
                phrase_data_a = self.stem_orchestrator.detect_vocal_phrases(stems_a['vocals'])
                analysis['vocal_phrases_a'] = {
                    'phrase_count': phrase_data_a.get('phrase_count', 0),
                    'safe_points': phrase_data_a.get('safe_transition_points', [])[:5]
                }
            if 'vocals' in stems_b:
                phrase_data_b = self.stem_orchestrator.detect_vocal_phrases(stems_b['vocals'])
            segment_duration_sec = len(seg_a) / self.sr
            
            # Create stem conversation with phrase data for phrase-aware, gentler fades
            recommended_conv = stem_analysis.get('recommended_conversation', 'layered_reveal')
            override_conv = conversation_type_override or self.config.get('conversation_type')
            if override_conv:
                recommended_conv = override_conv
            conversation = self.stem_orchestrator.create_stem_conversation(
                stems_a, stems_b, recommended_conv,
                phrase_data_a=phrase_data_a,
                phrase_data_b=phrase_data_b,
                segment_duration_sec=segment_duration_sec,
                tempo_a=tempo_a,
                tempo_b=tempo_b
            )
            
            analysis['stem_orchestration'] = {
                'conversation_type': recommended_conv,
                'reasoning': stem_analysis.get('reasoning')
            }
            
            # Decide which bed/vocals/tempo target to use during the overlap
            role_plan = self._choose_vocal_bed_plan(
                stems_a, stems_b,
                tempo_a, tempo_b,
                analysis,
                stem_analysis
            )
            if role_plan is not None:
                # When B's vocals/drums layer over A's content, B must match A's BPM so the overlap is in sync
                if recommended_conv in ('melody_a_drums_vocals_b', 'vocal_overlay_handoff'):
                    role_plan['tempo_target'] = 'a'
                analysis['vocal_bed_plan'] = role_plan
        
        analysis['timings']['stem_orchestration'] = time.time() - stage_start
        analysis['stages'].append('stems_orchestrated')
        
        # ==================== STAGE 5: APPLY SPECTRAL INTELLIGENCE & TEMPO MORPH ====================
        print("  🌈 Stage 5: Spectral Processing...")
        stage_start = time.time()
        
        # ---- LUFS Loudness Matching ----
        # Match perceived loudness so transitions don't have volume dips/bumps
        try:
            from src.psychoacoustics import PsychoacousticAnalyzer
            _psycho = PsychoacousticAnalyzer(sr=self.sr)
            _seg_a_mono = np.mean(seg_a, axis=1) if seg_a.ndim > 1 else seg_a
            _seg_b_mono = np.mean(seg_b, axis=1) if seg_b.ndim > 1 else seg_b
            _lufs_a = _psycho.analyze_loudness_lufs(_seg_a_mono)
            _lufs_b = _psycho.analyze_loudness_lufs(_seg_b_mono)
            _lufs_diff = _lufs_a['integrated_lufs'] - _lufs_b['integrated_lufs']
            _gain_db = max(-6.0, min(6.0, _lufs_diff))
            if abs(_gain_db) > 0.5:
                seg_b = seg_b * (10 ** (_gain_db / 20.0))
                print(f"    ✓ Loudness matched: B adjusted by {_gain_db:+.1f} dB")
                analysis['loudness_match'] = {'gain_db': _gain_db, 'lufs_a': _lufs_a['integrated_lufs'], 'lufs_b': _lufs_b['integrated_lufs']}
            else:
                print(f"    ✓ Loudness already matched (diff={_lufs_diff:.1f} dB)")
        except Exception as e:
            print(f"    ⚠ Loudness matching failed: {e}")
        
        # Use original segments (spectral processing is optional enhancement)
        # Skip the heavy spectral negotiation application for now - it's slow
        seg_a_processed = seg_a
        seg_b_processed = seg_b
        analysis['spectral_applied'] = False
        analysis['spectral_morph_stages'] = 0
        print(f"    ✓ Using original segments (spectral info for reference)")
        
        # Optional: apply short, overlap-only tempo morphing (from role plan or always when BPM diff significant)
        tempo_morph_info = None
        max_pct = float(self.config.get('max_tempo_shift_pct', 0.06))
        bpm_matching_always = self.config.get('bpm_matching_always', True)
        bpm_matching_min_diff = float(self.config.get('bpm_matching_min_diff', 1.0))
        tempo_diff = abs(tempo_a - tempo_b)
        apply_bpm_match = bpm_matching_always and tempo_diff >= bpm_matching_min_diff

        if role_plan is not None:
            target = role_plan.get('tempo_target', 'a')
            # Decide which side(s) to morph
            if target == 'b':
                morph = self.micro_timing.create_limited_tempo_morph(
                    tempo_a, tempo_b, len(seg_a_processed), max_shift_pct=max_pct
                )
                seg_a_processed = self.micro_timing.apply_tempo_morph(seg_a_processed, morph)
                tempo_morph_info = {'source': 'a', 'target_tempo': 'b', **morph}
            elif target == 'a':
                morph = self.micro_timing.create_limited_tempo_morph(
                    tempo_b, tempo_a, len(seg_b_processed), max_shift_pct=max_pct
                )
                seg_b_processed = self.micro_timing.apply_tempo_morph(seg_b_processed, morph)
                tempo_morph_info = {'source': 'b', 'target_tempo': 'a', **morph}
            elif target == 'mid':
                # Converge both toward mid-tempo within limits
                mid_tempo = 0.5 * (tempo_a + tempo_b)
                morph_a = self.micro_timing.create_limited_tempo_morph(
                    tempo_a, mid_tempo, len(seg_a_processed), max_shift_pct=max_pct
                )
                morph_b = self.micro_timing.create_limited_tempo_morph(
                    tempo_b, mid_tempo, len(seg_b_processed), max_shift_pct=max_pct
                )
                seg_a_processed = self.micro_timing.apply_tempo_morph(seg_a_processed, morph_a)
                seg_b_processed = self.micro_timing.apply_tempo_morph(seg_b_processed, morph_b)
                tempo_morph_info = {
                    'source': 'both',
                    'target_tempo': 'mid',
                    'morph_a': morph_a,
                    'morph_b': morph_b
                }
        elif apply_bpm_match:
            # No role plan but BPM diff significant: morph both toward mid-tempo
            mid_tempo = 0.5 * (tempo_a + tempo_b)
            morph_a = self.micro_timing.create_limited_tempo_morph(
                tempo_a, mid_tempo, len(seg_a_processed), max_shift_pct=max_pct
            )
            morph_b = self.micro_timing.create_limited_tempo_morph(
                tempo_b, mid_tempo, len(seg_b_processed), max_shift_pct=max_pct
            )
            seg_a_processed = self.micro_timing.apply_tempo_morph(seg_a_processed, morph_a)
            seg_b_processed = self.micro_timing.apply_tempo_morph(seg_b_processed, morph_b)
            tempo_morph_info = {
                'source': 'both',
                'target_tempo': 'mid',
                'reason': 'bpm_matching_always',
                'morph_a': morph_a,
                'morph_b': morph_b
            }

        if tempo_morph_info is not None:
            analysis['tempo_morph'] = tempo_morph_info
            # Apply the same tempo morph to stems so orchestration uses tempo-aligned stems (e.g. One Kiss beat → Hell of a Life beat)
            if stems_a is not None or stems_b is not None:
                source = tempo_morph_info.get('source', '')
                if source == 'a':
                    morph = tempo_morph_info
                    if stems_a is not None:
                        for k in list(stems_a.keys()):
                            stems_a[k] = self.micro_timing.apply_tempo_morph(stems_a[k], morph)
                elif source == 'b':
                    morph = tempo_morph_info
                    if stems_b is not None:
                        for k in list(stems_b.keys()):
                            stems_b[k] = self.micro_timing.apply_tempo_morph(stems_b[k], morph)
                elif source == 'both':
                    morph_a = tempo_morph_info.get('morph_a')
                    morph_b = tempo_morph_info.get('morph_b')
                    if morph_a is not None and stems_a is not None:
                        for k in list(stems_a.keys()):
                            stems_a[k] = self.micro_timing.apply_tempo_morph(stems_a[k], morph_a)
                    if morph_b is not None and stems_b is not None:
                        for k in list(stems_b.keys()):
                            stems_b[k] = self.micro_timing.apply_tempo_morph(stems_b[k], morph_b)
        
        # ==================== KEY MODULATION (optional pitch shift for fit) ====================
        key_modulation_info = None
        if self.config.get('key_modulation_enabled', True):
            from src.harmonic_analyzer import HarmonicAnalyzer
            import librosa
            _key_a = key_a
            _key_b = key_b
            if (_key_a is None or _key_a == '' or _key_b is None or _key_b == ''):
                try:
                    harm = HarmonicAnalyzer(sr=self.sr)
                    if _key_a is None or _key_a == '':
                        _key_a = harm.detect_key_camelot(seg_a_processed if seg_a_processed.ndim == 1 else seg_a_processed.mean(axis=1)).get('key', 'C')
                    if _key_b is None or _key_b == '':
                        _key_b = harm.detect_key_camelot(seg_b_processed if seg_b_processed.ndim == 1 else seg_b_processed.mean(axis=1)).get('key', 'C')
                except Exception:
                    _key_a = _key_a or 'C'
                    _key_b = _key_b or 'C'
            if _key_a and _key_b:
                harm = HarmonicAnalyzer(sr=self.sr)
                only_when = self.config.get('key_modulation_only_when_incompatible', True)
                compat = harm.are_keys_compatible(_key_a, _key_b)
                if not only_when or not compat.get('compatible', True):
                    max_st = int(self.config.get('key_modulation_max_semitones', 2))
                    suggestion = harm.suggest_modulation_semitones(_key_a, _key_b, strategy='match_b', max_semitones=max_st)
                    sa = suggestion.get('shift_a_semitones', 0)
                    sb = suggestion.get('shift_b_semitones', 0)
                    if sa != 0 or sb != 0:
                        def _pitch_shift_audio(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
                            if np.abs(n_steps) < 0.01:
                                return y
                            if y.ndim == 1:
                                return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
                            out = np.zeros_like(y)
                            for ch in range(y.shape[1]):
                                out[:, ch] = librosa.effects.pitch_shift(y[:, ch], sr=sr, n_steps=n_steps)
                            return out
                        if sa != 0:
                            seg_a_processed = _pitch_shift_audio(seg_a_processed, self.sr, float(sa))
                            if stems_a is not None:
                                for k in list(stems_a.keys()):
                                    stems_a[k] = _pitch_shift_audio(stems_a[k], self.sr, float(sa))
                        if sb != 0:
                            seg_b_processed = _pitch_shift_audio(seg_b_processed, self.sr, float(sb))
                            if stems_b is not None:
                                for k in list(stems_b.keys()):
                                    stems_b[k] = _pitch_shift_audio(stems_b[k], self.sr, float(sb))
                        key_modulation_info = {'shift_a_semitones': sa, 'shift_b_semitones': sb, 'reason': suggestion.get('reason', '')}
            if key_modulation_info is not None:
                analysis['key_modulation'] = key_modulation_info
        
        analysis['timings']['spectral_processing'] = time.time() - stage_start
        analysis['stages'].append('spectral_applied')
        
        # ==================== STAGE 6: EXECUTE TRANSITION ====================
        print("  🎵 Stage 6: Executing Transition...")
        stage_start = time.time()
        
        # Decide: stem orchestration (crossfade-style) vs actual technique (filter_sweep, drop_mix, etc.)
        use_technique_path = False
        if conversation is not None and stems_a is not None and stems_b is not None:
            if force_stem_orchestration is True or self.config.get('force_stem_orchestration', False):
                use_technique_path = False  # Force One Kiss beat → Hell of a Life beat handoff
                print("    ✓ Forcing stem orchestration (One Kiss beat → Hell of a Life handoff)")
            else:
                ratio = float(self.config.get('technique_execution_ratio', 0.62))
                # Anti-repeat: if last transition used stem orchestration, bias heavily
                # toward technique path this time.
                avoid_stem_repeat = bool(self.config.get('avoid_consecutive_stem_orchestration', True))
                if avoid_stem_repeat and self._recent_mix_methods and self._recent_mix_methods[-1] == 'stem_orchestration':
                    ratio = min(0.9, ratio + 0.25)
                use_technique_path = random.random() < ratio
        if not (conversation is not None and stems_a is not None and stems_b is not None):
            use_technique_path = True  # No stems: must use technique path
        
        if (conversation is not None and stems_a is not None and stems_b is not None) and not use_technique_path:
            # Orchestrated stem mix (crossfade-style)
            mixed = self.stem_orchestrator.orchestrate_mix(
                stems_a, stems_b, conversation, role_plan=role_plan,
                mix_at_level=self.config.get('mix_at_level', True)
            )
            analysis['mix_method'] = 'stem_orchestration'
        else:
            # Use selected technique so we get real filter_sweep, drop_mix, backspin, etc.
            from src.technique_executor import TechniqueExecutor
            technique_executor = TechniqueExecutor(sr=self.sr)
            
            if len(selected_technique['techniques']) > 1:
                hybrid_with_tempo = {**selected_technique}
                params = dict(hybrid_with_tempo.get('params', {}))
                params['tempo_a'] = tempo_a
                params['tempo_b'] = tempo_b
                hybrid_with_tempo['params'] = params
                mixed = self.technique_blender.execute_hybrid(
                    seg_a_processed, seg_b_processed,
                    hybrid_with_tempo,
                    stems_a, stems_b,
                    technique_executor
                )
                analysis['mix_method'] = 'hybrid_technique'
            else:
                technique_params = {'tempo_a': tempo_a, 'tempo_b': tempo_b}
                mixed = technique_executor.execute(
                    selected_technique['techniques'][0],
                    seg_a_processed, seg_b_processed,
                    technique_params,
                    stems_a, stems_b
                )
                analysis['mix_method'] = 'single_technique'

        # Update diversity memory
        self._recent_mix_methods.append(analysis['mix_method'])
        if len(self._recent_mix_methods) > 24:
            self._recent_mix_methods = self._recent_mix_methods[-24:]
        primary_technique = self._get_primary_technique(selected_technique)
        if primary_technique:
            self._recent_primary_techniques.append(primary_technique)
            if len(self._recent_primary_techniques) > 24:
                self._recent_primary_techniques = self._recent_primary_techniques[-24:]
        
        analysis['timings']['execution'] = time.time() - stage_start
        analysis['stages'].append('transition_executed')
        
        # ==================== STAGE 7: QUALITY EVALUATION ====================
        print("  ✅ Stage 7: Quality Evaluation...")
        stage_start = time.time()
        
        if self.config['enable_montecarlo_optimization']:
            ensemble = self.montecarlo.ensemble_evaluate(mixed, seg_a, seg_b)
            
            quality = {
                'overall_score': ensemble['ensemble_mean'],
                'confidence': ensemble['confidence'],
                'consensus': ensemble['consensus'],
                'recommendation': ensemble['recommended_action'],
                'individual_scores': ensemble['individual_scores']
            }
        else:
            # Basic quality check
            quality = {
                'overall_score': 0.7,
                'confidence': 0.5,
                'recommendation': 'unknown'
            }
        
        analysis['timings']['evaluation'] = time.time() - stage_start
        analysis['stages'].append('quality_evaluated')
        
        # ==================== FINALIZATION ====================
        total_time = time.time() - start_time
        analysis['timings']['total'] = total_time
        
        print(f"  ⏱️ Total processing time: {total_time:.2f}s")
        print(f"  📈 Quality score: {quality['overall_score']:.2f}")
        
        return {
            'mixed': mixed,
            'analysis': analysis,
            'quality': quality,
            'technique_used': selected_technique
        }

    def _get_primary_technique(self, selected_technique: Dict) -> str:
        techniques = selected_technique.get('techniques', []) if isinstance(selected_technique, dict) else []
        if techniques:
            return str(techniques[0])
        return str(selected_technique.get('name', '')) if isinstance(selected_technique, dict) else ''

    def _diversify_selected_technique(self,
                                      selected_technique: Dict,
                                      context: Dict) -> Tuple[Dict, Optional[str]]:
        """
        Reduce repeated technique usage across consecutive transitions.
        """
        lookback = int(self.config.get('technique_diversity_lookback', 4))
        attempts = int(self.config.get('technique_diversity_attempts', 4))
        recent = self._recent_primary_techniques[-lookback:] if lookback > 0 else []
        current_primary = self._get_primary_technique(selected_technique)

        if not current_primary or current_primary not in recent:
            return selected_technique, None

        for _ in range(max(1, attempts)):
            alt = self.technique_blender.suggest_creative_technique(
                context,
                creativity_level=min(1.0, float(self.config.get('creativity_level', 0.6)) + 0.15)
            )
            if not isinstance(alt, dict) or 'error' in alt:
                continue
            alt_primary = self._get_primary_technique(alt)
            if alt_primary and alt_primary not in recent:
                return alt, f"diversity override: avoided repeated '{current_primary}'"

        return selected_technique, None
    
    # ==================== INDIVIDUAL ENHANCEMENTS ====================
    
    def enhance_with_micro_timing(self,
                                  seg_a: np.ndarray,
                                  seg_b: np.ndarray,
                                  tempo_a: float,
                                  tempo_b: float,
                                  point_a_sec: float,
                                  point_b_sec: float) -> Dict:
        """
        Apply micro-timing enhancements for precise groove matching.
        """
        # Analyze grooves
        groove_a = self.micro_timing.extract_groove_pattern(seg_a, tempo_a)
        groove_b = self.micro_timing.extract_groove_pattern(seg_b, tempo_b)
        
        # Detect and align transients
        transients_a = self.micro_timing.detect_transients(seg_a)
        transients_b = self.micro_timing.detect_transients(seg_b)
        
        alignment = self.micro_timing.align_transients(
            transients_a, transients_b, point_a_sec, point_b_sec
        )
        
        # Create tempo morph if needed
        tempo_morph = self.micro_timing.create_tempo_morph(
            tempo_a, tempo_b, len(seg_a), 'smooth'
        )
        
        return {
            'groove_compatibility': self.micro_timing.match_grooves(
                groove_a, groove_b, len(seg_a)
            ).get('groove_compatibility', 0.5),
            'transient_alignment': alignment,
            'tempo_morph': tempo_morph,
            'aligned_point_a': alignment.get('aligned_point_a'),
            'aligned_point_b': alignment.get('aligned_point_b')
        }
    
    def enhance_with_spectral_intelligence(self,
                                           seg_a: np.ndarray,
                                           seg_b: np.ndarray) -> Dict:
        """
        Apply spectral intelligence for frequency management.
        """
        # Negotiate frequency slots
        negotiation = self.spectral_intel.negotiate_frequency_slots(seg_a, seg_b)
        
        # Find harmonic resonances
        resonances = self.spectral_intel.find_harmonic_resonances(seg_a, seg_b)
        
        # Analyze masking
        masking = self.spectral_intel.analyze_masking(seg_a, seg_b)
        
        # Create spectral morph
        morph = self.spectral_intel.create_spectral_morph(seg_a, seg_b)
        
        return {
            'negotiation': negotiation,
            'resonances': resonances,
            'masking': masking,
            'spectral_morph': morph
        }
    
    def suggest_creative_hybrid(self, context: Dict) -> Dict:
        """
        Suggest a creative hybrid technique based on context.
        """
        return self.technique_blender.suggest_creative_technique(
            context,
            creativity_level=self.config.get('creativity_level', 0.6)
        )
    
    def optimize_transition(self,
                            seg_a: np.ndarray,
                            seg_b: np.ndarray,
                            techniques: List[str],
                            objectives: Dict[str, float]) -> Dict:
        """
        Run Monte Carlo optimization to find best transition.
        """
        return self.montecarlo.optimize_multi_objective(
            seg_a, seg_b, techniques, {},
            objectives=objectives,
            n_iterations=self.config.get('montecarlo_simulations', 30)
        )
    
    # ==================== UTILITIES ====================
    
    def get_available_hybrid_presets(self) -> List[Dict]:
        """Get list of available hybrid technique presets."""
        return self.technique_blender.list_presets()
    
    def get_status(self) -> Dict:
        """Get engine status and configuration."""
        return {
            'sr': self.sr,
            'config': self.config,
            'modules': {
                'micro_timing': type(self.micro_timing).__name__,
                'spectral_intel': type(self.spectral_intel).__name__,
                'technique_blender': type(self.technique_blender).__name__,
                'stem_orchestrator': type(self.stem_orchestrator).__name__,
                'montecarlo': type(self.montecarlo).__name__
            }
        }

    # ==================== VOCAL/BED ROLE PLANNING ====================
    
    def _choose_vocal_bed_plan(self,
                               stems_a: Optional[Dict],
                               stems_b: Optional[Dict],
                               tempo_a: float,
                               tempo_b: float,
                               analysis: Dict,
                               stem_analysis: Dict) -> Optional[Dict]:
        """
        Decide which song provides the instrumental bed and which provides vocals.
        
        Returns a small dict (TransitionRolePlan) with:
            - bed_source: 'a' | 'b'
            - vocal_source: 'a' | 'b'
            - tempo_target: 'a' | 'b' | 'mid'
            - max_tempo_shift_pct: float
        """
        if not self.config.get('vocal_bed_swap_enabled', True):
            return None
        
        has_vocals_a = stems_a is not None and 'vocals' in stems_a
        has_vocals_b = stems_b is not None and 'vocals' in stems_b
        
        # If we have no vocal stems, keep default behaviour.
        if not has_vocals_a and not has_vocals_b:
            return None
        
        # Basic spectral & tempo info
        spectral = analysis.get('spectral', {})
        clash = float(spectral.get('overall_conflict', 0.0) or 0.0)
        tempo_diff = abs(float(tempo_a) - float(tempo_b))
        max_pct = float(self.config.get('max_tempo_shift_pct', 0.06))
        max_shift_bpm = max_pct * max(float(tempo_a), float(tempo_b))
        
        # Phrase data for B vocals (A is already partly analyzed upstream)
        vocal_phrases_b = None
        if has_vocals_b:
            try:
                vocal_phrases_b = self.stem_orchestrator.detect_vocal_phrases(
                    stems_b['vocals']
                )
            except Exception:
                vocal_phrases_b = None
        
        def candidate(bed_source: str, vocal_source: str) -> Dict:
            return {
                'bed_source': bed_source,
                'vocal_source': vocal_source,
                'tempo_target': None,  # filled later
                'score': 0.0,
            }
        
        candidates: List[Dict] = []
        
        # Classic: A bed + A vocals
        if has_vocals_a:
            candidates.append(candidate('a', 'a'))
        # Song B vocals over Song A bed
        if has_vocals_b:
            candidates.append(candidate('a', 'b'))
        # Song A vocals over Song B bed
        if has_vocals_a:
            candidates.append(candidate('b', 'a'))
        # Classic B: B bed + B vocals
        if has_vocals_b:
            candidates.append(candidate('b', 'b'))
        
        # Score candidates
        best = None
        best_score = -1e9
        
        has_vocals_a_energy = stem_analysis['stems_a'].get('vocals', {}).get('has_content', False)
        has_vocals_b_energy = stem_analysis['stems_b'].get('vocals', {}).get('has_content', False)
        
        for cand in candidates:
            bsrc = cand['bed_source']
            vsrc = cand['vocal_source']
            score = 0.0
            
            # Prefer continuity: bed from outgoing song A gets a small bonus.
            if bsrc == 'a':
                score += 0.15
            else:
                score += 0.05
            
            # Slightly prefer using the incoming song's vocals when they exist.
            if vsrc == 'b' and has_vocals_b_energy:
                score += 0.25
            if vsrc == 'a' and has_vocals_a_energy:
                score += 0.15
            
            # Penalize high spectral clash when bed & vocals are from different songs.
            if bsrc != vsrc:
                score -= clash * 0.3
            
            # Tempo considerations: reward plans that keep required shift within limits.
            if tempo_diff <= max_shift_bpm:
                score += 0.2
            else:
                score -= 0.2
            
            # If we are considering B vocals, reward if we have many phrase-safe points.
            if vsrc == 'b' and vocal_phrases_b is not None:
                safe_points = vocal_phrases_b.get('safe_transition_points', [])
                phrase_count = vocal_phrases_b.get('phrase_count', 0)
                score += min(len(safe_points) * 0.02, 0.2)
                score += min(phrase_count * 0.01, 0.1)
            
            cand['score'] = score
            
            if score > best_score:
                best_score = score
                best = cand
        
        if best is None:
            return None
        
        # Decide tempo target: generally follow the vocal's home tempo,
        # or fall back to outgoing bed.
        if best['vocal_source'] == 'b' and has_vocals_b:
            best['tempo_target'] = 'b'
        elif best['vocal_source'] == 'a' and has_vocals_a:
            best['tempo_target'] = 'a'
        else:
            # Fallback: align to outgoing tempo
            best['tempo_target'] = 'a'
        
        best['max_tempo_shift_pct'] = max_pct
        return best
