"""
Hybrid Technique Blender

Creative technique blending that exceeds standard DJ capabilities:
- Smooth blending of 2-3 techniques simultaneously
- Novel technique generation through combination
- Context-aware creativity adjustments
- Dynamic technique morphing during transitions

This module creates transitions that are genuinely novel - not just
choosing from a fixed menu, but creating custom hybrid techniques.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter1d
import random


class HybridTechniqueBlender:
    """
    Blends multiple DJ techniques together for creative transitions.
    
    Human DJs usually use one technique at a time.
    This engine layers 2-3 techniques with intelligent weighting.
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        
        # Define technique characteristics for intelligent blending
        self.technique_profiles = {
            'long_blend': {
                'category': 'crossfade',
                'energy_change': 'stable',
                'complexity': 0.3,
                'boldness': 0.2,
                'compatible_with': ['bass_swap', 'filter_sweep', 'echo_out'],
                'incompatible_with': ['quick_cut', 'backspin']
            },
            'quick_cut': {
                'category': 'cut',
                'energy_change': 'sudden',
                'complexity': 0.2,
                'boldness': 0.8,
                'compatible_with': ['double_drop'],
                'incompatible_with': ['long_blend', 'echo_out', 'filter_sweep']
            },
            'bass_swap': {
                'category': 'frequency',
                'energy_change': 'stable',
                'complexity': 0.5,
                'boldness': 0.4,
                'compatible_with': ['long_blend', 'filter_sweep', 'staggered_stem_mix'],
                'incompatible_with': ['quick_cut']
            },
            'filter_sweep': {
                'category': 'frequency',
                'energy_change': 'building',
                'complexity': 0.4,
                'boldness': 0.5,
                'compatible_with': ['long_blend', 'bass_swap', 'energy_build'],
                'incompatible_with': ['quick_cut']
            },
            'echo_out': {
                'category': 'effect',
                'energy_change': 'fading',
                'complexity': 0.4,
                'boldness': 0.6,
                'compatible_with': ['long_blend', 'drop_mix'],
                'incompatible_with': ['quick_cut', 'double_drop']
            },
            'drop_mix': {
                'category': 'energy',
                'energy_change': 'dip_then_build',
                'complexity': 0.6,
                'boldness': 0.7,
                'compatible_with': ['energy_build', 'double_drop'],
                'incompatible_with': ['long_blend']
            },
            'staggered_stem_mix': {
                'category': 'stem',
                'energy_change': 'gradual',
                'complexity': 0.7,
                'boldness': 0.4,
                'compatible_with': ['bass_swap', 'vocal_layering'],
                'incompatible_with': ['quick_cut']
            },
            'vocal_layering': {
                'category': 'stem',
                'energy_change': 'gradual',
                'complexity': 0.6,
                'boldness': 0.5,
                'compatible_with': ['staggered_stem_mix', 'long_blend'],
                'incompatible_with': ['quick_cut']
            },
            'energy_build': {
                'category': 'energy',
                'energy_change': 'building',
                'complexity': 0.5,
                'boldness': 0.6,
                'compatible_with': ['filter_sweep', 'drop_mix'],
                'incompatible_with': ['echo_out']
            },
            'double_drop': {
                'category': 'energy',
                'energy_change': 'peak',
                'complexity': 0.5,
                'boldness': 0.9,
                'compatible_with': ['drop_mix', 'quick_cut'],
                'incompatible_with': ['long_blend', 'echo_out']
            },
            'phrase_match': {
                'category': 'timing',
                'energy_change': 'stable',
                'complexity': 0.3,
                'boldness': 0.3,
                'compatible_with': ['long_blend', 'bass_swap', 'staggered_stem_mix'],
                'incompatible_with': []
            },
            'backspin': {
                'category': 'effect',
                'energy_change': 'dramatic',
                'complexity': 0.4,
                'boldness': 0.8,
                'compatible_with': ['quick_cut', 'drop_mix'],
                'incompatible_with': ['long_blend', 'filter_sweep']
            },
            'loop_transition': {
                'category': 'timing',
                'energy_change': 'stable',
                'complexity': 0.5,
                'boldness': 0.3,
                'compatible_with': ['long_blend', 'bass_swap', 'phrase_match'],
                'incompatible_with': ['quick_cut', 'backspin']
            },
            'drop_on_the_one': {
                'category': 'cut',
                'energy_change': 'sudden',
                'complexity': 0.2,
                'boldness': 0.8,
                'compatible_with': ['quick_cut', 'double_drop'],
                'incompatible_with': ['long_blend', 'echo_out']
            },
            'back_and_forth': {
                'category': 'timing',
                'energy_change': 'stable',
                'complexity': 0.6,
                'boldness': 0.5,
                'compatible_with': ['long_blend', 'phrase_match', 'vocal_layering'],
                'incompatible_with': ['quick_cut']
            },
            'drum_roll': {
                'category': 'energy',
                'energy_change': 'building',
                'complexity': 0.5,
                'boldness': 0.7,
                'compatible_with': ['filter_sweep', 'quick_cut', 'drop_mix'],
                'incompatible_with': ['long_blend']
            },
            'thematic_handoff': {
                'category': 'timing',
                'energy_change': 'stable',
                'complexity': 0.3,
                'boldness': 0.3,
                'compatible_with': ['phrase_match', 'long_blend', 'vocal_layering'],
                'incompatible_with': []
            }
        }
        
        # Pre-defined creative hybrid techniques
        self.hybrid_presets = {
            'cinematic_blend': {
                'techniques': ['filter_sweep', 'echo_out', 'long_blend'],
                'weights': [0.4, 0.3, 0.3],
                'description': 'Epic, movie-trailer style transition'
            },
            'festival_drop': {
                'techniques': ['energy_build', 'drop_mix', 'double_drop'],
                'weights': [0.3, 0.3, 0.4],
                'description': 'Maximum energy festival moment'
            },
            'smooth_operator': {
                'techniques': ['bass_swap', 'staggered_stem_mix', 'phrase_match'],
                'weights': [0.35, 0.35, 0.3],
                'description': 'Ultra-smooth professional blend'
            },
            'creative_chaos': {
                'techniques': ['backspin', 'filter_sweep', 'quick_cut'],
                'weights': [0.3, 0.4, 0.3],
                'description': 'Unexpected, creative transition'
            },
            'deep_immersion': {
                'techniques': ['long_blend', 'vocal_layering', 'bass_swap'],
                'weights': [0.35, 0.35, 0.3],
                'description': 'Deep, hypnotic transition'
            },
            'vocal_safe_handoff': {
                'techniques': ['phrase_match', 'staggered_stem_mix', 'echo_out'],
                'weights': [0.45, 0.35, 0.2],
                'description': 'Phrase-aware handoff with reduced vocal collision risk'
            },
            'low_end_guard': {
                'techniques': ['bass_swap', 'long_blend', 'phrase_match'],
                'weights': [0.4, 0.35, 0.25],
                'description': 'Protects low-end clarity while preserving phrase continuity'
            },
            'percussive_bridge': {
                'techniques': ['drum_roll', 'filter_sweep', 'bass_swap'],
                'weights': [0.4, 0.35, 0.25],
                'description': 'Percussive build then controlled spectral handoff'
            }
        }
    
    # ==================== TECHNIQUE COMPATIBILITY ====================
    
    def analyze_technique_compatibility(self, 
                                        tech_a: str, 
                                        tech_b: str) -> Dict:
        """
        Analyze compatibility between two techniques for blending.
        """
        profile_a = self.technique_profiles.get(tech_a, {})
        profile_b = self.technique_profiles.get(tech_b, {})
        
        if not profile_a or not profile_b:
            return {'compatible': False, 'score': 0.0, 'reason': 'Unknown technique'}
        
        # Check explicit incompatibility
        if tech_b in profile_a.get('incompatible_with', []):
            return {
                'compatible': False,
                'score': 0.0,
                'reason': f'{tech_a} is incompatible with {tech_b}'
            }
        
        # Check explicit compatibility
        if tech_b in profile_a.get('compatible_with', []):
            return {
                'compatible': True,
                'score': 0.9,
                'reason': f'{tech_a} and {tech_b} work well together'
            }
        
        # Calculate compatibility from profiles
        # Same category bonus
        category_match = profile_a.get('category') == profile_b.get('category')
        
        # Energy compatibility
        energy_a = profile_a.get('energy_change', 'stable')
        energy_b = profile_b.get('energy_change', 'stable')
        energy_compat = self._calculate_energy_compatibility(energy_a, energy_b)
        
        # Complexity balance
        complexity_diff = abs(
            profile_a.get('complexity', 0.5) - profile_b.get('complexity', 0.5)
        )
        complexity_compat = 1.0 - complexity_diff
        
        # Overall score
        score = 0.3 * (1.0 if category_match else 0.5) + \
                0.4 * energy_compat + \
                0.3 * complexity_compat
        
        return {
            'compatible': score > 0.4,
            'score': float(score),
            'category_match': category_match,
            'energy_compatibility': float(energy_compat),
            'reason': 'Calculated compatibility'
        }
    
    def _calculate_energy_compatibility(self, energy_a: str, energy_b: str) -> float:
        """Calculate energy change compatibility."""
        energy_pairs = {
            ('stable', 'stable'): 0.9,
            ('stable', 'gradual'): 0.8,
            ('gradual', 'gradual'): 0.9,
            ('building', 'peak'): 0.9,
            ('dip_then_build', 'peak'): 0.8,
            ('fading', 'stable'): 0.7,
            ('fading', 'building'): 0.5,
            ('sudden', 'stable'): 0.4,
            ('dramatic', 'building'): 0.6
        }
        
        key = (energy_a, energy_b)
        if key in energy_pairs:
            return energy_pairs[key]
        
        # Reverse check
        key_rev = (energy_b, energy_a)
        if key_rev in energy_pairs:
            return energy_pairs[key_rev]
        
        return 0.5  # Default neutral
    
    # ==================== HYBRID TECHNIQUE CREATION ====================
    
    def create_hybrid_technique(self,
                                techniques: List[str],
                                weights: Optional[List[float]] = None,
                                context: Optional[Dict] = None) -> Dict:
        """
        Create a hybrid technique from multiple techniques.
        
        Args:
            techniques: List of technique names to blend (2-3 recommended)
            weights: Optional weights for each technique (sums to 1.0)
            context: Optional context (energy level, genre, etc.)
        
        Returns:
            Hybrid technique specification
        """
        if len(techniques) < 2:
            return {'error': 'Need at least 2 techniques to blend'}
        
        if len(techniques) > 4:
            techniques = techniques[:4]  # Max 4 techniques
        
        # Validate compatibility
        compatibility_scores = []
        for i, tech_a in enumerate(techniques):
            for tech_b in techniques[i+1:]:
                compat = self.analyze_technique_compatibility(tech_a, tech_b)
                compatibility_scores.append(compat['score'])
        
        avg_compatibility = np.mean(compatibility_scores)
        
        if avg_compatibility < 0.3:
            return {
                'error': 'Techniques are not compatible for blending',
                'compatibility': float(avg_compatibility),
                'techniques': techniques
            }
        
        # Assign weights if not provided
        if weights is None:
            weights = self._calculate_optimal_weights(techniques, context)
        
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
        # Generate hybrid parameters
        hybrid_params = self._merge_technique_parameters(techniques, weights)
        
        # Create hybrid name
        hybrid_name = self._generate_hybrid_name(techniques, weights)
        
        # Calculate overall characteristics
        overall_complexity = sum(
            self.technique_profiles.get(t, {}).get('complexity', 0.5) * w
            for t, w in zip(techniques, weights)
        )
        
        overall_boldness = sum(
            self.technique_profiles.get(t, {}).get('boldness', 0.5) * w
            for t, w in zip(techniques, weights)
        )
        
        return {
            'name': hybrid_name,
            'techniques': techniques,
            'weights': weights,
            'params': hybrid_params,
            'compatibility': float(avg_compatibility),
            'complexity': float(overall_complexity),
            'boldness': float(overall_boldness),
            'description': self._describe_hybrid(techniques, weights)
        }
    
    def _calculate_optimal_weights(self, 
                                   techniques: List[str],
                                   context: Optional[Dict] = None) -> List[float]:
        """Calculate optimal weights based on technique compatibility and context."""
        n = len(techniques)
        weights = [1.0 / n] * n  # Start equal
        
        if context is None:
            return weights
        
        # Adjust based on context
        energy_direction = context.get('energy_direction', 'stable')
        harmonic_compatibility = context.get('harmonic_compatibility', 0.5)
        
        for i, tech in enumerate(techniques):
            profile = self.technique_profiles.get(tech, {})
            
            # Boost techniques that match energy direction
            if profile.get('energy_change') == energy_direction:
                weights[i] *= 1.3
            
            # Boost smooth techniques for harmonically compatible songs
            if harmonic_compatibility > 0.7 and profile.get('complexity', 0.5) < 0.5:
                weights[i] *= 1.2
            
            # Boost bold techniques for low compatibility
            if harmonic_compatibility < 0.4 and profile.get('boldness', 0.5) > 0.6:
                weights[i] *= 1.2
        
        return weights
    
    def _merge_technique_parameters(self, 
                                    techniques: List[str],
                                    weights: List[float]) -> Dict:
        """Merge parameters from multiple techniques with weighted blending."""
        merged = {}
        
        # Default parameters for each technique
        technique_defaults = {
            'long_blend': {'curve_shape': 'smooth', 'blend_ratio': 1.0},
            'bass_swap': {'swap_point_ratio': 0.5, 'swap_intensity': 1.0},
            'filter_sweep': {'filter_type': 'high_pass', 'sweep_range': (200, 8000)},
            'echo_out': {'delay_ms': 500, 'feedback': 0.4, 'wet_mix': 0.6},
            'drop_mix': {'dip_ratio': 0.3, 'dip_depth': 0.6},
            'energy_build': {'build_intensity': 1.0, 'filter_sweep': True},
            'staggered_stem_mix': {'beat_lead_ratio': 0.2, 'vocal_delay_ratio': 0.3},
            'vocal_layering': {'vocal_hold_ratio': 0.6},
            'double_drop': {'sync_point_ratio': 0.5, 'energy_boost': 1.2},
            'quick_cut': {'fade_ms': 100},
            'backspin': {'spin_duration_ratio': 0.6},
            'loop_transition': {'loop_length_bars': 4, 'loop_repeats': 4},
            'drop_on_the_one': {'fade_ms': 50},
            'back_and_forth': {'switch_interval_bars': 8, 'num_switches': 2},
            'drum_roll': {'roll_duration_ratio': 0.5},
            'thematic_handoff': {'phrase_length_bars': 16}
        }
        
        # Blend parameters
        for tech, weight in zip(techniques, weights):
            defaults = technique_defaults.get(tech, {})
            for key, value in defaults.items():
                if isinstance(value, (int, float)):
                    if key not in merged:
                        merged[key] = 0.0
                    merged[key] += value * weight
                elif isinstance(value, bool):
                    if key not in merged:
                        merged[key] = 0.0
                    merged[key] += (1.0 if value else 0.0) * weight
                else:
                    # For non-numeric, use highest-weighted technique
                    if key not in merged or weight > weights[techniques.index(merged.get(key + '_source', tech))]:
                        merged[key] = value
                        merged[key + '_source'] = tech
        
        # Convert boolean accumulations back
        for key in list(merged.keys()):
            if key.endswith('_source'):
                del merged[key]
            elif key in ['filter_sweep']:
                merged[key] = merged[key] > 0.5
        
        # Add technique list and weights
        merged['_techniques'] = techniques
        merged['_weights'] = weights
        
        return merged
    
    def _generate_hybrid_name(self, techniques: List[str], weights: List[float]) -> str:
        """Generate a creative name for the hybrid technique."""
        # Sort by weight
        sorted_pairs = sorted(zip(techniques, weights), key=lambda x: -x[1])
        dominant = sorted_pairs[0][0]
        
        prefixes = {
            'long_blend': 'Smooth',
            'quick_cut': 'Sharp',
            'bass_swap': 'Deep',
            'filter_sweep': 'Sweeping',
            'echo_out': 'Ethereal',
            'drop_mix': 'Dynamic',
            'energy_build': 'Rising',
            'staggered_stem_mix': 'Layered',
            'vocal_layering': 'Vocal',
            'double_drop': 'Epic',
            'backspin': 'Spinning',
            'loop_transition': 'Looping',
            'drop_on_the_one': 'Drop',
            'back_and_forth': 'Switch',
            'drum_roll': 'Rolling',
            'thematic_handoff': 'Thematic'
        }
        
        suffixes = {
            'long_blend': 'Blend',
            'quick_cut': 'Cut',
            'bass_swap': 'Swap',
            'filter_sweep': 'Sweep',
            'echo_out': 'Echo',
            'drop_mix': 'Drop',
            'energy_build': 'Build',
            'staggered_stem_mix': 'Layer',
            'vocal_layering': 'Blend',
            'double_drop': 'Drop',
            'backspin': 'Spin',
            'loop_transition': 'Loop',
            'drop_on_the_one': 'One',
            'back_and_forth': 'Switch',
            'drum_roll': 'Roll',
            'thematic_handoff': 'Handoff'
        }
        
        prefix = prefixes.get(dominant, 'Hybrid')
        
        # Second technique for suffix
        if len(sorted_pairs) > 1:
            secondary = sorted_pairs[1][0]
            suffix = suffixes.get(secondary, 'Mix')
        else:
            suffix = 'Mix'
        
        return f"{prefix} {suffix}"
    
    def _describe_hybrid(self, techniques: List[str], weights: List[float]) -> str:
        """Generate a human-readable description of the hybrid."""
        sorted_pairs = sorted(zip(techniques, weights), key=lambda x: -x[1])
        
        descriptions = []
        for tech, weight in sorted_pairs:
            pct = int(weight * 100)
            descriptions.append(f"{tech.replace('_', ' ')} ({pct}%)")
        
        return "Hybrid blend of: " + " + ".join(descriptions)
    
    # ==================== HYBRID EXECUTION ====================
    
    def execute_hybrid(self,
                       seg_a: np.ndarray,
                       seg_b: np.ndarray,
                       hybrid: Dict,
                       seg_a_stems: Optional[Dict] = None,
                       seg_b_stems: Optional[Dict] = None,
                       technique_executor = None) -> np.ndarray:
        """
        Execute a hybrid technique by blending multiple technique outputs.
        
        Args:
            seg_a: Audio segment from outgoing song
            seg_b: Audio segment from incoming song
            hybrid: Hybrid technique specification from create_hybrid_technique
            seg_a_stems: Optional stems
            seg_b_stems: Optional stems
            technique_executor: TechniqueExecutor instance
        
        Returns:
            Mixed audio
        """
        if technique_executor is None:
            # Import here to avoid circular dependency
            from src.technique_executor import TechniqueExecutor
            technique_executor = TechniqueExecutor(sr=self.sr)
        
        techniques = hybrid.get('techniques', [])
        weights = hybrid.get('weights', [])
        params = hybrid.get('params', {})
        
        if len(techniques) == 0:
            # Fallback to long blend
            return technique_executor.execute(
                'long_blend', seg_a, seg_b, {},
                seg_a_stems, seg_b_stems
            )
        
        # Execute each technique
        outputs = []
        for tech, weight in zip(techniques, weights):
            try:
                output = technique_executor.execute(
                    tech, seg_a.copy(), seg_b.copy(), params,
                    seg_a_stems, seg_b_stems
                )
                outputs.append((output, weight))
            except Exception as e:
                print(f"  ⚠ Technique {tech} failed: {e}")
                continue
        
        if len(outputs) == 0:
            # All failed, fallback
            return technique_executor.execute(
                'long_blend', seg_a, seg_b, {},
                seg_a_stems, seg_b_stems
            )
        
        # Blend outputs by weight
        n_samples = min(len(o[0]) for o in outputs)
        blended = np.zeros_like(outputs[0][0][:n_samples])
        
        total_weight = sum(w for _, w in outputs)
        for output, weight in outputs:
            normalized_weight = weight / total_weight
            blended += output[:n_samples] * normalized_weight
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(blended))
        if max_val > 0.95:
            blended = blended * (0.95 / max_val)
        
        return blended
    
    # ==================== CONTEXT-AWARE CREATIVITY ====================
    
    def suggest_creative_technique(self,
                                   context: Dict,
                                   creativity_level: float = 0.5) -> Dict:
        """
        Suggest a creative technique based on context.
        Uses presets often for variety and avoids always picking the same combo
        (e.g. echo_out + filter_sweep + staggered_stem_mix).
        
        Args:
            context: Dict with keys like:
                - harmonic_compatibility: 0-1
                - energy_a, energy_b: 0-1
                - tempo_diff: BPM difference
                - has_vocals_a, has_vocals_b: bool
                - section_a, section_b: str (intro, verse, chorus, etc.)
            creativity_level: 0.0 (conservative) to 1.0 (experimental)
        
        Returns:
            Suggested hybrid technique
        """
        harmonic = context.get('harmonic_compatibility', 0.5)
        energy_a = context.get('energy_a', 0.5)
        energy_b = context.get('energy_b', 0.5)
        tempo_diff = context.get('tempo_diff', 0)
        has_vocals_a = context.get('has_vocals_a', False)
        has_vocals_b = context.get('has_vocals_b', False)
        vocal_overlap_risk = float(context.get('vocal_overlap_risk', 0.5))
        section_a = context.get('section_a', '')
        section_b = context.get('section_b', '')
        
        # Determine energy direction
        if energy_b > energy_a * 1.2:
            energy_direction = 'building'
        elif energy_a > energy_b * 1.2:
            energy_direction = 'fading'
        else:
            energy_direction = 'stable'
        
        # Often pick a preset for real variety (not just crossfade-heavy combos)
        preset_names = list(self.hybrid_presets.keys())
        use_preset_prob = 0.35 + creativity_level * 0.25  # 0.35–0.6
        high_vocal_risk = vocal_overlap_risk >= 0.65
        if high_vocal_risk:
            # Prefer deterministic vocal-safe combinations when overlap risk is high.
            use_preset_prob = max(0.2, use_preset_prob - 0.2)
        if random.random() < use_preset_prob:
            # Slight context bias: building favors festival/percussive, stable favors smooth/vocal-safe.
            weights_by_preset = {name: 1.0 for name in preset_names}
            if energy_direction == 'building':
                for name, weight in {
                    'festival_drop': 2.2,
                    'percussive_bridge': 1.8,
                    'creative_chaos': 1.2
                }.items():
                    if name in weights_by_preset:
                        weights_by_preset[name] = weight
            elif energy_direction == 'fading':
                for name, weight in {
                    'cinematic_blend': 2.0,
                    'deep_immersion': 1.5,
                    'vocal_safe_handoff': 1.3
                }.items():
                    if name in weights_by_preset:
                        weights_by_preset[name] = weight
            else:
                for name, weight in {
                    'smooth_operator': 2.1,
                    'vocal_safe_handoff': 1.8,
                    'low_end_guard': 1.6
                }.items():
                    if name in weights_by_preset:
                        weights_by_preset[name] = weight
            if high_vocal_risk:
                for risky in ('creative_chaos', 'deep_immersion'):
                    if risky in weights_by_preset:
                        weights_by_preset[risky] *= 0.5
                for safer in ('vocal_safe_handoff', 'low_end_guard', 'smooth_operator'):
                    if safer in weights_by_preset:
                        weights_by_preset[safer] *= 1.35
            weights_preset = [weights_by_preset[name] for name in preset_names]
            preset_name = random.choices(preset_names, weights=weights_preset, k=1)[0]
            hybrid = self.get_preset_hybrid(preset_name)
            if hybrid is not None:
                hybrid['name'] = preset_name.replace('_', ' ').title()
                return hybrid
        
        # Rule-based but varied: randomize which techniques we add (avoid same trio every time)
        if high_vocal_risk:
            base_options = {
                'building': ['phrase_match', 'staggered_stem_mix', 'bass_swap', 'filter_sweep', 'energy_build'],
                'fading': ['echo_out', 'filter_sweep', 'phrase_match', 'bass_swap'],
                'stable': ['phrase_match', 'staggered_stem_mix', 'long_blend', 'bass_swap', 'loop_transition', 'thematic_handoff']
            }
            secondary_pool = ['phrase_match', 'staggered_stem_mix', 'bass_swap', 'filter_sweep', 'echo_out', 'loop_transition']
            bold_options = ['drum_roll', 'drop_on_the_one', 'quick_cut']
        else:
            base_options = {
                'building': ['energy_build', 'drop_mix', 'double_drop', 'filter_sweep', 'drum_roll'],
                'fading': ['echo_out', 'filter_sweep', 'drop_mix', 'bass_swap'],
                'stable': ['long_blend', 'bass_swap', 'staggered_stem_mix', 'phrase_match', 'vocal_layering', 'loop_transition', 'back_and_forth', 'thematic_handoff']
            }
            secondary_pool = ['bass_swap', 'filter_sweep', 'staggered_stem_mix', 'vocal_layering', 'phrase_match', 'drop_mix', 'energy_build', 'loop_transition']
            bold_options = ['quick_cut', 'backspin', 'double_drop', 'drop_mix', 'drop_on_the_one', 'drum_roll']
        
        chosen_techniques = []
        pool = base_options.get(energy_direction, base_options['stable'])
        # Pick 1–2 from base pool (random order)
        random.shuffle(pool)
        for t in pool:
            if t not in chosen_techniques:
                chosen_techniques.append(t)
                if len(chosen_techniques) >= 2:
                    break
        if len(chosen_techniques) < 2:
            chosen_techniques.append('long_blend' if 'long_blend' not in chosen_techniques else pool[0])
        
        # Add one context-aware secondary (varied, not always filter_sweep + staggered_stem_mix)
        if has_vocals_a and has_vocals_b and random.random() < 0.6:
            if high_vocal_risk:
                add = random.choice(['staggered_stem_mix', 'phrase_match', 'bass_swap'])
            else:
                add = random.choice(['staggered_stem_mix', 'vocal_layering', 'phrase_match'])
        elif harmonic < 0.6 and random.random() < 0.5:
            add = 'bass_swap'
        elif tempo_diff > 3 and random.random() < 0.5:
            add = random.choice(['filter_sweep', 'energy_build'])
        else:
            add = random.choice(secondary_pool)
        if add not in chosen_techniques:
            chosen_techniques.append(add)

        if vocal_overlap_risk >= 0.75 and 'phrase_match' not in chosen_techniques:
            if len(chosen_techniques) >= 3:
                chosen_techniques[-1] = 'phrase_match'
            else:
                chosen_techniques.append('phrase_match')
        
        # Sometimes add a bold technique for variety (not just safe crossfade)
        bold_prob = 0.15 if high_vocal_risk else 0.4
        if creativity_level > 0.4 and random.random() < bold_prob:
            bold = random.choice(bold_options)
            if bold not in chosen_techniques:
                chosen_techniques.append(bold)
        
        chosen_techniques = chosen_techniques[:3]
        
        return self.create_hybrid_technique(
            chosen_techniques,
            context={
                'energy_direction': energy_direction,
                'harmonic_compatibility': harmonic
            }
        )
    
    def get_preset_hybrid(self, preset_name: str) -> Optional[Dict]:
        """Get a pre-defined hybrid technique preset."""
        preset = self.hybrid_presets.get(preset_name)
        if preset is None:
            return None
        
        return self.create_hybrid_technique(
            preset['techniques'],
            preset['weights']
        )
    
    def list_presets(self) -> List[Dict]:
        """List all available hybrid presets."""
        return [
            {
                'name': name,
                'description': preset['description'],
                'techniques': preset['techniques']
            }
            for name, preset in self.hybrid_presets.items()
        ]
    
    # ==================== DYNAMIC MORPHING ====================
    
    def create_technique_morph(self,
                               start_technique: str,
                               end_technique: str,
                               transition_samples: int,
                               morph_stages: int = 10) -> Dict:
        """
        Create a technique morph that transitions between two techniques.
        
        Instead of using one technique throughout, this morphs
        from start_technique to end_technique during the transition.
        """
        compatibility = self.analyze_technique_compatibility(
            start_technique, end_technique
        )
        
        if not compatibility['compatible']:
            # Find intermediate technique
            intermediate = self._find_intermediate_technique(
                start_technique, end_technique
            )
            if intermediate:
                # Create two-stage morph
                morph_path = [start_technique, intermediate, end_technique]
            else:
                morph_path = [start_technique, end_technique]
        else:
            morph_path = [start_technique, end_technique]
        
        # Create weight curves for each technique
        n_techniques = len(morph_path)
        stage_samples = transition_samples // morph_stages
        
        weight_curves = {tech: np.zeros(transition_samples) for tech in morph_path}
        
        for stage in range(morph_stages):
            start_sample = stage * stage_samples
            end_sample = min((stage + 1) * stage_samples, transition_samples)
            
            progress = stage / morph_stages
            
            # Calculate weights at this stage
            weights = self._calculate_morph_weights(morph_path, progress)
            
            for tech, weight in weights.items():
                weight_curves[tech][start_sample:end_sample] = weight
        
        # Smooth the curves
        for tech in weight_curves:
            weight_curves[tech] = gaussian_filter1d(weight_curves[tech], sigma=stage_samples/4)
        
        return {
            'morph_path': morph_path,
            'weight_curves': {k: v.tolist() for k, v in weight_curves.items()},
            'morph_stages': morph_stages,
            'compatibility': compatibility['score']
        }
    
    def _find_intermediate_technique(self, tech_a: str, tech_b: str) -> Optional[str]:
        """Find a technique that bridges two incompatible techniques."""
        profile_a = self.technique_profiles.get(tech_a, {})
        profile_b = self.technique_profiles.get(tech_b, {})
        
        compatible_a = set(profile_a.get('compatible_with', []))
        compatible_b = set(profile_b.get('compatible_with', []))
        
        # Find intersection
        bridges = compatible_a & compatible_b
        
        if bridges:
            # Return the one with lowest complexity (smoothest bridge)
            return min(bridges, key=lambda t: 
                self.technique_profiles.get(t, {}).get('complexity', 1.0)
            )
        
        return None
    
    def _calculate_morph_weights(self, 
                                 morph_path: List[str],
                                 progress: float) -> Dict[str, float]:
        """Calculate technique weights at a morph progress point."""
        n = len(morph_path)
        weights = {}
        
        if n == 2:
            # Simple linear morph
            weights[morph_path[0]] = 1.0 - progress
            weights[morph_path[1]] = progress
        elif n == 3:
            # Three-stage morph with overlap
            if progress < 0.5:
                # First half: transition from first to middle
                local_progress = progress * 2
                weights[morph_path[0]] = 1.0 - local_progress
                weights[morph_path[1]] = local_progress
                weights[morph_path[2]] = 0.0
            else:
                # Second half: transition from middle to last
                local_progress = (progress - 0.5) * 2
                weights[morph_path[0]] = 0.0
                weights[morph_path[1]] = 1.0 - local_progress
                weights[morph_path[2]] = local_progress
        else:
            # More techniques: divide evenly
            segment = 1.0 / (n - 1)
            segment_idx = int(progress / segment)
            segment_idx = min(segment_idx, n - 2)
            local_progress = (progress - segment_idx * segment) / segment
            
            for i, tech in enumerate(morph_path):
                if i == segment_idx:
                    weights[tech] = 1.0 - local_progress
                elif i == segment_idx + 1:
                    weights[tech] = local_progress
                else:
                    weights[tech] = 0.0
        
        return weights
