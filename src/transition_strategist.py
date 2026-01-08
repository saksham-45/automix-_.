"""
Transition Strategy Selection Module

Context-aware transition technique selection based on:
- Harmonic compatibility
- Musical structure
- Energy flow
- Genre/style
"""
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.harmonic_analyzer import HarmonicAnalyzer
from src.structure_analyzer import StructureAnalyzer


@dataclass
class TransitionTechnique:
    """Represents a transition technique."""
    name: str
    duration_bars: int
    description: str
    energy_direction: str  # 'up', 'down', 'maintain'
    harmonic_requirements: str  # 'compatible', 'any'
    structure_preference: List[str]  # Preferred sections


class TransitionStrategist:
    """
    Selects optimal transition techniques based on context.
    """
    
    TECHNIQUES = {
        'long_blend': TransitionTechnique(
            name='long_blend',
            duration_bars=32,
            description='Smooth, gradual 32+ bar blend',
            energy_direction='maintain',
            harmonic_requirements='compatible',
            structure_preference=['verse', 'chorus']
        ),
        'quick_cut': TransitionTechnique(
            name='quick_cut',
            duration_bars=4,
            description='Fast, energetic 4-8 bar cut',
            energy_direction='up',
            harmonic_requirements='any',
            structure_preference=['drop', 'chorus']
        ),
        'bass_swap': TransitionTechnique(
            name='bass_swap',
            duration_bars=16,
            description='Bass frequency swap',
            energy_direction='maintain',
            harmonic_requirements='any',
            structure_preference=['verse', 'chorus']
        ),
        'filter_sweep': TransitionTechnique(
            name='filter_sweep',
            duration_bars=16,
            description='High/low pass filter sweep',
            energy_direction='down',
            harmonic_requirements='any',
            structure_preference=['outro', 'breakdown']
        ),
        'echo_out': TransitionTechnique(
            name='echo_out',
            duration_bars=8,
            description='Echo/delay exit effect',
            energy_direction='down',
            harmonic_requirements='any',
            structure_preference=['outro', 'verse']
        ),
        'drop_mix': TransitionTechnique(
            name='drop_mix',
            duration_bars=8,
            description='Energy drop before transition',
            energy_direction='down',
            harmonic_requirements='compatible',
            structure_preference=['chorus', 'breakdown']
        )
    }
    
    def __init__(self):
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.structure_analyzer = StructureAnalyzer()
    
    def select_technique(self,
                        key_a: str,
                        key_b: str,
                        tempo_a: float,
                        tempo_b: float,
                        section_a: str,
                        section_b: str,
                        energy_a: float,
                        energy_b: float,
                        clash_score: float) -> Dict:
        """
        Select optimal transition technique based on context.
        
        Returns:
            Dict with selected technique and parameters
        """
        # Check harmonic compatibility
        harmonic_score = self.harmonic_analyzer.score_transition_harmonics(
            key_a, key_b, tempo_a, tempo_b
        )
        keys_compatible = harmonic_score['key_compatibility']['compatible']
        
        # Determine energy flow
        energy_diff = energy_b - energy_a
        if energy_diff > 0.2:
            energy_direction = 'up'
        elif energy_diff < -0.2:
            energy_direction = 'down'
        else:
            energy_direction = 'maintain'
        
        # Score each technique
        technique_scores = {}
        
        for tech_name, tech in self.TECHNIQUES.items():
            score = 0.0
            
            # Harmonic compatibility
            if tech.harmonic_requirements == 'compatible' and keys_compatible:
                score += 0.3
            elif tech.harmonic_requirements == 'any':
                score += 0.2
            
            # Energy direction match
            if tech.energy_direction == energy_direction:
                score += 0.3
            elif tech.energy_direction == 'maintain' and abs(energy_diff) < 0.2:
                score += 0.3
            
            # Structure preference
            if section_a in tech.structure_preference or section_b in tech.structure_preference:
                score += 0.2
            
            # Frequency clash considerations
            if clash_score > 0.5 and tech_name == 'bass_swap':
                score += 0.3
            elif clash_score < 0.3 and tech_name == 'long_blend':
                score += 0.2
            
            technique_scores[tech_name] = score
        
        # Select best technique
        best_technique = max(technique_scores, key=technique_scores.get)
        best_score = technique_scores[best_technique]
        
        tech = self.TECHNIQUES[best_technique]
        
        # Calculate duration
        avg_tempo = (tempo_a + tempo_b) / 2
        bar_duration = 4 * (60 / avg_tempo)  # 4 beats per bar
        duration_sec = tech.duration_bars * bar_duration
        
        return {
            'technique': best_technique,
            'technique_name': tech.name,
            'duration_bars': tech.duration_bars,
            'duration_sec': duration_sec,
            'confidence': float(best_score),
            'energy_direction': energy_direction,
            'harmonic_compatible': keys_compatible,
            'technique_scores': technique_scores
        }
    
    def get_technique_parameters(self, technique: str, context: Dict) -> Dict:
        """
        Get specific parameters for a transition technique.
        
        Returns:
            Dict with technique-specific parameters
        """
        if technique not in self.TECHNIQUES:
            technique = 'long_blend'  # Default
        
        tech = self.TECHNIQUES[technique]
        
        params = {
            'duration_bars': tech.duration_bars,
            'duration_sec': context.get('duration_sec', tech.duration_bars * 2)
        }
        
        if technique == 'bass_swap':
            params['swap_point_ratio'] = 0.5  # Swap halfway through
            params['bass_cut_db'] = -12  # Cut outgoing bass by 12dB
            params['bass_boost_db'] = 6  # Boost incoming bass by 6dB
        
        elif technique == 'filter_sweep':
            params['filter_type'] = 'high_pass'  # or 'low_pass'
            params['filter_start_hz'] = 20
            params['filter_end_hz'] = 10000
            params['resonance'] = 0.7
        
        elif technique == 'echo_out':
            params['delay_time_ms'] = 500
            params['feedback'] = 0.4
            params['wet_mix'] = 0.6
        
        elif technique == 'long_blend':
            params['crossfade_type'] = 'equal_power'
            params['curve_shape'] = 'smooth'
        
        elif technique == 'quick_cut':
            params['cut_point'] = 'downbeat'
            params['fade_out_ms'] = 100
            params['fade_in_ms'] = 100
        
        return params

