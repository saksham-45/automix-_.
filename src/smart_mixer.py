"""
Smart Mixer - Human-Level Mixing System

Uses all advanced modules for perceptually-correct, smooth transitions.
"""
import numpy as np
import soundfile as sf
import librosa
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


class SmartMixer:
    """
    Creates human-level smooth transitions using all advanced analysis modules.
    """
    
    def __init__(self, sr: int = 44100, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length
        
        # Initialize all modules
        self.transition_finder = SmartTransitionFinder(sr=sr, hop_length=hop_length)
        self.beat_aligner = BeatAligner(sr=sr, hop_length=hop_length)
        self.psychoacoustics = PsychoacousticAnalyzer(sr=sr)
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.structure_analyzer = StructureAnalyzer(sr=sr, hop_length=hop_length)
        self.advanced_beatmatcher = AdvancedBeatMatcher(sr=sr, hop_length=hop_length)
        self.dynamic_processor = DynamicProcessor(sr=sr)
        self.transition_strategist = TransitionStrategist()
        self.crossfade_engine = CrossfadeEngine(sr=sr)
        self.quality_assessor = QualityAssessor(sr=sr)
    
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
        y_a, sr_a = librosa.load(song_a_path, sr=self.sr)
        y_b, sr_b = librosa.load(song_b_path, sr=self.sr)
        
        # Analyze both songs comprehensively (lightweight version for speed)
        print("\n[1/8] Analyzing Song A...")
        analysis_a = self._analyze_song_fast(y_a, song_a_analysis)
        
        print("[2/8] Analyzing Song B...")
        analysis_b = self._analyze_song_fast(y_b, song_b_analysis)
        
        # Find optimal transition points using structure and harmonics
        print("[3/8] Finding optimal transition points...")
        transition_pair = self._find_optimal_transition_points(
            y_a, y_b, analysis_a, analysis_b
        )
        
        # Advanced beat matching
        print("[4/8] Beat matching and phase alignment...")
        beat_match = self.advanced_beatmatcher.match_beats(
            y_a, y_b,
            transition_pair.song_a_point.time_sec,
            transition_pair.song_b_point.time_sec
        )
        
        aligned_a = beat_match['aligned_point_a_sec']
        aligned_b = beat_match['aligned_point_b_sec']
        
        # Select transition technique
        print("[5/8] Selecting transition technique...")
        clash_analysis = self.psychoacoustics.predict_frequency_clash(y_a, y_b)
        technique = self.transition_strategist.select_technique(
            analysis_a['key'],
            analysis_b['key'],
            analysis_a['tempo'],
            analysis_b['tempo'],
            transition_pair.song_a_point.structural_label,
            transition_pair.song_b_point.structural_label,
            transition_pair.song_a_point.energy,
            transition_pair.song_b_point.energy,
            clash_analysis['clash_score']
        )
        
        # Determine transition duration
        if transition_duration is None:
            transition_duration = technique['duration_sec']
        
        print(f"  Technique: {technique['technique_name']}")
        print(f"  Duration: {transition_duration:.1f}s ({technique['duration_bars']} bars)")
        
        # Extract segments
        print("[6/8] Extracting and processing segments...")
        seg_a, seg_b = self._extract_segments(y_a, y_b, aligned_a, aligned_b, transition_duration)
        
        # Apply dynamic EQ if needed
        if clash_analysis['clash_score'] > 0.3:
            print("  Applying dynamic EQ to prevent frequency clashes...")
            eq_analysis = self.dynamic_processor.analyze_frequency_clash(seg_a, seg_b)
            if 'bass_swap' in eq_analysis['recommendations']:
                seg_a, seg_b = self._apply_bass_swap(seg_a, seg_b, transition_duration)
        
        # Create perceptual crossfade curves
        print("[7/8] Creating perceptual crossfade curves...")
        vol_a, vol_b = self.crossfade_engine.create_lufs_matched_curves(
            seg_a, seg_b, len(seg_a)
        )
        
        # Apply adaptive crossfade
        vol_a, vol_b = self.crossfade_engine.create_adaptive_crossfade(
            seg_a, seg_b, len(seg_a), frequency_aware=True
        )
        
        # Mix
        print("[8/8] Mixing and quality assessment...")
        mixed = self.crossfade_engine.apply_crossfade(seg_a, seg_b, vol_a, vol_b)
        
        # Quality assessment
        quality = self.quality_assessor.assess_transition_quality(
            mixed, seg_a, seg_b,
            analysis_a['key'], analysis_b['key']
        )
        
        print(f"\n✓ Transition Quality: {quality['quality_rating'].upper()} ({quality['overall_score']:.2f})")
        print(f"  Smoothness: {quality['smoothness']['analysis']}")
        print(f"  Clarity: {quality['clarity']['analysis']}")
        print(f"  Harmonic tension: {quality['harmonic_tension']:.2f}")
        
        # Build final mix with context
        context_before = int(10 * self.sr)
        context_after = int(10 * self.sr)
        
        ctx_a_start = max(0, int(aligned_a * self.sr) - context_before)
        ctx_a = y_a[ctx_a_start:int(aligned_a * self.sr)]
        
        ctx_b_end = min(len(y_b), int(aligned_b * self.sr) + len(mixed) + context_after)
        ctx_b = y_b[int(aligned_b * self.sr) + len(mixed):ctx_b_end]
        
        final = np.concatenate([ctx_a, mixed, ctx_b], axis=0)
        
        return final
    
    def _analyze_song_fast(self, y: np.ndarray, existing_analysis: Optional[Dict] = None) -> Dict:
        """Fast song analysis - only what we need for transitions."""
        # Use sample of song for faster analysis (first 60 seconds)
        sample_length = min(60 * self.sr, len(y))
        y_sample = y[:sample_length] if len(y) > sample_length else y
        
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
        
        # Quick structure analysis (simplified)
        print("    Analyzing structure...")
        try:
            structure = self.structure_analyzer.analyze_structure(y_sample)
            # Scale structure times to full song if we used a sample
            if len(y) > sample_length:
                scale_factor = len(y) / sample_length
                for phrase in structure.get('phrases', []):
                    phrase['start_sec'] *= scale_factor
                    phrase['end_sec'] *= scale_factor
                for section in structure.get('sections', []):
                    section['start_sec'] *= scale_factor
                    section['end_sec'] *= scale_factor
                for i, point in enumerate(structure.get('best_mix_in_points', [])):
                    structure['best_mix_in_points'][i] = point * scale_factor
                for i, point in enumerate(structure.get('best_mix_out_points', [])):
                    structure['best_mix_out_points'][i] = point * scale_factor
        except Exception as e:
            print(f"    Warning: Structure analysis failed: {e}")
            # Fallback structure
            duration = len(y) / self.sr
            structure = {
                'best_mix_out_points': [max(0, duration - 30)],
                'best_mix_in_points': [min(30, duration * 0.1)],
                'sections': []
            }
        
        # Quick energy estimate
        energy = np.mean(np.abs(y_sample) ** 2)
        
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
        """Find optimal transition points using structure and harmonics."""
        # Use structure analysis to find best mix points
        structure_a = analysis_a['structure']
        structure_b = analysis_b['structure']
        
        # Get best mix-out points from A
        out_points = structure_a.get('best_mix_out_points', [])
        if not out_points:
            # Fallback: use last 30 seconds
            duration_a = len(y_a) / self.sr
            out_points = [max(0, duration_a - 30)]
        
        # Get best mix-in points from B
        in_points = structure_b.get('best_mix_in_points', [])
        if not in_points:
            # Fallback: use first 30 seconds
            duration_b = len(y_b) / self.sr
            in_points = [min(30, duration_b * 0.1)]
        
        # Score all combinations
        best_score = -1
        best_pair = None
        
        for point_a in out_points[:5]:  # Top 5 out points
            for point_b in in_points[:5]:  # Top 5 in points
                # Harmonic compatibility
                harmonic = self.harmonic_analyzer.score_transition_harmonics(
                    analysis_a['key'], analysis_b['key'],
                    analysis_a['tempo'], analysis_b['tempo']
                )
                
                # Structure compatibility
                # Find sections at these points
                section_a = self._get_section_at_time(point_a, structure_a)
                section_b = self._get_section_at_time(point_b, structure_b)
                
                # Score this pair
                score = (
                    harmonic['overall_score'] * 0.5 +
                    (1.0 if section_a in ['outro', 'verse'] else 0.7) * 0.25 +
                    (1.0 if section_b in ['intro', 'verse'] else 0.7) * 0.25
                )
                
                if score > best_score:
                    best_score = score
                    # Create transition points (simplified)
                    from src.smart_transition_finder import TransitionPoint
                    best_pair = TransitionPair(
                        song_a_point=TransitionPoint(
                            time_sec=point_a,
                            beat_aligned=True,
                            energy=0.5,
                            energy_trend='falling',
                            structural_label=section_a,
                            score=score,
                            beat_position=0
                        ),
                        song_b_point=TransitionPoint(
                            time_sec=point_b,
                            beat_aligned=True,
                            energy=0.5,
                            energy_trend='rising',
                            structural_label=section_b,
                            score=score,
                            beat_position=0
                        ),
                        compatibility_score=score,
                        tempo_match=True,
                        key_match=harmonic['key_compatibility']['compatible'],
                        beat_aligned=True
                    )
        
        if best_pair is None:
            # Fallback: create simple transition points
            from src.smart_transition_finder import TransitionPoint
            duration_a = len(y_a) / self.sr
            duration_b = len(y_b) / self.sr
            best_pair = TransitionPair(
                song_a_point=TransitionPoint(
                    time_sec=max(0, duration_a - 30),
                    beat_aligned=True,
                    energy=0.5,
                    energy_trend='falling',
                    structural_label='outro',
                    score=0.5,
                    beat_position=0
                ),
                song_b_point=TransitionPoint(
                    time_sec=min(30, duration_b * 0.1),
                    beat_aligned=True,
                    energy=0.5,
                    energy_trend='rising',
                    structural_label='intro',
                    score=0.5,
                    beat_position=0
                ),
                compatibility_score=0.5,
                tempo_match=True,
                key_match=True,
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
        """Extract transition segments."""
        n_samples = int(duration * self.sr)
        
        start_a = int(point_a * self.sr)
        end_a = min(start_a + n_samples, len(y_a))
        seg_a = y_a[start_a:end_a]
        
        start_b = int(point_b * self.sr)
        end_b = min(start_b + n_samples, len(y_b))
        seg_b = y_b[start_b:end_b]
        
        # Ensure same length
        min_len = min(len(seg_a), len(seg_b))
        seg_a = seg_a[:min_len]
        seg_b = seg_b[:min_len]
        
        # Pad if needed
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
        
        return seg_a, seg_b
    
    def _apply_bass_swap(self, seg_a: np.ndarray, seg_b: np.ndarray, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply bass swap technique."""
        automation = self.dynamic_processor.create_bass_swap_automation(duration)
        return self.dynamic_processor.apply_bass_swap(seg_a, seg_b, automation)
