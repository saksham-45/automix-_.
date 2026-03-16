"""
Monte Carlo Quality Optimizer

Predictive quality assurance that exceeds human DJ capabilities:
- Monte Carlo simulation (test 100+ transition variations)
- Multi-objective optimization (smoothness + creativity + surprise)
- Perceptual quality prediction before committing
- Ensemble evaluation for robust scoring

This module simulates many transition options and picks the best.
Human DJs can try 2-3 options. We try 100+.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import random


class MonteCarloQualityOptimizer:
    """
    Monte Carlo simulation for optimal transition selection.
    
    Instead of picking one transition and hoping it works,
    we simulate many variations and pick the best one.
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        
        # Quality dimensions and their weights
        self.quality_weights = {
            'smoothness': 0.25,
            'clarity': 0.20,
            'energy_flow': 0.20,
            'harmonic_quality': 0.15,
            'creativity': 0.10,
            'surprise': 0.10
        }
        
        # Parameter ranges for Monte Carlo exploration
        self.parameter_ranges = {
            'transition_duration': (8.0, 24.0),  # 8-24 seconds
            'bass_swap_ratio': (0.3, 0.7),  # Where to swap bass
            'filter_sweep_start': (100, 500),  # Hz
            'filter_sweep_end': (2000, 10000),  # Hz
            'echo_delay_ms': (200, 800),
            'energy_dip_ratio': (0.2, 0.5),
            'stem_stagger_offset': (0.1, 0.4),
            'morph_depth': (0.3, 0.9)  # Added for progressive_morph
        }
    
    # ==================== MONTE CARLO SIMULATION ====================
    
    def simulate_transitions(self,
                            seg_a: np.ndarray,
                            seg_b: np.ndarray,
                            technique: str,
                            base_params: Dict,
                            n_simulations: int = 50,
                            executor_fn: Optional[Callable] = None) -> Dict:
        """
        Run Monte Carlo simulation to find optimal transition parameters.
        
        Args:
            seg_a: Audio segment from outgoing song
            seg_b: Audio segment from incoming song
            technique: Transition technique to use
            base_params: Base parameters to vary
            n_simulations: Number of simulations to run
            executor_fn: Function to execute transition (technique_executor.execute)
        
        Returns:
            Best parameters and simulation results
        """
        if executor_fn is None:
            # Create default executor if not provided
            from src.technique_executor import TechniqueExecutor
            technique_executor = TechniqueExecutor(sr=self.sr)
            executor_fn = lambda t, a, b, p: technique_executor.execute(t, a, b, p)
        
        # Generate parameter variations
        variations = self._generate_variations(base_params, n_simulations)
        
        # Run simulations
        results = []
        best_score = -1
        best_params = base_params
        best_output = None
        
        for i, params in enumerate(variations):
            try:
                # Execute transition with these parameters
                output = executor_fn(technique, seg_a.copy(), seg_b.copy(), params)
                
                # Evaluate quality
                quality = self._evaluate_quality(output, seg_a, seg_b, params)
                
                results.append({
                    'params': params,
                    'quality': quality,
                    'overall_score': quality['overall_score']
                })
                
                if quality['overall_score'] > best_score:
                    best_score = quality['overall_score']
                    best_params = params
                    best_output = output
                    
            except Exception as e:
                results.append({
                    'params': params,
                    'error': str(e),
                    'overall_score': 0.0
                })
        
        # Calculate statistics
        scores = [r['overall_score'] for r in results]
        
        return {
            'best_params': best_params,
            'best_score': float(best_score),
            'best_output': best_output,
            'n_simulations': n_simulations,
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'score_range': (float(np.min(scores)), float(np.max(scores))),
            'improvements': {
                k: float(best_params.get(k, base_params.get(k, 0)))
                for k in base_params
            },
            'detailed_results': results[:10]  # Top 10 for reference
        }
    
    def _generate_variations(self, 
                            base_params: Dict,
                            n: int) -> List[Dict]:
        """Generate parameter variations using Latin Hypercube sampling."""
        variations = [base_params.copy()]  # Always include original
        
        for _ in range(n - 1):
            varied = base_params.copy()
            
            # Vary each numeric parameter
            for key, value in base_params.items():
                if isinstance(value, (int, float)):
                    # Get range if defined, else vary by ±30%
                    if key in self.parameter_ranges:
                        low, high = self.parameter_ranges[key]
                    else:
                        low = value * 0.7
                        high = value * 1.3
                    
                    # Sample from range
                    varied[key] = random.uniform(low, high)
                    
                    # Keep as int if original was int
                    if isinstance(value, int):
                        varied[key] = int(varied[key])
                
                elif isinstance(value, bool):
                    # Flip with small probability
                    if random.random() < 0.2:
                        varied[key] = not value
            
            variations.append(varied)
        
        return variations
    
    def _evaluate_quality(self,
                         output: np.ndarray,
                         seg_a: np.ndarray,
                         seg_b: np.ndarray,
                         params: Dict) -> Dict:
        """
        Fast quality evaluation for Monte Carlo simulation.
        
        Lighter weight than full quality assessment for speed.
        """
        n_samples = len(output)
        
        # Smoothness: spectral flux
        spectral_flux = self._calculate_spectral_flux(output)
        smoothness = 1.0 - min(1.0, spectral_flux / 100)
        
        # Clarity: spectral contrast
        clarity = self._calculate_clarity(output)
        
        # Energy flow: RMS stability
        energy_flow = self._calculate_energy_flow(output)
        
        # Harmonic quality: simplified harmonic analysis
        harmonic_quality = self._calculate_harmonic_quality(output, seg_a, seg_b)
        
        # Creativity: uniqueness of the mix
        creativity = self._calculate_creativity(params)
        
        # Surprise: how unexpected the transition is
        surprise = self._calculate_surprise(output, seg_a, seg_b)
        
        # Overall weighted score
        overall = (
            smoothness * self.quality_weights['smoothness'] +
            clarity * self.quality_weights['clarity'] +
            energy_flow * self.quality_weights['energy_flow'] +
            harmonic_quality * self.quality_weights['harmonic_quality'] +
            creativity * self.quality_weights['creativity'] +
            surprise * self.quality_weights['surprise']
        )
        
        return {
            'overall_score': float(overall),
            'smoothness': float(smoothness),
            'clarity': float(clarity),
            'energy_flow': float(energy_flow),
            'harmonic_quality': float(harmonic_quality),
            'creativity': float(creativity),
            'surprise': float(surprise)
        }
    
    def _calculate_spectral_flux(self, y: np.ndarray) -> float:
        """Fast spectral flux calculation."""
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        
        # Simple FFT-based flux
        n_frames = min(50, len(y) // 1024)
        if n_frames < 2:
            return 0.0
        
        frame_size = len(y) // n_frames
        spectra = []
        
        for i in range(n_frames):
            start = i * frame_size
            end = start + frame_size
            spec = np.abs(np.fft.fft(y[start:end])[:frame_size//2])
            spectra.append(spec)
        
        # Calculate flux
        flux = 0.0
        for i in range(1, len(spectra)):
            diff = spectra[i] - spectra[i-1]
            flux += np.sum(np.maximum(0, diff) ** 2)
        
        return flux / n_frames
    
    def _calculate_clarity(self, y: np.ndarray) -> float:
        """Fast clarity estimation."""
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        
        # Spectral contrast approximation
        spec = np.abs(np.fft.fft(y)[:len(y)//2])
        
        # Ratio of peak to mean
        if np.mean(spec) > 0:
            contrast = np.max(spec) / np.mean(spec)
            clarity = min(1.0, contrast / 10)
        else:
            clarity = 0.5
        
        return clarity
    
    def _calculate_energy_flow(self, y: np.ndarray) -> float:
        """Calculate energy flow stability."""
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        
        # RMS in windows
        window_size = int(0.5 * self.sr)
        n_windows = len(y) // window_size
        
        if n_windows < 3:
            return 0.7
        
        rms_values = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            rms = np.sqrt(np.mean(y[start:end] ** 2))
            rms_values.append(rms)
        
        rms_values = np.array(rms_values)
        
        # Lower variation = better flow
        if np.mean(rms_values) > 0:
            cv = np.std(rms_values) / np.mean(rms_values)
            flow = 1.0 - min(1.0, cv)
        else:
            flow = 0.5
        
        return flow
    
    def _calculate_harmonic_quality(self,
                                    y: np.ndarray,
                                    seg_a: np.ndarray,
                                    seg_b: np.ndarray) -> float:
        """Fast harmonic quality estimation."""
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        
        # Check for dissonance in middle section
        mid_start = len(y) // 3
        mid_end = 2 * len(y) // 3
        mid_section = y[mid_start:mid_end]
        
        # Spectral flatness as proxy for harmonic content
        spec = np.abs(np.fft.fft(mid_section)[:len(mid_section)//2])
        
        if len(spec) == 0 or np.mean(spec) == 0:
            return 0.5
        
        geometric_mean = np.exp(np.mean(np.log(spec + 1e-10)))
        arithmetic_mean = np.mean(spec)
        
        flatness = geometric_mean / arithmetic_mean
        
        # Higher flatness = more noise-like = worse harmony
        # Lower flatness = more tonal = better harmony
        harmonic = 1.0 - flatness
        
        return max(0.0, min(1.0, harmonic))
    
    def _calculate_creativity(self, params: Dict) -> float:
        """Calculate creativity score based on parameter choices."""
        creativity = 0.5  # Baseline
        
        # Non-standard parameter values increase creativity
        if params.get('transition_duration', 16) < 10 or params.get('transition_duration', 16) > 20:
            creativity += 0.1
        
        if params.get('bass_swap_ratio', 0.5) < 0.4 or params.get('bass_swap_ratio', 0.5) > 0.6:
            creativity += 0.1
        
        if params.get('use_echo', False):
            creativity += 0.1
        
        if params.get('filter_sweep', False):
            creativity += 0.15
        
        return min(1.0, creativity)
    
    def _calculate_surprise(self,
                           y: np.ndarray,
                           seg_a: np.ndarray,
                           seg_b: np.ndarray) -> float:
        """Calculate how surprising the transition is."""
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if seg_a.ndim > 1:
            seg_a = np.mean(seg_a, axis=1)
        if seg_b.ndim > 1:
            seg_b = np.mean(seg_b, axis=1)
        
        # Compare middle of transition to simple crossfade
        mid_start = len(y) // 3
        mid_end = 2 * len(y) // 3
        actual_mid = y[mid_start:mid_end]
        
        # Simple crossfade reference
        n = min(len(seg_a), len(seg_b), len(y))
        alpha = 0.5
        expected_mid = seg_a[mid_start:mid_end] * (1-alpha) + seg_b[mid_start:mid_end] * alpha
        
        if len(actual_mid) == 0 or len(expected_mid) == 0:
            return 0.5
        
        # Correlation to expected
        if len(actual_mid) == len(expected_mid):
            corr = np.corrcoef(actual_mid[:1000], expected_mid[:1000])[0, 1]
            surprise = 1.0 - abs(corr)
        else:
            surprise = 0.5
        
        return max(0.0, min(1.0, surprise))
    
    # ==================== MULTI-OBJECTIVE OPTIMIZATION ====================
    
    def optimize_multi_objective(self,
                                seg_a: np.ndarray,
                                seg_b: np.ndarray,
                                techniques: List[str],
                                base_params: Dict,
                                objectives: Dict[str, float],
                                n_iterations: int = 30) -> Dict:
        """
        Multi-objective optimization for transition parameters.
        
        Allows custom weighting of objectives:
        - smoothness: How smooth is the transition?
        - creativity: How creative/unexpected?
        - energy_match: Does energy flow naturally?
        - clarity: Can you hear both tracks clearly?
        
        Args:
            objectives: Dict of objective weights, e.g. {'smoothness': 0.5, 'creativity': 0.5}
        """
        # Normalize objective weights
        total = sum(objectives.values())
        normalized_objectives = {k: v/total for k, v in objectives.items()}
        
        # Update quality weights
        original_weights = self.quality_weights.copy()
        for obj, weight in normalized_objectives.items():
            if obj in self.quality_weights:
                self.quality_weights[obj] = weight
        
        # Run simulation for each technique
        all_results = []
        
        for technique in techniques:
            try:
                result = self.simulate_transitions(
                    seg_a, seg_b, technique, base_params,
                    n_simulations=n_iterations // len(techniques)
                )
                result['technique'] = technique
                all_results.append(result)
            except Exception as e:
                all_results.append({
                    'technique': technique,
                    'error': str(e),
                    'best_score': 0.0
                })
        
        # Restore original weights
        self.quality_weights = original_weights
        
        # Find pareto-optimal solutions
        pareto_front = self._find_pareto_front(all_results)
        
        # Select best overall
        best_result = max(all_results, key=lambda x: x.get('best_score', 0))
        
        return {
            'best_technique': best_result.get('technique'),
            'best_params': best_result.get('best_params'),
            'best_score': best_result.get('best_score', 0),
            'pareto_front': pareto_front,
            'all_results': [
                {
                    'technique': r.get('technique'),
                    'best_score': r.get('best_score', 0),
                    'score_mean': r.get('score_mean', 0)
                }
                for r in all_results
            ],
            'objectives_used': normalized_objectives
        }
    
    def _find_pareto_front(self, results: List[Dict]) -> List[Dict]:
        """Find Pareto-optimal solutions from results."""
        pareto_front = []
        
        for result in results:
            if 'error' in result:
                continue
            
            # Check if dominated
            is_dominated = False
            for other in results:
                if 'error' in other:
                    continue
                
                if other.get('best_score', 0) > result.get('best_score', 0):
                    if other.get('score_mean', 0) >= result.get('score_mean', 0):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_front.append({
                    'technique': result.get('technique'),
                    'best_score': result.get('best_score', 0)
                })
        
        return pareto_front
    
    # ==================== ENSEMBLE EVALUATION ====================
    
    def ensemble_evaluate(self,
                          output: np.ndarray,
                          seg_a: np.ndarray,
                          seg_b: np.ndarray,
                          n_evaluators: int = 5) -> Dict:
        """
        Ensemble evaluation using multiple quality metrics.
        
        Different evaluation strategies vote on quality,
        providing more robust scoring.
        """
        evaluations = []
        
        # 1. Standard quality metrics
        standard = self._evaluate_quality(output, seg_a, seg_b, {})
        evaluations.append(('standard', standard['overall_score']))
        
        # 2. Beginning-focused evaluation
        begin_section = output[:len(output)//3]
        begin_qual = self._evaluate_quality(begin_section, seg_a[:len(begin_section)], seg_b[:len(begin_section)], {})
        evaluations.append(('beginning', begin_qual['overall_score']))
        
        # 3. Middle-focused evaluation
        mid_start = len(output)//3
        mid_end = 2*len(output)//3
        mid_section = output[mid_start:mid_end]
        mid_qual = self._evaluate_quality(mid_section, seg_a[mid_start:mid_end], seg_b[mid_start:mid_end], {})
        evaluations.append(('middle', mid_qual['overall_score']))
        
        # 4. End-focused evaluation
        end_section = output[2*len(output)//3:]
        end_qual = self._evaluate_quality(end_section, seg_a[-len(end_section):], seg_b[-len(end_section):], {})
        evaluations.append(('ending', end_qual['overall_score']))
        
        # 5. Energy-focused evaluation (custom weights)
        original_weights = self.quality_weights.copy()
        self.quality_weights = {
            'smoothness': 0.15,
            'clarity': 0.15,
            'energy_flow': 0.50,
            'harmonic_quality': 0.10,
            'creativity': 0.05,
            'surprise': 0.05
        }
        energy_qual = self._evaluate_quality(output, seg_a, seg_b, {})
        evaluations.append(('energy_focused', energy_qual['overall_score']))
        self.quality_weights = original_weights
        
        # Calculate ensemble statistics
        scores = [e[1] for e in evaluations]
        
        return {
            'ensemble_mean': float(np.mean(scores)),
            'ensemble_std': float(np.std(scores)),
            'ensemble_min': float(np.min(scores)),
            'ensemble_max': float(np.max(scores)),
            'individual_scores': dict(evaluations),
            'confidence': float(1.0 - np.std(scores)),  # Lower variance = higher confidence
            'consensus': all(s > 0.5 for s in scores),  # All evaluators agree it's OK
            'recommended_action': self._recommend_from_ensemble(scores)
        }
    
    def _recommend_from_ensemble(self, scores: List[float]) -> str:
        """Recommend action based on ensemble scores."""
        mean = np.mean(scores)
        std = np.std(scores)
        
        if mean > 0.8 and std < 0.1:
            return 'excellent_keep'
        elif mean > 0.7:
            return 'good_keep'
        elif mean > 0.5:
            if std > 0.2:
                return 'inconsistent_review'
            else:
                return 'acceptable_keep'
        else:
            return 'poor_reject'
    
    # ==================== PREDICTION BEFORE COMMIT ====================
    
    def predict_quality(self,
                        seg_a: np.ndarray,
                        seg_b: np.ndarray,
                        technique: str,
                        params: Dict) -> Dict:
        """
        Predict transition quality WITHOUT actually creating the mix.
        
        Uses lightweight heuristics to estimate quality.
        Much faster than full simulation.
        """
        # Analyze input signals
        if seg_a.ndim > 1:
            seg_a_mono = np.mean(seg_a, axis=1)
        else:
            seg_a_mono = seg_a
            
        if seg_b.ndim > 1:
            seg_b_mono = np.mean(seg_b, axis=1)
        else:
            seg_b_mono = seg_b
        
        # Predict smoothness based on spectral similarity
        spec_a = np.abs(np.fft.fft(seg_a_mono[:4096]))[:2048]
        spec_b = np.abs(np.fft.fft(seg_b_mono[:4096]))[:2048]
        
        spectral_corr = np.corrcoef(spec_a, spec_b)[0, 1]
        predicted_smoothness = max(0.3, 0.5 + 0.5 * spectral_corr)
        
        # Predict clarity based on spectral overlap
        overlap = np.minimum(spec_a, spec_b)
        overlap_ratio = np.sum(overlap) / (np.sum(spec_a) + np.sum(spec_b) + 1e-10)
        predicted_clarity = max(0.3, 1.0 - overlap_ratio)
        
        # Predict energy flow based on RMS
        rms_a = np.sqrt(np.mean(seg_a_mono ** 2))
        rms_b = np.sqrt(np.mean(seg_b_mono ** 2))
        rms_ratio = min(rms_a, rms_b) / (max(rms_a, rms_b) + 1e-10)
        predicted_energy = 0.3 + 0.7 * rms_ratio
        
        # Technique-based predictions (all techniques supported; unknown get 0.0)
        technique_bonus = {
            'long_blend': 0.1,
            'bass_swap': 0.05 if overlap_ratio > 0.3 else -0.05,
            'filter_sweep': 0.05,
            'echo_out': 0.0,
            'quick_cut': -0.1 if spectral_corr < 0.5 else 0.1,
            'staggered_stem_mix': 0.1,
            'phrase_match': 0.08,
            'loop_transition': 0.05,
            'drop_on_the_one': -0.05 if spectral_corr < 0.5 else 0.08,
            'back_and_forth': 0.06,
            'drum_roll': 0.05,
            'thematic_handoff': 0.08,
            'backspin': 0.0,
            'double_drop': 0.05,
            'acapella_overlay': 0.08,
            'modulation': 0.05,
            'energy_build': 0.05,
            'breakdown_to_build': 0.05,
            'drop_mix': 0.0,
            'partial_stem_separation': 0.06,
            'vocal_layering': 0.08,
            'progressive_morph': 0.12,  # New technique bonus
        }
        
        technique_adj = technique_bonus.get(technique, 0.0)
        
        # Overall prediction
        predicted_overall = (
            predicted_smoothness * 0.35 +
            predicted_clarity * 0.30 +
            predicted_energy * 0.25 +
            0.5 * 0.10  # Neutral creativity/surprise
        ) + technique_adj
        
        predicted_overall = max(0.0, min(1.0, predicted_overall))
        
        return {
            'predicted_quality': float(predicted_overall),
            'predicted_smoothness': float(predicted_smoothness),
            'predicted_clarity': float(predicted_clarity),
            'predicted_energy_flow': float(predicted_energy),
            'spectral_correlation': float(spectral_corr),
            'spectral_overlap': float(overlap_ratio),
            'technique_adjustment': float(technique_adj),
            'confidence': 0.7,  # Prediction confidence
            'recommendation': 'proceed' if predicted_overall > 0.55 else 'reconsider'
        }
    
    def should_proceed(self,
                       seg_a: np.ndarray,
                       seg_b: np.ndarray,
                       technique: str,
                       params: Dict,
                       threshold: float = 0.5) -> Tuple[bool, Dict]:
        """
        Quick check whether to proceed with transition.
        
        Returns (should_proceed, prediction_data)
        """
        prediction = self.predict_quality(seg_a, seg_b, technique, params)
        
        should = prediction['predicted_quality'] >= threshold
        
        return should, prediction
