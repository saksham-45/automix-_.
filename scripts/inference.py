#!/usr/bin/env python3
"""
AI DJ Inference Script

Use the trained model to predict transitions between songs.

Usage:
    # Predict transition for two songs
    python scripts/inference.py --song-a song_a.wav --song-b song_b.wav
    
    # Interactive mode
    python scripts/inference.py --interactive
    
    # Generate curves and visualize
    python scripts/inference.py --song-a song_a.wav --song-b song_b.wav --visualize
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.decision_nn import DecisionNN, prepare_features
from src.models.curve_lstm import CurveLSTM
from src.models.combined_model import CombinedDJModel


class DJTransitionPredictor:
    """
    Inference wrapper for the AI DJ model.
    """
    
    def __init__(self, 
                 decision_nn_path: Optional[str] = None,
                 curve_lstm_path: Optional[str] = None,
                 combined_path: Optional[str] = None):
        """
        Load trained models.
        
        Args:
            decision_nn_path: Path to DecisionNN checkpoint
            curve_lstm_path: Path to CurveLSTM checkpoint
            combined_path: Path to combined model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.decision_nn = None
        self.curve_lstm = None
        self.combined = None
        
        # Try to load models
        if combined_path and Path(combined_path).exists():
            self._load_combined(combined_path)
        else:
            if decision_nn_path and Path(decision_nn_path).exists():
                self._load_decision_nn(decision_nn_path)
            if curve_lstm_path and Path(curve_lstm_path).exists():
                self._load_curve_lstm(curve_lstm_path)
        
        # Check what we have
        if self.decision_nn is None and self.combined is None:
            print("WARNING: No models loaded. Run training first.")
            print("  python scripts/train_model.py --stage 1")
    
    def _load_decision_nn(self, path: str):
        """Load DecisionNN model."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.decision_nn = DecisionNN(input_dim=13)
        self.decision_nn.load_state_dict(checkpoint['model_state_dict'])
        self.decision_nn = self.decision_nn.to(self.device)
        self.decision_nn.eval()
        print(f"Loaded DecisionNN from: {path}")
    
    def _load_curve_lstm(self, path: str):
        """Load CurveLSTM model."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.curve_lstm = CurveLSTM(context_dim=32, song_feature_dim=13)
        self.curve_lstm.load_state_dict(checkpoint['model_state_dict'])
        self.curve_lstm = self.curve_lstm.to(self.device)
        self.curve_lstm.eval()
        print(f"Loaded CurveLSTM from: {path}")
    
    def _load_combined(self, path: str):
        """Load combined model."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.combined = CombinedDJModel()
        self.combined.load_state_dict(checkpoint['model_state_dict'])
        self.combined = self.combined.to(self.device)
        self.combined.eval()
        print(f"Loaded combined model from: {path}")
    
    def predict_from_features(self, 
                              song_a_features: Dict,
                              song_b_features: Dict,
                              generate_curves: bool = True,
                              duration_sec: float = 10.0) -> Dict:
        """
        Predict transition given song features.
        
        Args:
            song_a_features: Features for song A (outgoing)
            song_b_features: Features for song B (incoming)
            generate_curves: Whether to generate automation curves
            duration_sec: Duration of transition
        
        Returns:
            Prediction dictionary
        """
        # Build sample dict matching training format
        sample = {
            'tempo_outgoing': song_a_features.get('tempo', 120),
            'tempo_incoming': song_b_features.get('tempo', 120),
            'energy_before': song_a_features.get('energy', 0.5),
            'energy_after': song_b_features.get('energy', 0.5),
            'energy_during': (song_a_features.get('energy', 0.5) + song_b_features.get('energy', 0.5)) / 2,
            'key_outgoing': song_a_features.get('key', 'C'),
            'key_incoming': song_b_features.get('key', 'C'),
            'key_compatible': self._keys_compatible(
                song_a_features.get('key', 'C'),
                song_b_features.get('key', 'C')
            ),
            'harmonic_tension': song_a_features.get('harmonic_tension', 0.1),
            'spectral_smoothness': 0.8,
            'frequency_masking': 0.1,
            'energy_build': song_b_features.get('energy', 0.5) > song_a_features.get('energy', 0.5),
            'energy_dip': False,
        }
        
        # Prepare features
        features, key_a, key_b = prepare_features(sample)
        
        # Convert to tensors
        features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        key_a_t = torch.tensor([key_a], dtype=torch.long).to(self.device)
        key_b_t = torch.tensor([key_b], dtype=torch.long).to(self.device)
        
        # Predict
        with torch.no_grad():
            if self.combined is not None:
                # Use combined model
                duration_frames = int(duration_sec * 10)
                outputs = self.combined(features_t, key_a_t, key_b_t, duration_frames)
            elif self.decision_nn is not None:
                decision_outputs = self.decision_nn(features_t, key_a_t, key_b_t)
                outputs = decision_outputs
                
                # Generate curves if LSTM is available and requested
                if generate_curves and self.curve_lstm is not None:
                    context = decision_outputs['context_vector']
                    duration_frames = int(duration_sec * 10)
                    curve_outputs = self.curve_lstm(context, features_t, duration_frames=duration_frames)
                    outputs['curves'] = curve_outputs['curves']
            else:
                return {'error': 'No model loaded'}
        
        # Process outputs
        technique_names = ['cut', 'blend', 'drop']
        technique_idx = outputs['technique_probs'].argmax(dim=1).item()
        
        result = {
            'technique': technique_names[technique_idx],
            'technique_confidence': outputs['technique_probs'][0, technique_idx].item(),
            'bass_swap': outputs['bass_swap'].item() > 0.5,
            'bass_swap_prob': outputs['bass_swap'].item(),
            'low_cut_incoming': outputs['low_cut'].item() > 0.5,
            'high_cut_outgoing': outputs['high_cut'].item() > 0.5,
            'duration_bars': int(round(outputs['duration_bars'].item())),
            'key_compatible': sample['key_compatible'],
        }
        
        # Add curves if available
        if generate_curves and 'curves' in outputs:
            curves = outputs['curves'].cpu().numpy()[0]  # [frames, params]
            param_names = CurveLSTM.PARAM_NAMES
            
            result['curves'] = {
                'time': np.linspace(0, duration_sec, curves.shape[0]).tolist(),
            }
            for i, name in enumerate(param_names):
                result['curves'][name] = curves[:, i].tolist()
        
        return result
    
    def _keys_compatible(self, key_a: str, key_b: str) -> bool:
        """Check if two keys are compatible."""
        key_to_idx = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }
        
        idx_a = key_to_idx.get(key_a, 0)
        idx_b = key_to_idx.get(key_b, 0)
        
        diff = abs(idx_a - idx_b)
        compatible = [0, 5, 7, 3, 4, 8, 9]  # Same, 4th, 5th, minor 3rd, etc.
        
        return diff in compatible or (12 - diff) in compatible
    
    def predict_from_audio(self, 
                          song_a_path: str, 
                          song_b_path: str,
                          duration_sec: float = 10.0) -> Dict:
        """
        Predict transition from audio files.
        
        Args:
            song_a_path: Path to song A audio
            song_b_path: Path to song B audio
            duration_sec: Transition duration
        
        Returns:
            Prediction dictionary
        """
        try:
            from src.song_analyzer import SongAnalyzer
            
            analyzer = SongAnalyzer()
            
            print(f"Analyzing {Path(song_a_path).name}...")
            analysis_a = analyzer.analyze(song_a_path)
            
            print(f"Analyzing {Path(song_b_path).name}...")
            analysis_b = analyzer.analyze(song_b_path)
            
            # Extract features
            # Handle key extraction (may be dict with 'key' or direct string)
            key_a = analysis_a.get('harmony', {}).get('key', 'C')
            if isinstance(key_a, dict):
                key_a = key_a.get('key', 'C')
            elif isinstance(key_a, list) and len(key_a) > 0:
                key_a = key_a[0] if isinstance(key_a[0], str) else 'C'
            if not isinstance(key_a, str):
                key_a = 'C'
            
            key_b = analysis_b.get('harmony', {}).get('key', 'C')
            if isinstance(key_b, dict):
                key_b = key_b.get('key', 'C')
            elif isinstance(key_b, list) and len(key_b) > 0:
                key_b = key_b[0] if isinstance(key_b[0], str) else 'C'
            if not isinstance(key_b, str):
                key_b = 'C'
            
            song_a_features = {
                'tempo': analysis_a.get('tempo', {}).get('bpm', 120) if isinstance(analysis_a.get('tempo'), dict) else 120,
                'key': key_a,
                'energy': analysis_a.get('energy', {}).get('overall_energy', 0.5) if isinstance(analysis_a.get('energy'), dict) else 0.5,
            }
            
            song_b_features = {
                'tempo': analysis_b.get('tempo', {}).get('bpm', 120) if isinstance(analysis_b.get('tempo'), dict) else 120,
                'key': key_b,
                'energy': analysis_b.get('energy', {}).get('overall_energy', 0.5) if isinstance(analysis_b.get('energy'), dict) else 0.5,
            }
            
            return self.predict_from_features(song_a_features, song_b_features, 
                                             duration_sec=duration_sec)
        
        except ImportError:
            print("SongAnalyzer not available. Using manual feature input.")
            return {'error': 'SongAnalyzer not available'}
    
    def interactive_mode(self):
        """Interactive prediction mode."""
        print("\n" + "="*60)
        print("AI DJ TRANSITION PREDICTOR - Interactive Mode")
        print("="*60)
        
        while True:
            print("\nEnter song features (or 'quit' to exit):")
            
            # Song A
            print("\n--- SONG A (outgoing) ---")
            tempo_a = input("Tempo (BPM) [120]: ").strip() or "120"
            key_a = input("Key (C, D, E, etc.) [C]: ").strip() or "C"
            energy_a = input("Energy (0-1) [0.5]: ").strip() or "0.5"
            
            if tempo_a.lower() == 'quit':
                break
            
            # Song B
            print("\n--- SONG B (incoming) ---")
            tempo_b = input("Tempo (BPM) [120]: ").strip() or "120"
            key_b = input("Key (C, D, E, etc.) [C]: ").strip() or "C"
            energy_b = input("Energy (0-1) [0.5]: ").strip() or "0.5"
            
            # Predict
            song_a = {'tempo': float(tempo_a), 'key': key_a, 'energy': float(energy_a)}
            song_b = {'tempo': float(tempo_b), 'key': key_b, 'energy': float(energy_b)}
            
            result = self.predict_from_features(song_a, song_b)
            
            # Display results
            print("\n" + "-"*40)
            print("PREDICTION:")
            print("-"*40)
            print(f"  Technique: {result.get('technique', 'unknown').upper()}")
            print(f"  Confidence: {result.get('technique_confidence', 0)*100:.1f}%")
            print(f"  Duration: {result.get('duration_bars', 5)} bars")
            print(f"  Bass Swap: {'YES' if result.get('bass_swap') else 'NO'}")
            print(f"  Low Cut Incoming: {'YES' if result.get('low_cut_incoming') else 'NO'}")
            print(f"  High Cut Outgoing: {'YES' if result.get('high_cut_outgoing') else 'NO'}")
            print(f"  Key Compatible: {'YES' if result.get('key_compatible') else 'NO'}")


def visualize_curves(prediction: Dict, output_path: Optional[str] = None):
    """Visualize predicted automation curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    if 'curves' not in prediction:
        print("No curves in prediction")
        return
    
    curves = prediction['curves']
    time = curves['time']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Volume curves
    axes[0].plot(time, curves['volume_a'], label='Volume A', color='blue')
    axes[0].plot(time, curves['volume_b'], label='Volume B', color='red')
    axes[0].set_ylabel('Volume')
    axes[0].set_title('Volume Crossfade')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # EQ curves
    axes[1].plot(time, curves['bass_a'], label='Bass A', linestyle='--')
    axes[1].plot(time, curves['bass_b'], label='Bass B', linestyle='--')
    axes[1].plot(time, curves['high_a'], label='High A', linestyle=':')
    axes[1].plot(time, curves['high_b'], label='High B', linestyle=':')
    axes[1].set_ylabel('EQ Level')
    axes[1].set_title('EQ Automation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Filter
    axes[2].plot(time, curves['filter_freq'], label='Filter Freq', color='purple')
    axes[2].plot(time, curves['filter_res'], label='Filter Res', color='green')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Filter')
    axes[2].set_title('Filter Automation')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='AI DJ Transition Predictor')
    parser.add_argument('--song-a', help='Path to song A (outgoing)')
    parser.add_argument('--song-b', help='Path to song B (incoming)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Visualize curves')
    parser.add_argument('--model-dir', default='models/',
                       help='Directory containing model checkpoints')
    parser.add_argument('--output', '-o', help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Find models
    model_dir = Path(args.model_dir)
    decision_nn_path = model_dir / 'decision_nn.pt'
    curve_lstm_path = model_dir / 'curve_lstm.pt'
    combined_path = model_dir / 'combined_model.pt'
    
    # Initialize predictor
    predictor = DJTransitionPredictor(
        decision_nn_path=str(decision_nn_path) if decision_nn_path.exists() else None,
        curve_lstm_path=str(curve_lstm_path) if curve_lstm_path.exists() else None,
        combined_path=str(combined_path) if combined_path.exists() else None
    )
    
    if args.interactive:
        predictor.interactive_mode()
    elif args.song_a and args.song_b:
        print("\nPredicting transition...")
        result = predictor.predict_from_audio(args.song_a, args.song_b)
        
        print("\n" + "="*40)
        print("PREDICTION RESULT:")
        print("="*40)
        print(json.dumps({k: v for k, v in result.items() if k != 'curves'}, indent=2))
        
        if args.visualize and 'curves' in result:
            visualize_curves(result, args.output)
    else:
        # Demo with manual features
        print("\nDemo prediction (no audio files provided):")
        
        song_a = {'tempo': 128, 'key': 'Am', 'energy': 0.7}
        song_b = {'tempo': 130, 'key': 'Em', 'energy': 0.8}
        
        print(f"\nSong A: {song_a}")
        print(f"Song B: {song_b}")
        
        result = predictor.predict_from_features(song_a, song_b)
        
        print("\n" + "="*40)
        print("PREDICTION:")
        print("="*40)
        print(json.dumps({k: v for k, v in result.items() if k != 'curves'}, indent=2))


if __name__ == '__main__':
    main()

