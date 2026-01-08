#!/usr/bin/env python3
"""
End-to-end testing script for the combined AI DJ model.

Tests the full pipeline:
1. Load both DecisionNN and CurveLSTM
2. Run predictions on test set
3. Compare predictions vs ground truth
4. Generate evaluation metrics
"""
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.decision_nn import DecisionNN, prepare_features, prepare_targets
from src.models.curve_lstm import CurveLSTM, prepare_curve_targets
from torch.utils.data import Dataset, DataLoader


class TransitionDataset(Dataset):
    def __init__(self, samples: list, include_curves: bool = False):
        self.samples = samples
        self.include_curves = include_curves
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        features, key_a, key_b = prepare_features(sample)
        targets = prepare_targets(sample)
        
        result = {
            'features': torch.tensor(features, dtype=torch.float32),
            'key_a': torch.tensor(key_a, dtype=torch.long),
            'key_b': torch.tensor(key_b, dtype=torch.long),
            'technique': torch.tensor(targets['technique'], dtype=torch.long),
            'bass_swap': torch.tensor(targets['bass_swap'], dtype=torch.float32),
            'duration_bars': torch.tensor(targets['duration_bars'], dtype=torch.float32),
        }
        
        if self.include_curves:
            curves = prepare_curve_targets(sample)
            result['curves'] = torch.tensor(curves, dtype=torch.float32)
        
        return result


def load_models(decision_nn_path: Path, curve_lstm_path: Path, device: str = 'cpu'):
    """Load both DecisionNN and CurveLSTM models."""
    # Load DecisionNN
    checkpoint_nn = torch.load(decision_nn_path, map_location=device, weights_only=False)
    decision_nn = DecisionNN(input_dim=13)
    decision_nn.load_state_dict(checkpoint_nn['model_state_dict'])
    decision_nn = decision_nn.to(device)
    decision_nn.eval()
    
    # Load CurveLSTM
    checkpoint_lstm = torch.load(curve_lstm_path, map_location=device, weights_only=False)
    curve_lstm = CurveLSTM(context_dim=32, song_feature_dim=13)
    curve_lstm.load_state_dict(checkpoint_lstm['model_state_dict'])
    curve_lstm = curve_lstm.to(device)
    curve_lstm.eval()
    
    return decision_nn, curve_lstm


def test_combined_model(decision_nn: DecisionNN,
                       curve_lstm: CurveLSTM,
                       data_loader: DataLoader,
                       device: str = 'cpu'):
    """Test the combined model on a dataset."""
    all_predictions = {
        'technique': [],
        'bass_swap': [],
        'duration': [],
        'curves': [],
    }
    
    all_targets = {
        'technique': [],
        'bass_swap': [],
        'duration': [],
        'curves': [],
    }
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            key_a = batch['key_a'].to(device)
            key_b = batch['key_b'].to(device)
            
            # DecisionNN predictions
            decision_outputs = decision_nn(features, key_a, key_b)
            context = decision_outputs['context_vector']
            
            # CurveLSTM predictions
            duration_frames = batch['curves'].shape[1] if 'curves' in batch else 100
            curve_outputs = curve_lstm(context, features, duration_frames=duration_frames)
            
            # Collect predictions
            technique_pred = decision_outputs['technique_probs'].argmax(dim=1).cpu().numpy()
            bass_swap_pred = (decision_outputs['bass_swap'] > 0.5).cpu().numpy().astype(int)
            duration_pred = decision_outputs['duration_bars'].cpu().numpy()
            curves_pred = curve_outputs['curves'].cpu().numpy()
            
            all_predictions['technique'].extend(technique_pred)
            all_predictions['bass_swap'].extend(bass_swap_pred)
            all_predictions['duration'].extend(duration_pred)
            all_predictions['curves'].extend(curves_pred)
            
            # Collect targets
            all_targets['technique'].extend(batch['technique'].cpu().numpy())
            all_targets['bass_swap'].extend(batch['bass_swap'].cpu().numpy().astype(int))
            all_targets['duration'].extend(batch['duration_bars'].cpu().numpy())
            if 'curves' in batch:
                all_targets['curves'].extend(batch['curves'].cpu().numpy())
    
    return all_predictions, all_targets


def calculate_curve_metrics(predicted: np.ndarray, target: np.ndarray) -> Dict:
    """Calculate metrics for curve predictions."""
    # MSE per parameter
    mse_per_param = np.mean((predicted - target) ** 2, axis=(0, 1))
    
    # Overall MSE
    mse_overall = np.mean((predicted - target) ** 2)
    
    # Smoothness (frame-to-frame variance)
    pred_diff = np.diff(predicted, axis=1)
    target_diff = np.diff(target, axis=1)
    smoothness_pred = np.mean(pred_diff ** 2)
    smoothness_target = np.mean(target_diff ** 2)
    
    return {
        'mse_overall': float(mse_overall),
        'mse_per_param': mse_per_param.tolist(),
        'smoothness_pred': float(smoothness_pred),
        'smoothness_target': float(smoothness_target),
        'smoothness_ratio': float(smoothness_pred / (smoothness_target + 1e-6)),
    }


def main():
    parser = argparse.ArgumentParser(description='End-to-end model testing')
    parser.add_argument('--decision-nn', type=str, default='models/decision_nn.pt',
                       help='Path to DecisionNN checkpoint')
    parser.add_argument('--curve-lstm', type=str, default='models/curve_lstm.pt',
                       help='Path to CurveLSTM checkpoint')
    parser.add_argument('--test-data', type=str, default='data/training_splits/test.json',
                       help='Path to test data')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/auto)')
    parser.add_argument('--output', type=str, default='models/end_to_end_results.json',
                       help='Output path for results')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*60)
    print("END-TO-END MODEL TESTING")
    print("="*60)
    print(f"DecisionNN: {args.decision_nn}")
    print(f"CurveLSTM: {args.curve_lstm}")
    print(f"Test Data: {args.test_data}")
    print(f"Device: {device}")
    
    # Load models
    print("\nLoading models...")
    if not Path(args.decision_nn).exists():
        print(f"ERROR: DecisionNN not found: {args.decision_nn}")
        return
    if not Path(args.curve_lstm).exists():
        print(f"ERROR: CurveLSTM not found: {args.curve_lstm}")
        return
    
    decision_nn, curve_lstm = load_models(
        Path(args.decision_nn),
        Path(args.curve_lstm),
        device
    )
    print("✓ Models loaded")
    
    # Load test data
    print("\nLoading test data...")
    with open(args.test_data) as f:
        test_samples = json.load(f)
    
    print(f"Test samples: {len(test_samples)}")
    
    # Create dataset and loader
    test_dataset = TransitionDataset(test_samples, include_curves=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Test
    print("\nRunning end-to-end evaluation...")
    predictions, targets = test_combined_model(decision_nn, curve_lstm, test_loader, device)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # Decision metrics
    technique_acc = (np.array(predictions['technique']) == np.array(targets['technique'])).mean()
    bass_swap_acc = (np.array(predictions['bass_swap']) == np.array(targets['bass_swap'])).mean()
    duration_mae = np.abs(np.array(predictions['duration']) - np.array(targets['duration'])).mean()
    
    # Curve metrics
    if len(predictions['curves']) > 0 and len(targets['curves']) > 0:
        pred_curves = np.array(predictions['curves'])
        target_curves = np.array(targets['curves'])
        curve_metrics = calculate_curve_metrics(pred_curves, target_curves)
    else:
        curve_metrics = {}
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Technique Accuracy: {technique_acc*100:.2f}%")
    print(f"Bass Swap Accuracy: {bass_swap_acc*100:.2f}%")
    print(f"Duration MAE: {duration_mae:.3f} bars")
    
    if curve_metrics:
        print(f"\nCurve Metrics:")
        print(f"  Overall MSE: {curve_metrics['mse_overall']:.4f}")
        print(f"  Smoothness Ratio: {curve_metrics['smoothness_ratio']:.3f}")
        print(f"    (1.0 = perfect match, >1.0 = more jittery, <1.0 = smoother)")
    
    # Save results
    results = {
        'decision_metrics': {
            'technique_accuracy': float(technique_acc),
            'bass_swap_accuracy': float(bass_swap_acc),
            'duration_mae': float(duration_mae),
        },
        'curve_metrics': curve_metrics,
        'num_samples': len(test_samples),
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to: {args.output}")


if __name__ == '__main__':
    main()

