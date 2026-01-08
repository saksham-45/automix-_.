#!/usr/bin/env python3
"""
Comprehensive model evaluation script.

Evaluates:
- Technique classification accuracy
- Bass swap prediction accuracy
- Duration prediction MAE
- Per-technique precision/recall
- Confusion matrix
"""
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.decision_nn import DecisionNN, prepare_features, prepare_targets
from torch.utils.data import Dataset, DataLoader


class TransitionDataset(Dataset):
    def __init__(self, samples: list):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        features, key_a, key_b = prepare_features(sample)
        targets = prepare_targets(sample)
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'key_a': torch.tensor(key_a, dtype=torch.long),
            'key_b': torch.tensor(key_b, dtype=torch.long),
            'technique': torch.tensor(targets['technique'], dtype=torch.long),
            'bass_swap': torch.tensor(targets['bass_swap'], dtype=torch.float32),
            'duration_bars': torch.tensor(targets['duration_bars'], dtype=torch.float32),
        }


def load_model(model_path: Path, device: str = 'cpu'):
    """Load trained DecisionNN model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = DecisionNN(input_dim=13)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_decision_nn(model: DecisionNN, data_loader: DataLoader, device: str = 'cpu'):
    """Evaluate DecisionNN on a dataset."""
    all_predictions = {
        'technique': [],
        'bass_swap': [],
        'duration': [],
    }
    
    all_targets = {
        'technique': [],
        'bass_swap': [],
        'duration': [],
    }
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            key_a = batch['key_a'].to(device)
            key_b = batch['key_b'].to(device)
            
            outputs = model(features, key_a, key_b)
            
            # Predictions
            technique_pred = outputs['technique_probs'].argmax(dim=1).cpu().numpy()
            bass_swap_pred = (outputs['bass_swap'] > 0.5).cpu().numpy().astype(int)
            duration_pred = outputs['duration_bars'].cpu().numpy()
            
            all_predictions['technique'].extend(technique_pred)
            all_predictions['bass_swap'].extend(bass_swap_pred)
            all_predictions['duration'].extend(duration_pred)
            
            # Targets
            all_targets['technique'].extend(batch['technique'].cpu().numpy())
            all_targets['bass_swap'].extend(batch['bass_swap'].cpu().numpy().astype(int))
            all_targets['duration'].extend(batch['duration_bars'].cpu().numpy())
    
    return all_predictions, all_targets


def print_metrics(predictions: dict, targets: dict, technique_names: list = None):
    """Print evaluation metrics."""
    if technique_names is None:
        technique_names = ['cut', 'blend', 'drop']
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    # Technique classification
    print("\n1. TECHNIQUE CLASSIFICATION")
    print("-" * 60)
    
    tech_pred = np.array(predictions['technique'])
    tech_true = np.array(targets['technique'])
    
    accuracy = (tech_pred == tech_true).mean()
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        tech_true, tech_pred, average=None, zero_division=0
    )
    
    print("\nPer-technique metrics:")
    for i, name in enumerate(technique_names):
        if i < len(precision):
            print(f"  {name:10s}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, "
                  f"F1={f1[i]:.3f}, Support={support[i]}")
    
    # Confusion matrix
    cm = confusion_matrix(tech_true, tech_pred, labels=range(len(technique_names)))
    print("\nConfusion Matrix:")
    print("         Predicted:")
    print("         ", end="")
    for name in technique_names:
        print(f"{name:>8s}", end="")
    print()
    for i, name in enumerate(technique_names):
        print(f"True {name:5s}:", end="")
        for j in range(len(technique_names)):
            if i < cm.shape[0] and j < cm.shape[1]:
                print(f"{cm[i][j]:8d}", end="")
            else:
                print(f"{0:8d}", end="")
        print()
    
    # Bass swap prediction
    print("\n2. BASS SWAP PREDICTION")
    print("-" * 60)
    
    bass_pred = np.array(predictions['bass_swap'])
    bass_true = np.array(targets['bass_swap'])
    
    bass_accuracy = (bass_pred == bass_true).mean()
    print(f"Accuracy: {bass_accuracy*100:.2f}%")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        bass_true, bass_pred, average=None, zero_division=0
    )
    print(f"  No swap (0): Precision={precision[0]:.3f}, Recall={recall[0]:.3f}, "
          f"F1={f1[0]:.3f}, Support={support[0]}")
    print(f"  Swap (1):    Precision={precision[1]:.3f}, Recall={recall[1]:.3f}, "
          f"F1={f1[1]:.3f}, Support={support[1]}")
    
    # Duration prediction
    print("\n3. DURATION PREDICTION (bars)")
    print("-" * 60)
    
    dur_pred = np.array(predictions['duration'])
    dur_true = np.array(targets['duration'])
    
    mae = np.abs(dur_pred - dur_true).mean()
    rmse = np.sqrt(((dur_pred - dur_true) ** 2).mean())
    mape = np.abs((dur_pred - dur_true) / (dur_true + 1e-6)).mean() * 100
    
    print(f"MAE (Mean Absolute Error): {mae:.3f} bars")
    print(f"RMSE (Root Mean Squared Error): {rmse:.3f} bars")
    print(f"MAPE (Mean Absolute % Error): {mape:.2f}%")
    print(f"True duration range: {dur_true.min():.1f} - {dur_true.max():.1f} bars")
    print(f"Pred duration range: {dur_pred.min():.1f} - {dur_pred.max():.1f} bars")
    
    return {
        'technique_accuracy': accuracy,
        'technique_precision': precision,
        'technique_recall': recall,
        'technique_f1': f1,
        'bass_swap_accuracy': bass_accuracy,
        'duration_mae': mae,
        'duration_rmse': rmse,
        'duration_mape': mape,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model', '-m', type=str, default='models/decision_nn.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data', '-d', type=str, default='data/training_splits/test.json',
                       help='Path to test data JSON')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Device: {device}")
    
    # Load model
    if not Path(args.model).exists():
        print(f"\nERROR: Model file not found: {args.model}")
        return
    
    print("\nLoading model...")
    model, checkpoint = load_model(Path(args.model), device)
    
    if 'num_val_samples' in checkpoint:
        print(f"Model trained on: {checkpoint.get('num_train_samples', '?')} train, "
              f"{checkpoint.get('num_val_samples', '?')} val samples")
    
    # Load data
    print("\nLoading test data...")
    with open(args.data) as f:
        test_samples = json.load(f)
    
    print(f"Test samples: {len(test_samples)}")
    
    # Create dataset and loader
    test_dataset = TransitionDataset(test_samples)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate
    print("\nRunning evaluation...")
    predictions, targets = evaluate_decision_nn(model, test_loader, device)
    
    # Print metrics
    metrics = print_metrics(predictions, targets)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Technique Accuracy: {metrics['technique_accuracy']*100:.2f}%")
    print(f"Bass Swap Accuracy: {metrics['bass_swap_accuracy']*100:.2f}%")
    print(f"Duration MAE: {metrics['duration_mae']:.3f} bars")
    
    # Save results
    output_path = Path(args.model).parent / 'evaluation_results.json'
    metrics_dict = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            metrics_dict[k] = v.tolist()
        else:
            metrics_dict[k] = float(v) if isinstance(v, (np.integer, np.floating)) else v
    
    results = {
        'model_path': str(args.model),
        'test_data_path': str(args.data),
        'num_samples': len(test_samples),
        'metrics': metrics_dict
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved evaluation results to: {output_path}")


if __name__ == '__main__':
    main()

