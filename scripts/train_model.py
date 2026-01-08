#!/usr/bin/env python3
"""
Training Pipeline for AI DJ Model

Supports:
- Stage 1: Train DecisionNN only (requires 500+ samples)
- Stage 2: Train CurveLSTM only (requires 2000+ samples)
- Stage 3: End-to-end fine-tuning

Usage:
    python scripts/train_model.py --stage 1  # Decision NN only
    python scripts/train_model.py --stage 2  # Curve LSTM only
    python scripts/train_model.py --stage 3  # End-to-end
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.decision_nn import DecisionNN, DecisionNNLoss, prepare_features, prepare_targets
from src.models.curve_lstm import CurveLSTM, CurveLSTMLoss, prepare_curve_targets
from src.models.combined_model import CombinedDJModel, CombinedLoss, save_model


class TransitionDataset(Dataset):
    """Dataset for DJ transition training."""
    
    def __init__(self, samples: List[Dict], include_curves: bool = False):
        self.samples = samples
        self.include_curves = include_curves
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Prepare features
        features, key_a, key_b = prepare_features(sample)
        
        # Prepare targets
        targets = prepare_targets(sample)
        
        result = {
            'features': torch.tensor(features, dtype=torch.float32),
            'key_a': torch.tensor(key_a, dtype=torch.long),
            'key_b': torch.tensor(key_b, dtype=torch.long),
            'technique': torch.tensor(targets['technique'], dtype=torch.long),
            'bass_swap': torch.tensor(targets['bass_swap'], dtype=torch.float32),
            'low_cut': torch.tensor(targets['low_cut'], dtype=torch.float32),
            'high_cut': torch.tensor(targets['high_cut'], dtype=torch.float32),
            'duration_bars': torch.tensor(targets['duration_bars'], dtype=torch.float32),
        }
        
        # Add curves if needed
        if self.include_curves:
            curves = prepare_curve_targets(sample)
            result['curves'] = torch.tensor(curves, dtype=torch.float32)
        
        return result


def load_training_data(use_splits: bool = True) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load training data with splits.
    
    Args:
        use_splits: If True, load from prepared splits. If False, load all and split randomly.
    
    Returns:
        (train_samples, val_samples, test_samples)
    """
    if use_splits:
        # Load from prepared splits
        split_dir = Path('data/training_splits')
        if split_dir.exists():
            with open(split_dir / 'train.json') as f:
                train = json.load(f)
            with open(split_dir / 'val.json') as f:
                val = json.load(f)
            with open(split_dir / 'test.json') as f:
                test = json.load(f)
            return train, val, test
    
    # Fallback: load all and return as single list for backward compatibility
    data_paths = [
        Path('data/all_training_data.json'),
        Path('data/merged_training_data.json'),
    ]
    
    samples = []
    
    for path in data_paths:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict) and 'samples' in data:
                samples.extend(data['samples'])
            elif isinstance(data, list):
                samples.extend(data)
            break
    
    for analysis_file in Path('data').rglob('*_analysis.json'):
        try:
            with open(analysis_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                for s in data:
                    if not any(s.get('from_track') == existing.get('from_track') and 
                              s.get('to_track') == existing.get('to_track') 
                              for existing in samples):
                        samples.append(s)
        except:
            pass
    
    # Return as single list for old API
    return samples, [], []


def train_decision_nn(train_samples: List[Dict],
                     val_samples: List[Dict],
                     epochs: int = 100,
                     batch_size: int = 16,
                     lr: float = 0.001) -> Tuple[DecisionNN, Dict]:
    """
    Train the DecisionNN model.
    
    Args:
        train_samples: Training samples
        val_samples: Validation samples
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    
    Returns:
        (trained_model, training_history)
    """
    print("\n" + "="*60)
    print("TRAINING DECISION NEURAL NETWORK")
    print("="*60)
    
    # Create datasets
    train_dataset = TransitionDataset(train_samples, include_curves=False)
    val_dataset = TransitionDataset(val_samples, include_curves=False)
    
    print(f"\nDataset: {len(train_samples)} train, {len(val_samples)} validation")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = DecisionNN(input_dim=13)  # Adjusted for our features
    model = model.to(device)
    
    # Loss and optimizer
    criterion = DecisionNNLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            features = batch['features'].to(device)
            key_a = batch['key_a'].to(device)
            key_b = batch['key_b'].to(device)
            
            targets = {
                'technique': batch['technique'].to(device),
                'bass_swap': batch['bass_swap'].to(device),
                'low_cut': batch['low_cut'].to(device),
                'high_cut': batch['high_cut'].to(device),
                'duration_bars': batch['duration_bars'].to(device),
            }
            
            optimizer.zero_grad()
            outputs = model(features, key_a, key_b)
            losses = criterion(outputs, targets)
            losses['total'].backward()
            optimizer.step()
            
            train_losses.append(losses['total'].item())
        
        # Validation
        model.eval()
        val_losses = []
        correct_technique = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                key_a = batch['key_a'].to(device)
                key_b = batch['key_b'].to(device)
                
                targets = {
                    'technique': batch['technique'].to(device),
                    'bass_swap': batch['bass_swap'].to(device),
                    'low_cut': batch['low_cut'].to(device),
                    'high_cut': batch['high_cut'].to(device),
                    'duration_bars': batch['duration_bars'].to(device),
                }
                
                outputs = model(features, key_a, key_b)
                losses = criterion(outputs, targets)
                val_losses.append(losses['total'].item())
                
                # Accuracy
                pred = outputs['technique_probs'].argmax(dim=1)
                correct_technique += (pred == targets['technique']).sum().item()
                total += len(pred)
        
        # Metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        accuracy = correct_technique / total if total > 0 else 0
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(accuracy)
        
        scheduler.step(avg_val_loss)
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, Accuracy={accuracy:.2%}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\nBest Validation Loss: {best_val_loss:.4f}")
    
    return model, history


def train_curve_lstm(train_samples: List[Dict],
                    val_samples: List[Dict],
                    context_provider: DecisionNN,
                    epochs: int = 100,
                    batch_size: int = 16,
                    lr: float = 0.001) -> Tuple[CurveLSTM, Dict]:
    """
    Train the CurveLSTM model.
    
    Args:
        train_samples: Training samples
        val_samples: Validation samples
        context_provider: Trained DecisionNN to provide context vectors
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
    
    Returns:
        (trained_model, training_history)
    """
    print("\n" + "="*60)
    print("TRAINING CURVE LSTM")
    print("="*60)
    
    # Create datasets with curves
    train_dataset = TransitionDataset(train_samples, include_curves=True)
    val_dataset = TransitionDataset(val_samples, include_curves=True)
    
    print(f"\nDataset: {len(train_samples)} train, {len(val_samples)} validation")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Models
    context_provider = context_provider.to(device)
    context_provider.eval()  # Freeze
    
    model = CurveLSTM(context_dim=32, song_feature_dim=13)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = CurveLSTMLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # History
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            features = batch['features'].to(device)
            key_a = batch['key_a'].to(device)
            key_b = batch['key_b'].to(device)
            target_curves = batch['curves'].to(device)
            
            # Get context from frozen DecisionNN
            with torch.no_grad():
                decision_out = context_provider(features, key_a, key_b)
                context = decision_out['context_vector']
            
            # Forward LSTM
            optimizer.zero_grad()
            lstm_out = model(context, features, duration_frames=target_curves.shape[1])
            
            losses = criterion(lstm_out['curves'], target_curves)
            losses['total'].backward()
            optimizer.step()
            
            train_losses.append(losses['total'].item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                key_a = batch['key_a'].to(device)
                key_b = batch['key_b'].to(device)
                target_curves = batch['curves'].to(device)
                
                decision_out = context_provider(features, key_a, key_b)
                context = decision_out['context_vector']
                
                lstm_out = model(context, features, duration_frames=target_curves.shape[1])
                losses = criterion(lstm_out['curves'], target_curves)
                val_losses.append(losses['total'].item())
        
        # Metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\nBest Validation Loss: {best_val_loss:.4f}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train AI DJ Model')
    parser.add_argument('--stage', '-s', type=int, choices=[1, 2, 3], default=1,
                       help='Training stage: 1=DecisionNN, 2=CurveLSTM, 3=End-to-end')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output', '-o', default='models/',
                       help='Output directory for saved models')
    
    args = parser.parse_args()
    
    # Load data with splits
    print("Loading training data...")
    train_samples, val_samples, test_samples = load_training_data(use_splits=True)
    
    if len(train_samples) == 0:
        # Fallback to old method
        all_samples = load_training_data(use_splits=False)
        if isinstance(all_samples, tuple):
            all_samples = all_samples[0]
        print(f"Loaded {len(all_samples)} samples (no splits found, will split randomly)")
        train_samples = all_samples
        val_samples = []
    else:
        print(f"Loaded: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
    
    # Check data requirements
    if args.stage == 1 and len(train_samples) < 100:
        print(f"\nWARNING: Only {len(train_samples)} training samples available.")
        print("Recommended minimum for DecisionNN: 300-500 samples")
        print("Training may overfit!")
    
    if args.stage >= 2 and len(train_samples) < 500:
        print(f"\nWARNING: Only {len(train_samples)} training samples available.")
        print("Recommended minimum for CurveLSTM: 2000+ samples")
        print("Consider collecting more data first.")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training
    if args.stage == 1:
        # Train DecisionNN only
        model, history = train_decision_nn(
            train_samples,
            val_samples if len(val_samples) > 0 else train_samples[:int(len(train_samples)*0.2)],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        
        # Save
        model_path = output_dir / 'decision_nn.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'num_train_samples': len(train_samples),
            'num_val_samples': len(val_samples) if len(val_samples) > 0 else int(len(train_samples)*0.2),
            'timestamp': datetime.now().isoformat()
        }, model_path)
        print(f"\nSaved DecisionNN to: {model_path}")
    
    elif args.stage == 2:
        # Load DecisionNN first
        nn_path = output_dir / 'decision_nn.pt'
        if not nn_path.exists():
            print("ERROR: Must train DecisionNN first (stage 1)")
            return
        
        checkpoint = torch.load(nn_path, map_location='cpu', weights_only=False)
        decision_nn = DecisionNN(input_dim=13)
        decision_nn.load_state_dict(checkpoint['model_state_dict'])
        
        # Train CurveLSTM
        if len(val_samples) == 0:
            print("ERROR: Need validation set for LSTM training")
            return
        
        model, history = train_curve_lstm(
            train_samples,
            val_samples,
            context_provider=decision_nn,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        
        # Save
        model_path = output_dir / 'curve_lstm.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'num_train_samples': len(train_samples),
            'num_val_samples': len(val_samples),
            'timestamp': datetime.now().isoformat()
        }, model_path)
        print(f"\nSaved CurveLSTM to: {model_path}")
    
    elif args.stage == 3:
        # End-to-end (not implemented yet)
        print("Stage 3 (end-to-end) training not yet implemented.")
        print("Train stages 1 and 2 separately for now.")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()

