"""
Combined DJ Model

End-to-end model that combines:
1. DecisionNN: Predicts transition technique and parameters
2. CurveLSTM: Generates automation curves

The context vector from DecisionNN is passed to CurveLSTM
to inform curve generation based on the predicted technique.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

from .decision_nn import DecisionNN, DecisionNNLoss
from .curve_lstm import CurveLSTM, CurveLSTMLoss


class CombinedDJModel(nn.Module):
    """
    Combined model for AI DJ transitions.
    
    Pipeline:
    1. Input song features -> DecisionNN -> technique, parameters, context
    2. Context + features -> CurveLSTM -> automation curves
    
    Can be trained:
    - Stage 1: DecisionNN only (needs less data)
    - Stage 2: CurveLSTM only (with frozen DecisionNN)
    - Stage 3: End-to-end fine-tuning
    """
    
    def __init__(self,
                 input_dim: int = 15,
                 hidden_dim: int = 64,
                 context_dim: int = 32,
                 lstm_hidden_dim: int = 128,
                 lstm_layers: int = 2,
                 num_techniques: int = 3,
                 num_curve_params: int = 10,
                 dropout: float = 0.3):
        super().__init__()
        
        # Decision Neural Network
        self.decision_nn = DecisionNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            num_techniques=num_techniques,
            dropout=dropout
        )
        
        # Curve LSTM
        self.curve_lstm = CurveLSTM(
            context_dim=context_dim,
            song_feature_dim=input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            num_output_params=num_curve_params,
            dropout=dropout
        )
        
        self.context_dim = context_dim
        self.input_dim = input_dim
    
    def forward(self,
                x: torch.Tensor,
                key_a: Optional[torch.Tensor] = None,
                key_b: Optional[torch.Tensor] = None,
                duration_frames: int = 100) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through both networks.
        
        Args:
            x: Input features [batch, input_dim]
            key_a: Key indices for song A [batch]
            key_b: Key indices for song B [batch]
            duration_frames: Number of curve frames to generate
        
        Returns:
            Combined outputs from both networks
        """
        # Decision NN forward
        decision_outputs = self.decision_nn(x, key_a, key_b)
        
        # Get context vector
        context = decision_outputs['context_vector']  # [batch, context_dim]
        
        # Use original features for LSTM (before key embedding)
        # If keys are separate, x should be [batch, input_dim - 2]
        # For simplicity, we'll pass x directly
        song_features = x[:, :self.input_dim] if x.shape[1] >= self.input_dim else x
        
        # Pad if necessary
        if song_features.shape[1] < self.input_dim:
            padding = torch.zeros(song_features.shape[0], 
                                 self.input_dim - song_features.shape[1],
                                 device=song_features.device)
            song_features = torch.cat([song_features, padding], dim=1)
        
        # Curve LSTM forward
        curve_outputs = self.curve_lstm(context, song_features, duration_frames)
        
        # Combine outputs
        return {
            **decision_outputs,
            'curves': curve_outputs['curves'],
            'curve_hidden': curve_outputs['hidden']
        }
    
    def forward_decision_only(self,
                             x: torch.Tensor,
                             key_a: Optional[torch.Tensor] = None,
                             key_b: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward through DecisionNN only."""
        return self.decision_nn(x, key_a, key_b)
    
    def forward_curves_only(self,
                           context: torch.Tensor,
                           song_features: torch.Tensor,
                           duration_frames: int = 100) -> Dict[str, torch.Tensor]:
        """Forward through CurveLSTM only."""
        return self.curve_lstm(context, song_features, duration_frames)
    
    def freeze_decision_nn(self):
        """Freeze DecisionNN parameters for curve-only training."""
        for param in self.decision_nn.parameters():
            param.requires_grad = False
    
    def unfreeze_decision_nn(self):
        """Unfreeze DecisionNN parameters."""
        for param in self.decision_nn.parameters():
            param.requires_grad = True
    
    def freeze_curve_lstm(self):
        """Freeze CurveLSTM parameters for decision-only training."""
        for param in self.curve_lstm.parameters():
            param.requires_grad = False
    
    def unfreeze_curve_lstm(self):
        """Unfreeze CurveLSTM parameters."""
        for param in self.curve_lstm.parameters():
            param.requires_grad = True
    
    def predict(self,
               x: torch.Tensor,
               key_a: Optional[torch.Tensor] = None,
               key_b: Optional[torch.Tensor] = None,
               duration_sec: float = 10.0,
               fps: int = 10) -> Dict:
        """
        Make predictions for inference.
        
        Returns human-readable predictions.
        """
        self.eval()
        
        duration_frames = int(duration_sec * fps)
        
        with torch.no_grad():
            outputs = self.forward(x, key_a, key_b, duration_frames)
            
            # Decision outputs
            technique_idx = outputs['technique_probs'].argmax(dim=1).cpu().numpy()
            technique_names = ['cut', 'blend', 'drop']
            
            # Curve outputs
            curves = outputs['curves'].cpu().numpy()
            
            result = {
                # Decisions
                'technique': [technique_names[i] for i in technique_idx],
                'bass_swap': (outputs['bass_swap'] > 0.5).cpu().numpy(),
                'low_cut': (outputs['low_cut'] > 0.5).cpu().numpy(),
                'high_cut': (outputs['high_cut'] > 0.5).cpu().numpy(),
                'duration_bars': outputs['duration_bars'].round().cpu().numpy().astype(int),
                
                # Curves
                'curves': {},
                'time': np.linspace(0, duration_sec, duration_frames)
            }
            
            # Unpack curves
            for i, name in enumerate(self.curve_lstm.PARAM_NAMES):
                result['curves'][name] = curves[:, :, i]
            
            return result


class CombinedLoss(nn.Module):
    """
    Combined loss for end-to-end training.
    """
    
    def __init__(self,
                 decision_weight: float = 1.0,
                 curve_weight: float = 1.0):
        super().__init__()
        
        self.decision_weight = decision_weight
        self.curve_weight = curve_weight
        
        self.decision_loss = DecisionNNLoss()
        self.curve_loss = CurveLSTMLoss()
    
    def forward(self,
                outputs: Dict[str, torch.Tensor],
                decision_targets: Dict[str, torch.Tensor],
                curve_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss.
        
        Args:
            outputs: Model outputs
            decision_targets: Targets for DecisionNN
            curve_targets: [batch, frames, params] targets for CurveLSTM
        """
        losses = {}
        
        # Decision losses
        decision_losses = self.decision_loss(outputs, decision_targets)
        for k, v in decision_losses.items():
            losses[f'decision_{k}'] = v * self.decision_weight
        
        # Curve losses (if targets provided)
        if curve_targets is not None and 'curves' in outputs:
            curve_losses = self.curve_loss(outputs['curves'], curve_targets)
            for k, v in curve_losses.items():
                losses[f'curve_{k}'] = v * self.curve_weight
        
        # Total
        losses['total'] = sum(v for k, v in losses.items() if k != 'total')
        
        return losses


def create_model(config: Optional[Dict] = None) -> CombinedDJModel:
    """
    Factory function to create model with config.
    
    Args:
        config: Optional configuration dict
    
    Returns:
        Configured CombinedDJModel
    """
    default_config = {
        'input_dim': 13,  # Features without key indices
        'hidden_dim': 64,
        'context_dim': 32,
        'lstm_hidden_dim': 128,
        'lstm_layers': 2,
        'num_techniques': 3,
        'num_curve_params': 10,
        'dropout': 0.3
    }
    
    if config:
        default_config.update(config)
    
    return CombinedDJModel(**default_config)


def save_model(model: CombinedDJModel, path: str, metadata: Optional[Dict] = None):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)


def load_model(path: str, config: Optional[Dict] = None) -> Tuple[CombinedDJModel, Dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint.get('metadata', {})

