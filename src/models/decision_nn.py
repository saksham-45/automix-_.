"""
Decision Neural Network

Predicts transition decisions given song features:
- technique: cut/blend/drop
- bass_swap: yes/no
- duration_bars: 4-16
- eq_moves: multi-label
- context_vector: passed to LSTM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List


class DecisionNN(nn.Module):
    """
    Neural Network for transition decision making.
    
    Input Features (15):
        - tempo_A, tempo_B (normalized 0-1)
        - key_A, key_B (one-hot encoded, 12 dims each -> or embedded)
        - energy_A, energy_B (0-1)
        - key_compatible (0/1)
        - harmonic_tension (0-1)
        - spectral_smoothness (0-1)
        - bass_energy_A, bass_energy_B (0-1)
        - tempo_diff (normalized)
        - energy_diff (normalized)
    
    Outputs:
        - technique: 3 classes (cut, blend, drop)
        - bass_swap: 1 (sigmoid)
        - duration_bars: 1 (regression, 4-16)
        - low_cut_incoming: 1 (sigmoid)
        - high_cut_outgoing: 1 (sigmoid)
        - context_vector: 32 dims (for LSTM)
    """
    
    def __init__(self, 
                 input_dim: int = 15,
                 hidden_dim: int = 64,
                 context_dim: int = 32,
                 num_techniques: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        
        # Key embedding (12 possible keys)
        self.key_embedding = nn.Embedding(12, 8)
        
        # Adjusted input dim after key embedding
        # Input features (without keys) + 2 key embeddings (8 dims each)
        # prepare_features returns 13 features (no keys), keys are passed separately
        adjusted_input = input_dim + 16  # 13 + 16 = 29
        
        # Shared hidden layers
        self.fc1 = nn.Linear(adjusted_input, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Output heads
        # Technique classification (cut, blend, drop)
        self.technique_head = nn.Linear(hidden_dim // 2, num_techniques)
        
        # Binary decisions (sigmoid outputs)
        self.bass_swap_head = nn.Linear(hidden_dim // 2, 1)
        self.low_cut_head = nn.Linear(hidden_dim // 2, 1)
        self.high_cut_head = nn.Linear(hidden_dim // 2, 1)
        
        # Duration regression (4-16 bars)
        self.duration_head = nn.Linear(hidden_dim // 2, 1)
        
        # Context vector for LSTM
        self.context_head = nn.Linear(hidden_dim // 2, context_dim)
    
    def forward(self, x: torch.Tensor, 
                key_a: torch.Tensor = None, 
                key_b: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [batch, input_dim] or [batch, input_dim - 2] if keys separate
            key_a: Key indices for song A [batch] (0-11)
            key_b: Key indices for song B [batch] (0-11)
        
        Returns:
            Dictionary with all outputs
        """
        batch_size = x.shape[0]
        
        # Handle key embeddings if provided separately
        if key_a is not None and key_b is not None:
            key_a_emb = self.key_embedding(key_a)  # [batch, 8]
            key_b_emb = self.key_embedding(key_b)  # [batch, 8]
            x = torch.cat([x, key_a_emb, key_b_emb], dim=1)
        
        # Hidden layers
        h = self.fc1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout1(h)
        
        h = self.fc2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.dropout2(h)
        
        # Output heads
        technique_logits = self.technique_head(h)  # [batch, 3]
        
        bass_swap = torch.sigmoid(self.bass_swap_head(h))  # [batch, 1]
        low_cut = torch.sigmoid(self.low_cut_head(h))  # [batch, 1]
        high_cut = torch.sigmoid(self.high_cut_head(h))  # [batch, 1]
        
        # Duration: scale to 4-16 range
        duration_raw = self.duration_head(h)  # [batch, 1]
        duration = 4 + 12 * torch.sigmoid(duration_raw)  # [batch, 1] in [4, 16]
        
        # Context vector for LSTM
        context = self.context_head(h)  # [batch, context_dim]
        
        return {
            'technique_logits': technique_logits,
            'technique_probs': F.softmax(technique_logits, dim=1),
            'bass_swap': bass_swap.squeeze(-1),
            'low_cut': low_cut.squeeze(-1),
            'high_cut': high_cut.squeeze(-1),
            'duration_bars': duration.squeeze(-1),
            'context_vector': context
        }
    
    def predict(self, x: torch.Tensor, 
                key_a: torch.Tensor = None,
                key_b: torch.Tensor = None) -> Dict[str, np.ndarray]:
        """
        Make predictions (inference mode).
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, key_a, key_b)
            
            # Convert to numpy and apply thresholds
            technique_idx = outputs['technique_probs'].argmax(dim=1).cpu().numpy()
            technique_names = ['cut', 'blend', 'drop']
            
            return {
                'technique': [technique_names[i] for i in technique_idx],
                'bass_swap': (outputs['bass_swap'] > 0.5).cpu().numpy(),
                'low_cut': (outputs['low_cut'] > 0.5).cpu().numpy(),
                'high_cut': (outputs['high_cut'] > 0.5).cpu().numpy(),
                'duration_bars': outputs['duration_bars'].round().cpu().numpy().astype(int),
                'context_vector': outputs['context_vector'].cpu().numpy()
            }


class DecisionNNLoss(nn.Module):
    """
    Combined loss function for DecisionNN.
    """
    
    def __init__(self, 
                 technique_weight: float = 1.0,
                 binary_weight: float = 1.0,
                 duration_weight: float = 0.5):
        super().__init__()
        
        self.technique_weight = technique_weight
        self.binary_weight = binary_weight
        self.duration_weight = duration_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss.
        
        Args:
            outputs: Model outputs
            targets: Ground truth labels
                - technique: [batch] int labels (0, 1, 2)
                - bass_swap: [batch] float (0 or 1)
                - low_cut: [batch] float (0 or 1)
                - high_cut: [batch] float (0 or 1)
                - duration_bars: [batch] float (4-16)
        """
        losses = {}
        
        # Technique classification loss
        if 'technique' in targets:
            losses['technique'] = self.ce_loss(
                outputs['technique_logits'], 
                targets['technique']
            ) * self.technique_weight
        
        # Binary losses
        if 'bass_swap' in targets:
            losses['bass_swap'] = self.bce_loss(
                outputs['bass_swap'], 
                targets['bass_swap']
            ) * self.binary_weight
        
        if 'low_cut' in targets:
            losses['low_cut'] = self.bce_loss(
                outputs['low_cut'], 
                targets['low_cut']
            ) * self.binary_weight
        
        if 'high_cut' in targets:
            losses['high_cut'] = self.bce_loss(
                outputs['high_cut'], 
                targets['high_cut']
            ) * self.binary_weight
        
        # Duration regression loss
        if 'duration_bars' in targets:
            losses['duration'] = self.mse_loss(
                outputs['duration_bars'], 
                targets['duration_bars']
            ) * self.duration_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


def prepare_features(sample: Dict) -> Tuple[np.ndarray, int, int]:
    """
    Prepare features from a training sample.
    
    Args:
        sample: Training sample dict from analysis
    
    Returns:
        (features_array, key_a_idx, key_b_idx)
    """
    # Key mapping
    key_to_idx = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }
    
    # Extract tempo (handle string format)
    def parse_tempo(t):
        if isinstance(t, str):
            return float(t.strip('[]'))
        elif isinstance(t, list):
            return t[0]
        return float(t)
    
    tempo_a = parse_tempo(sample.get('tempo_outgoing', 120))
    tempo_b = parse_tempo(sample.get('tempo_incoming', 120))
    
    # Normalize tempo to 0-1 (assuming 60-180 BPM range)
    tempo_a_norm = (tempo_a - 60) / 120
    tempo_b_norm = (tempo_b - 60) / 120
    
    features = np.array([
        tempo_a_norm,
        tempo_b_norm,
        sample.get('energy_before', 0.5),
        sample.get('energy_after', 0.5),
        1.0 if sample.get('key_compatible') else 0.0,
        sample.get('harmonic_tension', 0.1),
        sample.get('spectral_smoothness', 0.8),
        sample.get('frequency_masking', 0.1),
        # Energy difference
        abs(sample.get('energy_after', 0.5) - sample.get('energy_before', 0.5)),
        # Tempo difference (normalized)
        abs(tempo_a_norm - tempo_b_norm),
        # Energy during transition
        sample.get('energy_during', 0.5),
        # Additional context
        1.0 if sample.get('energy_build') else 0.0,
        1.0 if sample.get('energy_dip') else 0.0,
    ], dtype=np.float32)
    
    # Get key indices
    key_a = sample.get('key_outgoing', 'C')
    key_b = sample.get('key_incoming', 'C')
    key_a_idx = key_to_idx.get(key_a, 0)
    key_b_idx = key_to_idx.get(key_b, 0)
    
    return features, key_a_idx, key_b_idx


def prepare_targets(sample: Dict) -> Dict[str, np.ndarray]:
    """
    Prepare target labels from a training sample.
    """
    # Technique mapping
    technique_map = {'cut': 0, 'blend': 1, 'linear': 1, 'drop': 2}
    technique = sample.get('crossfade_type', 'cut')
    technique_idx = technique_map.get(technique, 0)
    
    # Binary targets
    bass_swap = sample.get('bass_swap_detected') in [True, 'True']
    low_cut = sample.get('low_cut_on_incoming') in [True, 'True']
    high_cut = sample.get('high_cut_on_outgoing') in [True, 'True']
    
    # Duration
    duration = sample.get('bars_duration', 5)
    
    return {
        'technique': technique_idx,
        'bass_swap': float(bass_swap),
        'low_cut': float(low_cut),
        'high_cut': float(high_cut),
        'duration_bars': float(duration)
    }

