"""
Curve Generation LSTM

Generates frame-by-frame mixer automation curves:
- Volume A, Volume B
- Bass A, Bass B
- Mid A, Mid B
- High A, High B
- Filter frequency
- Filter resonance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class CurveLSTM(nn.Module):
    """
    LSTM for generating transition automation curves.
    
    Takes context from DecisionNN and song features,
    outputs frame-by-frame mixer parameters.
    
    Input:
        - context_vector: [batch, 32] from DecisionNN
        - song_features: [batch, 15] normalized song features
        - duration_frames: int, number of output frames (default 100 = 10 sec)
    
    Output:
        - curves: [batch, duration_frames, num_params]
            - volume_a, volume_b: 0-1
            - bass_a, bass_b: 0-1 (EQ gain normalized)
            - mid_a, mid_b: 0-1
            - high_a, high_b: 0-1
            - filter_freq: 0-1 (mapped to 20-20000 Hz)
            - filter_res: 0-1
    """
    
    # Output parameters
    PARAM_NAMES = [
        'volume_a', 'volume_b',
        'bass_a', 'bass_b',
        'mid_a', 'mid_b',
        'high_a', 'high_b',
        'filter_freq', 'filter_res'
    ]
    
    def __init__(self,
                 context_dim: int = 32,
                 song_feature_dim: int = 15,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_output_params: int = 10,
                 dropout: float = 0.2):
        super().__init__()
        
        self.context_dim = context_dim
        self.song_feature_dim = song_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_output_params = num_output_params
        
        # Input projection
        input_dim = context_dim + song_feature_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_output_params)
        )
        
        # Learned initial curve bias (typical starting point)
        self.initial_bias = nn.Parameter(torch.tensor([
            1.0, 0.0,  # volume_a starts at 1, volume_b at 0
            0.5, 0.5,  # bass neutral
            0.5, 0.5,  # mid neutral
            0.5, 0.5,  # high neutral
            0.5, 0.3   # filter mid-range, low resonance
        ]))
    
    def forward(self, 
                context: torch.Tensor,
                song_features: torch.Tensor,
                duration_frames: int = 100,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Dict[str, torch.Tensor]:
        """
        Generate automation curves.
        
        Args:
            context: Context vector from DecisionNN [batch, context_dim]
            song_features: Song feature vector [batch, song_feature_dim]
            duration_frames: Number of frames to generate
            hidden: Optional initial hidden state
        
        Returns:
            Dictionary with curves and hidden states
        """
        batch_size = context.shape[0]
        device = context.device
        
        # Combine context and song features
        combined = torch.cat([context, song_features], dim=1)  # [batch, input_dim]
        
        # Project to hidden dim
        x = self.input_proj(combined)  # [batch, hidden_dim]
        
        # Expand for sequence
        x = x.unsqueeze(1).expand(-1, duration_frames, -1)  # [batch, seq_len, hidden_dim]
        
        # Add positional encoding (time progress 0-1)
        time_progress = torch.linspace(0, 1, duration_frames, device=device)
        time_progress = time_progress.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        time_encoding = time_progress.expand(batch_size, -1, 1)
        
        # Concatenate time encoding (need to adjust input_proj or add it differently)
        # For simplicity, we'll modulate x with time
        x = x * (0.5 + 0.5 * time_encoding)  # Modulate based on progress
        
        # LSTM forward
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # Generate output curves
        raw_curves = self.output_proj(lstm_out)  # [batch, seq_len, num_params]
        
        # Apply sigmoid to bound outputs to [0, 1]
        curves = torch.sigmoid(raw_curves + self.initial_bias)
        
        # Apply smoothness constraint (curves should be smooth)
        # This is done via the loss function, not here
        
        return {
            'curves': curves,  # [batch, duration_frames, num_params]
            'hidden': hidden,
            'param_names': self.PARAM_NAMES
        }
    
    def generate_curves(self,
                       context: torch.Tensor,
                       song_features: torch.Tensor,
                       duration_sec: float = 10.0,
                       fps: int = 10) -> Dict[str, np.ndarray]:
        """
        Generate curves for inference.
        
        Args:
            context: Context from DecisionNN
            song_features: Song features
            duration_sec: Duration in seconds
            fps: Frames per second
        
        Returns:
            Dictionary mapping param names to numpy curves
        """
        self.eval()
        
        duration_frames = int(duration_sec * fps)
        
        with torch.no_grad():
            outputs = self.forward(context, song_features, duration_frames)
            curves = outputs['curves'].cpu().numpy()  # [batch, frames, params]
        
        # Convert to dict of curves
        result = {}
        for i, name in enumerate(self.PARAM_NAMES):
            result[name] = curves[:, :, i]  # [batch, frames]
        
        # Add time axis
        result['time'] = np.linspace(0, duration_sec, duration_frames)
        
        return result
    
    def curves_to_mixer_params(self, curves: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert normalized curves to actual mixer parameters.
        
        Args:
            curves: Dict of normalized curves (0-1)
        
        Returns:
            Dict with actual parameter values
        """
        return {
            'volume_a_db': curves['volume_a'] * -60,  # 0 to -60 dB
            'volume_b_db': curves['volume_b'] * -60,
            'bass_a_db': (curves['bass_a'] - 0.5) * 24,  # -12 to +12 dB
            'bass_b_db': (curves['bass_b'] - 0.5) * 24,
            'mid_a_db': (curves['mid_a'] - 0.5) * 24,
            'mid_b_db': (curves['mid_b'] - 0.5) * 24,
            'high_a_db': (curves['high_a'] - 0.5) * 24,
            'high_b_db': (curves['high_b'] - 0.5) * 24,
            'filter_freq_hz': 20 * np.power(1000, curves['filter_freq']),  # 20-20000 Hz log scale
            'filter_res': curves['filter_res'],
            'time': curves['time']
        }


class CurveLSTMLoss(nn.Module):
    """
    Loss function for curve generation.
    
    Combines:
    - MSE for curve values
    - Smoothness penalty (penalize sudden changes)
    - Boundary constraints (volume crossfade should complete)
    """
    
    def __init__(self,
                 mse_weight: float = 1.0,
                 smoothness_weight: float = 0.1,
                 boundary_weight: float = 0.5):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.smoothness_weight = smoothness_weight
        self.boundary_weight = boundary_weight
    
    def forward(self, 
                predicted: torch.Tensor,
                target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate curve generation loss.
        
        Args:
            predicted: [batch, frames, params] predicted curves
            target: [batch, frames, params] target curves
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # MSE loss on curve values
        losses['mse'] = F.mse_loss(predicted, target) * self.mse_weight
        
        # Smoothness loss: penalize large frame-to-frame changes
        pred_diff = predicted[:, 1:, :] - predicted[:, :-1, :]
        target_diff = target[:, 1:, :] - target[:, :-1, :]
        losses['smoothness'] = F.mse_loss(pred_diff, target_diff) * self.smoothness_weight
        
        # Boundary loss: volume_a should end near 0, volume_b should end near 1
        # (for a standard crossfade)
        volume_a_end = predicted[:, -1, 0]  # volume_a at end
        volume_b_end = predicted[:, -1, 1]  # volume_b at end
        volume_a_start = predicted[:, 0, 0]  # volume_a at start
        volume_b_start = predicted[:, 0, 1]  # volume_b at start
        
        # Ideal: volume_a goes 1->0, volume_b goes 0->1
        boundary_loss = (
            (volume_a_start - 1.0).pow(2).mean() +
            (volume_a_end - 0.0).pow(2).mean() +
            (volume_b_start - 0.0).pow(2).mean() +
            (volume_b_end - 1.0).pow(2).mean()
        )
        losses['boundary'] = boundary_loss * self.boundary_weight
        
        # Total
        losses['total'] = sum(losses.values())
        
        return losses


def prepare_curve_targets(sample: Dict, num_frames: int = 100) -> np.ndarray:
    """
    Prepare target curves from a training sample.
    
    Args:
        sample: Training sample with volume/EQ curves
        num_frames: Number of frames to output
    
    Returns:
        [num_frames, num_params] array
    """
    num_params = len(CurveLSTM.PARAM_NAMES)
    curves = np.zeros((num_frames, num_params))
    
    # Default crossfade pattern (linear)
    t = np.linspace(0, 1, num_frames)
    
    # Volume curves
    if 'volume_curve_outgoing' in sample and sample['volume_curve_outgoing']:
        vol_out = np.array(sample['volume_curve_outgoing'])
        if len(vol_out) > 0:
            # Normalize to 0-1
            vol_out_values = vol_out[:, 1] if vol_out.ndim > 1 else vol_out  # dB values
            vol_out_norm = np.clip((vol_out_values + 60) / 60, 0, 1)  # -60 to 0 dB -> 0 to 1
            # Always interpolate to num_frames
            curves[:, 0] = np.interp(
                np.linspace(0, 1, num_frames),
                np.linspace(0, 1, len(vol_out_norm)),
                vol_out_norm
            )
    else:
        curves[:, 0] = 1 - t  # Linear fade out
    
    if 'volume_curve_incoming' in sample and sample['volume_curve_incoming']:
        vol_in = np.array(sample['volume_curve_incoming'])
        if len(vol_in) > 0:
            vol_in_values = vol_in[:, 1] if vol_in.ndim > 1 else vol_in
            vol_in_norm = np.clip((vol_in_values + 60) / 60, 0, 1)
            # Always interpolate to num_frames
            curves[:, 1] = np.interp(
                np.linspace(0, 1, num_frames),
                np.linspace(0, 1, len(vol_in_norm)),
                vol_in_norm
            )
    else:
        curves[:, 1] = t  # Linear fade in
    
    # EQ curves from bass/mid/high energy
    if 'bass_energy_curve' in sample and sample['bass_energy_curve']:
        bass = np.array(sample['bass_energy_curve'])
        if len(bass) > 0:
            bass_values = bass[:, 1]
            bass_interp = np.interp(
                np.linspace(0, 1, num_frames),
                np.linspace(0, 1, len(bass_values)),
                bass_values
            )
            # Split into A and B based on crossfade progress
            curves[:, 2] = bass_interp * (1 - t)  # bass_a
            curves[:, 3] = bass_interp * t  # bass_b
    else:
        curves[:, 2] = 0.5 * (1 - t)
        curves[:, 3] = 0.5 * t
    
    # Mid curves
    if 'mid_energy_curve' in sample and sample['mid_energy_curve']:
        mid = np.array(sample['mid_energy_curve'])
        if len(mid) > 0:
            mid_values = mid[:, 1]
            mid_interp = np.interp(
                np.linspace(0, 1, num_frames),
                np.linspace(0, 1, len(mid_values)),
                mid_values
            )
            curves[:, 4] = mid_interp * (1 - t)
            curves[:, 5] = mid_interp * t
    else:
        curves[:, 4] = 0.5 * (1 - t)
        curves[:, 5] = 0.5 * t
    
    # High curves
    if 'high_energy_curve' in sample and sample['high_energy_curve']:
        high = np.array(sample['high_energy_curve'])
        if len(high) > 0:
            high_values = high[:, 1]
            high_interp = np.interp(
                np.linspace(0, 1, num_frames),
                np.linspace(0, 1, len(high_values)),
                high_values
            )
            curves[:, 6] = high_interp * (1 - t)
            curves[:, 7] = high_interp * t
    else:
        curves[:, 6] = 0.5 * (1 - t)
        curves[:, 7] = 0.5 * t
    
    # Filter (default: sweep from low to high during transition)
    curves[:, 8] = t  # filter_freq: 0->1 (low to high)
    curves[:, 9] = 0.3 * np.ones(num_frames)  # filter_res: constant low
    
    return curves.astype(np.float32)

