"""
Stem Separator Module

Separates audio into stems (drums, bass, vocals, other) using demucs.
Optimized for transition segments to prevent heavy instruments from carrying over.
"""
import numpy as np
import torch
from typing import Dict, Optional, Tuple
import warnings
from src.utils import get_best_device, get_device_name
warnings.filterwarnings('ignore')


class StemSeparator:
    """
    Separates audio into stems using demucs library.
    Optimized for short segments (transition regions).
    """
    
    def __init__(self, model_name: str = "htdemucs", device: Optional[str] = None):
        """
        Initialize stem separator.
        
        Args:
            model_name: demucs model to use ('htdemucs', 'htdemucs_ft', 'mdx_extra')
            device: Device to use ('mps', 'cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = get_best_device(device)
        self.model = None
        self._model_loaded = False
        
    def _load_model(self):
        """Lazy load the demucs model."""
        if self._model_loaded:
            return
            
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            self.get_model = get_model
            self.apply_model = apply_model
            self._model_loaded = True
            device_name = get_device_name(self.device)
            print(f"  ✓ Stem separation model ready ({self.model_name}) on {device_name}")
        except ImportError:
            raise ImportError(
                "demucs not installed. Install with: pip install demucs"
            )
    
    def separate_stems(self, 
                     audio: np.ndarray, 
                     sr: int = 44100) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems.
        
        Args:
            audio: Audio array (mono or stereo)
            sr: Sample rate
            
        Returns:
            Dictionary with keys: 'drums', 'bass', 'vocals', 'other'
            Each value is a numpy array of the same shape as input
        """
        self._load_model()
        
        # Ensure stereo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        
        try:
            # Load model (cached after first load)
            if self.model is None:
                self.model = self.get_model(self.model_name)
                self.model.to(self.device)
                self.model.eval()
            
            # Convert audio to tensor format expected by demucs
            # demucs expects: [channels, samples] -> [1, channels, samples]
            wav = torch.from_numpy(audio.T).float()  # [channels, samples]
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)  # Add channel dimension if mono
            wav = wav.unsqueeze(0)  # Add batch dimension: [1, channels, samples]
            wav = wav.to(self.device)
            
            # Apply model
            with torch.no_grad():
                sources = self.apply_model(
                    self.model, 
                    wav, 
                    shifts=1,  # Single pass for speed
                    split=True, 
                    overlap=0.25, 
                    progress=False
                )
            
            # Extract stems
            # sources shape: [batch, sources, channels, samples]
            # Sources order: ['drums', 'bass', 'other', 'vocals']
            sources = sources[0].cpu().numpy()  # Remove batch dimension: [sources, channels, samples]
            
            # Convert to [samples, channels] format
            stems = {
                'drums': sources[0].T,   # [samples, channels]
                'bass': sources[1].T,
                'other': sources[2].T,
                'vocals': sources[3].T
            }
            
            # Ensure stereo format [samples, channels]
            for key in stems:
                if stems[key].ndim == 1:
                    stems[key] = np.column_stack([stems[key], stems[key]])
                elif stems[key].shape[1] == 1:
                    # Mono - duplicate to stereo
                    stems[key] = np.column_stack([stems[key][:, 0], stems[key][:, 0]])
            
            return stems
            
        except Exception as e:
            print(f"  ⚠ Stem separation failed: {e}")
            print(f"  → Falling back to original audio")
            # Return original audio split equally (fallback)
            return {
                'drums': audio * 0.3,
                'bass': audio * 0.2,
                'other': audio * 0.4,
                'vocals': audio * 0.1
            }
    
    def separate_segment(self, 
                        segment: np.ndarray, 
                        sr: int = 44100) -> Dict[str, np.ndarray]:
        """
        Separate a transition segment (optimized for short segments).
        
        Args:
            segment: Audio segment (typically 16 seconds)
            sr: Sample rate
            
        Returns:
            Dictionary with separated stems
        """
        return self.separate_stems(segment, sr)
    
    def recombine_stems(self, 
                       stems: Dict[str, np.ndarray],
                       include_stems: Optional[list] = None) -> np.ndarray:
        """
        Recombine stems into full audio.
        
        Args:
            stems: Dictionary of stems
            include_stems: List of stem names to include (None = all)
            
        Returns:
            Recombined audio array
        """
        if include_stems is None:
            include_stems = list(stems.keys())
        
        result = np.zeros_like(stems[include_stems[0]])
        for stem_name in include_stems:
            if stem_name in stems:
                result += stems[stem_name]
        
        return result
