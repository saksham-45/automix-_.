"""
Stem Separator Module

Separates audio into stems (drums, bass, vocals, other) using demucs.
Optimized for transition segments to prevent heavy instruments from carrying over.
"""
import numpy as np
import torch
import os
import hashlib
from collections import OrderedDict
from pathlib import Path
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
        # ── Stem cache ──────────────────────────────────────────────────────
        # Separation is the most expensive step; identical segments (look-ahead
        # pre-render, re-renders, repeated tracks) must never be recomputed. We
        # key by content hash + sr + model, with an in-memory LRU backed by an
        # on-disk .npz cache. This is a pure efficiency win — same audio out.
        self.cache_enabled = True
        self._mem_cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        self._mem_cache_cap = 64
        self._disk_cache_dir = Path(
            os.getenv("STEM_CACHE_DIR", Path(__file__).resolve().parent.parent / "data" / "cache" / "stems")
        )
        self.cache_stats = {"hits": 0, "misses": 0, "silence": 0}

    def _cache_key(self, audio: np.ndarray, sr: int) -> str:
        h = hashlib.md5()
        h.update(np.ascontiguousarray(audio, dtype=np.float32).tobytes())
        h.update(f"|{sr}|{self.model_name}".encode())
        return h.hexdigest()

    def _cache_get(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        if not self.cache_enabled:
            return None
        if key in self._mem_cache:
            self._mem_cache.move_to_end(key)
            self.cache_stats["hits"] += 1
            return {k: v.copy() for k, v in self._mem_cache[key].items()}
        p = self._disk_cache_dir / f"{key}.npz"
        if p.exists():
            try:
                data = np.load(p)
                stems = {k: data[k] for k in data.files}
                self._mem_put(key, stems)
                self.cache_stats["hits"] += 1
                return {k: v.copy() for k, v in stems.items()}
            except Exception:
                pass
        return None

    def _mem_put(self, key: str, stems: Dict[str, np.ndarray]):
        self._mem_cache[key] = {k: v.copy() for k, v in stems.items()}
        self._mem_cache.move_to_end(key)
        while len(self._mem_cache) > self._mem_cache_cap:
            self._mem_cache.popitem(last=False)

    def _cache_put(self, key: str, stems: Dict[str, np.ndarray]):
        if not self.cache_enabled:
            return
        self._mem_put(key, stems)
        try:
            self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(self._disk_cache_dir / f"{key}.npz",
                     **{k: np.asarray(v, dtype=np.float32) for k, v in stems.items()})
        except Exception:
            pass
        
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
        # Normalize to stereo float32 (consistent for hashing AND processing).
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        # Silence-skip: near-silent input never needs the model.
        if float(np.max(np.abs(audio))) < 1e-5:
            self.cache_stats["silence"] += 1
            z = np.zeros_like(audio)
            return {'drums': z.copy(), 'bass': z.copy(), 'other': z.copy(), 'vocals': z.copy()}

        # Cache lookup (memory -> disk). Pure efficiency: identical audio in => identical stems out.
        key = self._cache_key(audio, sr)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        self.cache_stats["misses"] += 1
        self._load_model()
        try:
            stems = self._separate_compute(audio)
            self._cache_put(key, stems)
            return stems
        except Exception as e:
            print(f"  ⚠ Stem separation failed: {e}")
            print(f"  → Falling back to original audio as a single 'other' stem")
            # Put the whole mix in ONE stem and zero the rest (do NOT cache the
            # fallback). The previous fallback returned scaled copies of the FULL
            # mix in every stem -> quadruple-layered/comb-filtered mush and per-stem
            # muting did nothing. 'other' is never specially faded, so this behaves
            # like a plain crossfade downstream.
            zeros = np.zeros_like(audio)
            return {
                'drums': zeros.copy(),
                'bass': zeros.copy(),
                'other': audio.copy(),
                'vocals': zeros.copy(),
            }

    def _separate_compute(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Run demucs on stereo float32 [samples, channels]. Raises on failure."""
        # Load model (cached after first load), on the best device (MPS/CUDA/CPU).
        if self.model is None:
            self.model = self.get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()

        # demucs expects [batch, channels, samples]
        wav = torch.from_numpy(audio.T).float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        wav = wav.unsqueeze(0).to(self.device)

        with torch.no_grad():
            sources = self.apply_model(
                self.model, wav,
                shifts=1, split=True, overlap=0.25, progress=False
            )

        # sources: [batch, source, channels, samples]; order drums/bass/other/vocals
        sources = sources[0].cpu().numpy()
        stems = {
            'drums': sources[0].T,
            'bass': sources[1].T,
            'other': sources[2].T,
            'vocals': sources[3].T,
        }
        for key in stems:
            if stems[key].ndim == 1:
                stems[key] = np.column_stack([stems[key], stems[key]])
            elif stems[key].shape[1] == 1:
                stems[key] = np.column_stack([stems[key][:, 0], stems[key][:, 0]])
        return stems
    
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
