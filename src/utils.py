"""Utility functions for audio analysis and data processing."""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


def compute_song_id(audio_path: str) -> str:
    """Compute a unique ID for a song based on file path and size."""
    file_path = Path(audio_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Use file path + size + modification time for ID
    stat = file_path.stat()
    content = f"{audio_path}{stat.st_size}{stat.st_mtime}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def save_analysis_json(analysis: Dict[str, Any], output_path: str) -> None:
    """Save analysis results to JSON file with proper formatting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    analysis_serializable = convert_numpy(analysis)
    
    with open(output_path, 'w') as f:
        json.dump(analysis_serializable, f, indent=2, ensure_ascii=False)


def load_analysis_json(json_path: str) -> Dict[str, Any]:
    """Load analysis results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_path(path: str) -> str:
    """Normalize file path for cross-platform compatibility."""
    return str(Path(path).resolve())


def get_audio_info(audio_path: str) -> Dict[str, Any]:
    """Get basic audio file information."""
    import librosa
    
    try:
        y, sr = librosa.load(audio_path, duration=1.0, sr=None)
        duration = librosa.get_duration(path=audio_path)
        
        file_path = Path(audio_path)
        stat = file_path.stat()
        
        return {
            "file_path": str(file_path.resolve()),
            "file_size_bytes": stat.st_size,
            "sample_rate": sr,
            "duration_sec": duration,
            "channels": 2 if len(y.shape) > 1 else 1
        }
    except Exception as e:
        return {"error": str(e)}


def compress_curve(values: np.ndarray, target_points: int = 100) -> Dict[str, list]:
    """Compress a dense curve to fewer points for storage."""
    if len(values) <= target_points:
        return {
            "values": values.tolist(),
            "compressed": False
        }
    
    # Downsample using decimation
    indices = np.linspace(0, len(values) - 1, target_points, dtype=int)
    compressed = values[indices]
    
    return {
        "values": compressed.tolist(),
        "indices": indices.tolist(),
        "original_length": len(values),
        "compressed": True
    }


def expand_curve(compressed_data: Dict[str, list]) -> np.ndarray:
    """Expand a compressed curve back to original length."""
    if not compressed_data.get("compressed", False):
        return np.array(compressed_data["values"])
    
    # Interpolate back to original length
    original_length = compressed_data["original_length"]
    compressed_values = np.array(compressed_data["values"])
    indices = np.array(compressed_data["indices"])
    
    expanded = np.interp(
        np.arange(original_length),
        indices,
        compressed_values
    )
    
    return expanded

