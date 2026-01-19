#!/usr/bin/env python3
"""
Fast analyzer for transition segments.
Optimized for 30-60s segments instead of full songs.
"""
import librosa
import numpy as np
from typing import Dict, Optional


class FastSegmentAnalyzer:
    """
    Lightweight analyzer for transition segments.
    Designed for real-time analysis of 30-60s segments.
    
    Key differences from full song analysis:
    - Only analyzes what's needed for transitions (key, tempo, energy, basic structure)
    - Uses smaller analysis windows (15-30s instead of full song)
    - Skips expensive operations (full structure analysis, embeddings)
    - Compatible with existing SmartMixer._analyze_song_fast() format
    """
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512, sr: Optional[int] = None):
        """
        Initialize fast segment analyzer.
        
        Args:
            sample_rate: Sample rate (default: 44100)
            hop_length: Hop length for analysis (default: 512)
            sr: Alias for sample_rate (for compatibility)
        """
        self.sr = sample_rate if sr is None else sr
        self.hop_length = hop_length
        
        # Import only what we need (lightweight)
        from src.harmonic_analyzer import HarmonicAnalyzer
        from src.structure_analyzer import StructureAnalyzer
        
        self.harmonic_analyzer = HarmonicAnalyzer()
        # Use smaller window for structure analysis
        self.structure_analyzer = StructureAnalyzer(sr=sample_rate, hop_length=hop_length)
    
    def analyze_segment(self, segment_audio: np.ndarray, 
                       segment_path: Optional[str] = None,
                       cache_key: Optional[str] = None) -> Dict:
        """
        Fast analysis of a transition segment.
        
        Only extracts what's needed for transitions:
        - Key (harmonic compatibility)
        - Tempo (beat matching)
        - Energy (transition selection)
        - Basic structure points (phrase boundaries)
        
        Args:
            segment_audio: Audio array (30-60s segment)
            segment_path: Optional path for caching (not used yet)
            cache_key: Optional cache key (not used yet, for future caching)
            
        Returns:
            Simplified analysis dict (compatible with existing _analyze_song_fast format)
            Format: {'key': str, 'tempo': float, 'structure': dict, 'energy': float}
        """
        duration = len(segment_audio) / self.sr
        
        # 1. Fast key detection (uses first 15s for speed)
        key_sample = segment_audio[:min(len(segment_audio), int(15 * self.sr))]
        print("    Detecting key...")
        try:
            key_info = self.harmonic_analyzer.detect_key_camelot(key_sample)
            key = key_info.get('key', 'C')
        except Exception as e:
            print(f"    Warning: Key detection failed: {e}")
            key = 'C'
        
        # 2. Fast tempo detection (uses first 30s or full segment)
        tempo_sample = segment_audio[:min(len(segment_audio), int(30 * self.sr))]
        print("    Detecting tempo...")
        try:
            tempo, beats = librosa.beat.beat_track(y=tempo_sample, sr=self.sr, hop_length=self.hop_length)
            tempo = float(tempo)
        except Exception as e:
            print(f"    Warning: Tempo detection failed: {e}")
            tempo = 120.0
        
        # 3. Basic structure (find transition points only)
        print("    Finding structure points...")
        try:
            # For segments, we only need to know:
            # - Where to mix OUT (if outgoing) -> last 30s of segment
            # - Where to mix IN (if incoming) -> first 30s of segment
            structure = {
                'best_mix_out_points': [max(0, duration - 30)],
                'best_mix_in_points': [min(30, duration * 0.1)],
                'sections': [],
                'phrases': []
            }
            
            # If segment is long enough, try to find phrase boundaries
            if duration > 30:
                # Use structure analyzer but with reduced search
                struct_result = self.structure_analyzer.analyze_structure(segment_audio)
                
                # Extract only relevant points
                mix_out = struct_result.get('best_mix_out_points', [])
                mix_in = struct_result.get('best_mix_in_points', [])
                
                if isinstance(mix_out, np.ndarray):
                    mix_out = mix_out.tolist()
                if isinstance(mix_in, np.ndarray):
                    mix_in = mix_in.tolist()
                
                if mix_out:
                    # Scale to segment coordinates (they're already relative)
                    # Take top 3 points, ensure they're valid
                    structure['best_mix_out_points'] = [
                        min(float(p), duration - 10) for p in mix_out[:3]  # Top 3 points
                    ]
                if mix_in:
                    structure['best_mix_in_points'] = [
                        min(float(p), duration * 0.5) for p in mix_in[:3]  # Top 3 points
                    ]
        except Exception as e:
            print(f"    Warning: Structure analysis failed: {e}")
            # Use defaults
            duration_sec = len(segment_audio) / self.sr
            structure = {
                'best_mix_out_points': [max(0, duration_sec - 30)],
                'best_mix_in_points': [min(30, duration_sec * 0.1)],
                'sections': [],
                'phrases': []
            }
        
        # 4. Energy (simple RMS on first 10s)
        energy_sample = segment_audio[:min(len(segment_audio), int(10 * self.sr))]
        energy = float(np.mean(np.abs(energy_sample) ** 2))
        
        return {
            'key': key,
            'tempo': tempo,
            'structure': structure,
            'energy': energy
        }
    
    def analyze_segment_file(self, segment_path: str, 
                            cache_key: Optional[str] = None) -> Dict:
        """
        Analyze segment from file path.
        
        Args:
            segment_path: Path to audio segment file
            cache_key: Optional cache key for future caching
            
        Returns:
            Analysis dict compatible with SmartMixer._analyze_song_fast format
        """
        y, sr = librosa.load(segment_path, sr=self.sr)
        return self.analyze_segment(y, segment_path, cache_key)
