"""
Complete song analysis module - extracts all features a DJ perceives
"""

import librosa
import numpy as np
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class SongAnalyzer:
    """
    Analyzes a song and extracts comprehensive features for DJ/transition AI training.
    No audio is stored - only extracted knowledge.
    """
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        self.sr = sample_rate
        self.hop_length = hop_length
        
    def analyze(self, audio_path: str) -> Dict:
        """
        Complete analysis of a song.
        Returns comprehensive feature dictionary.
        """
        print(f"Analyzing: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr)
        duration = len(y) / sr
        
        # Compute song ID from audio hash
        song_id = self._compute_song_id(y)
        
        # Extract all features
        analysis = {
            "song_id": song_id,
            "metadata": self._extract_metadata(audio_path, duration, sr),
            "tempo": self._analyze_tempo(y, sr),
            "harmony": self._analyze_harmony(y, sr),
            "energy": self._analyze_energy(y, sr),
            "spectrum": self._analyze_spectrum(y, sr),
            "timbre": self._analyze_timbre(y, sr),
            "onsets": self._analyze_onsets(y, sr),
            "structure": self._analyze_structure(y, sr),
            "vocals": self._analyze_vocals(y, sr),
            "stereo": self._analyze_stereo(y, sr),
            "embeddings": self._compute_embeddings(y, sr)
        }
        
        print(f"✓ Analysis complete: {duration:.1f}s")
        return analysis
    
    def _compute_song_id(self, y: np.ndarray) -> str:
        """Compute unique ID from audio content"""
        # Use first 10 seconds for hashing
        sample = y[:min(len(y), self.sr * 10)]
        hash_obj = hashlib.sha256(sample.tobytes())
        return hash_obj.hexdigest()[:16]
    
    def _extract_metadata(self, audio_path: str, duration: float, sr: int) -> Dict:
        """Extract basic metadata"""
        return {
            "duration_sec": duration,
            "sample_rate": sr,
            "file_path": str(audio_path),
            "file_name": Path(audio_path).name
        }
    
    def _analyze_tempo(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze tempo, beats, and rhythm"""
        print("  Analyzing tempo and rhythm...")
        
        # Overall tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        
        # Beat positions
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
        
        # Downbeat detection (simplified - assume 4/4)
        downbeats = beats[::4]  # Every 4th beat
        downbeat_times = librosa.frames_to_time(downbeats, sr=sr, hop_length=self.hop_length)
        
        # Beat strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        beat_strengths = onset_env[beats]
        
        # Tempo stability (analyze tempo over time)
        window_sec = 10
        hop_sec = 5
        tempo_curve = []
        times_sec = []
        
        for start in range(0, int(len(y) / sr) - window_sec, hop_sec):
            segment = y[int(start * sr):int((start + window_sec) * sr)]
            if len(segment) > 0:
                seg_tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
                tempo_curve.append(float(seg_tempo))
                times_sec.append(start)
        
        # Rhythm features
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=self.hop_length)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
        
        # Microtiming analysis
        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            expected_interval = 60.0 / tempo
            deviations = (beat_intervals - expected_interval) * 1000  # ms
            microtiming_mean = float(np.mean(np.abs(deviations)))
            microtiming_std = float(np.std(deviations))
        else:
            microtiming_mean = 0.0
            microtiming_std = 0.0
        
        return {
            "tempo": {
                "bpm": float(tempo),
                "bpm_confidence": 0.95,  # librosa doesn't provide this, estimate
                "tempo_variation_std": float(np.std(tempo_curve)) if tempo_curve else 0.0,
                "tempo_curve": {
                    "times_sec": times_sec,
                    "bpm_values": tempo_curve
                }
            },
            "time_signature": {
                "numerator": 4,
                "denominator": 4,
                "confidence": 0.95
            },
            "beat_grid": {
                "first_beat_sec": float(beat_times[0]) if len(beat_times) > 0 else 0.0,
                "beat_interval_sec": float(60.0 / tempo),
                "total_beats": len(beat_times),
                "beat_positions_sec": beat_times.tolist(),
                "beat_strengths": beat_strengths.tolist(),
                "downbeat_positions_sec": downbeat_times.tolist()
            },
            "rhythm_features": {
                "onset_rate_per_sec": len(onset_times) / (len(y) / sr),
                "syncopation_score": 0.3,  # Placeholder - complex calculation
                "rhythmic_complexity": min(1.0, len(onset_times) / len(beat_times)) if len(beat_times) > 0 else 0.0
            },
            "microtiming": {
                "beat_deviation_ms_mean": microtiming_mean,
                "beat_deviation_ms_std": microtiming_std,
                "human_feel_score": 1.0 - min(1.0, microtiming_std / 50.0)  # Normalize
            }
        }
    
    def _analyze_harmony(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze key, chords, and harmony"""
        print("  Analyzing harmony...")
        
        # Chromagram for key detection
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        
        # Estimate key (simplified - use chroma average)
        chroma_mean = np.mean(chroma, axis=1)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Find dominant pitch class
        dominant_pc = int(np.argmax(chroma_mean))
        estimated_key = key_names[dominant_pc]
        
        # Simple major/minor detection (compare major vs minor triads)
        major_triad = [0, 4, 7]
        minor_triad = [0, 3, 7]
        major_score = sum(chroma_mean[(dominant_pc + offset) % 12] for offset in major_triad)
        minor_score = sum(chroma_mean[(dominant_pc + offset) % 12] for offset in minor_triad)
        mode = "major" if major_score > minor_score else "minor"
        
        # Camelot wheel mapping
        camelot_map = {
            'C': '8B', 'C#': '3B', 'D': '10B', 'D#': '5B', 'E': '12B', 'F': '7B',
            'F#': '2B', 'G': '9B', 'G#': '4B', 'A': '11B', 'A#': '6B', 'B': '1B'
        }
        camelot_minor_map = {
            'C': '5A', 'C#': '12A', 'D': '7A', 'D#': '2A', 'E': '9A', 'F': '4A',
            'F#': '11A', 'G': '6A', 'G#': '1A', 'A': '8A', 'A#': '3A', 'B': '10A'
        }
        camelot = camelot_map[estimated_key] if mode == "major" else camelot_minor_map[estimated_key]
        
        # Key over time
        window_frames = int(10 * sr / self.hop_length)  # 10 second windows
        hop_frames = int(5 * sr / self.hop_length)  # 5 second hop
        key_over_time = []
        
        for start in range(0, chroma.shape[1] - window_frames, hop_frames):
            window_chroma = chroma[:, start:start + window_frames]
            window_mean = np.mean(window_chroma, axis=1)
            window_pc = int(np.argmax(window_mean))
            window_key = key_names[window_pc]
            key_over_time.append({
                "start": start * self.hop_length / sr,
                "end": (start + window_frames) * self.hop_length / sr,
                "key": window_key,
                "confidence": 0.8
            })
        
        # Pitch class distribution
        pc_dist = (chroma_mean / chroma_mean.sum()).tolist()
        pc_distribution = {key_names[i]: float(pc_dist[i]) for i in range(12)}
        
        return {
            "key": {
                "estimated_key": estimated_key,
                "mode": mode,
                "camelot": camelot,
                "confidence": 0.85
            },
            "key_over_time": key_over_time,
            "pitch_class_distribution": pc_distribution,
            "chord_sequence": []  # Placeholder - requires more complex analysis
        }
    
    def _analyze_energy(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze energy, dynamics, and loudness"""
        print("  Analyzing energy and dynamics...")
        
        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        rms_normalized = (rms - rms.min()) / (rms.max() - rms.min() + 1e-10)
        
        # Time axis
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=self.hop_length)
        
        # Peak detection
        peak = np.abs(y)
        peak_db = librosa.amplitude_to_db(peak, ref=np.max)
        
        # Energy statistics
        energy_stats = {
            "mean": float(np.mean(rms_normalized)),
            "std": float(np.std(rms_normalized)),
            "min": float(np.min(rms_normalized)),
            "max": float(np.max(rms_normalized)),
            "percentile_10": float(np.percentile(rms_normalized, 10)),
            "percentile_90": float(np.percentile(rms_normalized, 90))
        }
        
        # Find peaks and valleys
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(rms_normalized, distance=int(sr / self.hop_length))
        valleys, _ = find_peaks(-rms_normalized, distance=int(sr / self.hop_length))
        
        peak_times = times[peaks].tolist() if len(peaks) > 0 else []
        valley_times = times[valleys].tolist() if len(valleys) > 0 else []
        
        # Loudness (simplified - use RMS as proxy)
        integrated_lufs = float(np.mean(rms_db)) - 23.0  # Rough conversion
        
        return {
            "loudness": {
                "integrated_lufs": integrated_lufs,
                "loudness_range_lu": float(np.max(rms_db) - np.min(rms_db)),
                "dynamic_range_db": float(np.max(rms_db) - np.min(rms_db))
            },
            "energy_curve": {
                "window_sec": self.hop_length / sr,
                "times_sec": times.tolist(),
                "rms_db": rms_db.tolist(),
                "rms_normalized": rms_normalized.tolist(),
                "peak_db": peak_db[::self.hop_length].tolist()[:len(times)]
            },
            "energy_statistics": energy_stats,
            "energy_contour": {
                "peak_positions_sec": peak_times[:10],  # Top 10 peaks
                "valley_positions_sec": valley_times[:10]
            }
        }
    
    def _analyze_spectrum(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze spectral content - critical for mixing"""
        print("  Analyzing spectral content...")
        
        # STFT
        S = np.abs(librosa.stft(y, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)[0]
        
        # Frequency bands
        band_masks = {
            "sub_bass_20_60": (freqs >= 20) & (freqs < 60),
            "bass_60_250": (freqs >= 60) & (freqs < 250),
            "low_mid_250_500": (freqs >= 250) & (freqs < 500),
            "mid_500_2000": (freqs >= 500) & (freqs < 2000),
            "high_mid_2000_6000": (freqs >= 2000) & (freqs < 6000),
            "high_6000_20000": (freqs >= 6000) & (freqs <= sr/2)
        }
        
        # Compute band energies over time
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=self.hop_length)
        band_energies = {}
        
        for band_name, mask in band_masks.items():
            band_S = S[mask, :]
            band_energy = np.mean(band_S, axis=0)
            band_energy_normalized = (band_energy - band_energy.min()) / (band_energy.max() - band_energy.min() + 1e-10)
            
            band_energies[band_name] = {
                "energy_mean": float(np.mean(band_energy_normalized)),
                "energy_std": float(np.std(band_energy_normalized)),
                "presence_ratio": float(np.mean(band_energy_normalized > 0.1)),
                "curve": band_energy_normalized[::10].tolist()  # Downsample for storage
            }
        
        # Spectral balance
        bass_energy = np.mean(S[band_masks["bass_60_250"], :])
        mid_energy = np.mean(S[band_masks["mid_500_2000"], :])
        high_energy = np.mean(S[band_masks["high_6000_20000"], :])
        
        return {
            "spectral_shape": {
                "centroid_hz_mean": float(np.mean(spectral_centroid)),
                "centroid_hz_std": float(np.std(spectral_centroid)),
                "bandwidth_hz_mean": float(np.mean(spectral_bandwidth)),
                "rolloff_hz_mean": float(np.mean(spectral_rolloff)),
                "flatness_mean": float(np.mean(spectral_flatness))
            },
            "spectral_curve": {
                "times_sec": times[::10].tolist(),  # Downsample
                "centroid_hz": spectral_centroid[::10].tolist(),
                "bandwidth_hz": spectral_bandwidth[::10].tolist(),
                "rolloff_hz": spectral_rolloff[::10].tolist()
            },
            "frequency_bands": band_energies,
            "spectral_balance": {
                "bass_to_mid_ratio": float(bass_energy / (mid_energy + 1e-10)),
                "mid_to_high_ratio": float(mid_energy / (high_energy + 1e-10)),
                "overall_brightness": float(np.mean(spectral_centroid) / 5000.0)  # Normalize
            }
        }
    
    def _analyze_timbre(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze timbre and texture"""
        print("  Analyzing timbre...")
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=self.hop_length)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Timbre descriptors (simplified)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        
        brightness = np.mean(spectral_centroid) / 5000.0
        roughness = float(np.std(zero_crossing_rate))
        
        return {
            "mfcc": {
                "n_mfcc": 20,
                "coefficients_mean": np.mean(mfcc, axis=1).tolist(),
                "coefficients_std": np.std(mfcc, axis=1).tolist(),
                "delta_mean": np.mean(mfcc_delta, axis=1).tolist(),
                "delta_delta_mean": np.mean(mfcc_delta2, axis=1).tolist()
            },
            "timbre_descriptors": {
                "brightness": float(brightness),
                "roughness": float(roughness),
                "sharpness": float(np.mean(spectral_rolloff) / 10000.0),
                "fullness": 0.7  # Placeholder
            }
        }
    
    def _analyze_onsets(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze onsets and transients"""
        print("  Analyzing onsets...")
        
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=self.hop_length)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        onset_strengths = onset_strength[onset_frames]
        
        return {
            "onsets": {
                "onset_times_sec": onset_times.tolist(),
                "onset_strengths": onset_strengths.tolist(),
                "onset_rate_per_sec": len(onset_times) / (len(y) / sr),
                "onset_regularity": 0.85  # Placeholder
            },
            "percussive_content": {
                "drum_presence": min(1.0, len(onset_times) / (len(y) / sr) / 4.0),  # Normalize to ~4 beats/sec
                "percussion_pattern_type": "four_on_floor"  # Placeholder
            }
        }
    
    def _analyze_structure(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze song structure - critical for finding mix points"""
        print("  Analyzing structure...")
        
        # Use self-similarity matrix for structure
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        
        # Simplified structure detection
        # In production, use msaf or similar
        duration = len(y) / sr
        
        # Assume typical EDM structure (intro, buildup, drop, breakdown, outro)
        sections = []
        if duration > 60:
            sections.append({"type": "intro", "start_sec": 0.0, "end_sec": min(16.0, duration * 0.1)})
            sections.append({"type": "verse", "start_sec": min(16.0, duration * 0.1), "end_sec": duration * 0.4})
            sections.append({"type": "buildup", "start_sec": duration * 0.4, "end_sec": duration * 0.5})
            sections.append({"type": "drop", "start_sec": duration * 0.5, "end_sec": duration * 0.7})
            sections.append({"type": "breakdown", "start_sec": duration * 0.7, "end_sec": duration * 0.85})
            sections.append({"type": "outro", "start_sec": duration * 0.85, "end_sec": duration})
        
        # Find intro/outro for mixing
        intro_quality = 0.8  # Placeholder
        outro_quality = 0.75  # Placeholder
        
        return {
            "structure": {
                "sections": sections
            },
            "mixability": {
                "intro_quality": intro_quality,
                "intro_length_bars": 8,
                "intro_has_beat": True,
                "outro_quality": outro_quality,
                "outro_length_bars": 8,
                "outro_has_beat": True,
                "best_mix_in_points_sec": [0.0],
                "best_mix_out_points_sec": [duration * 0.9]
            }
        }
    
    def _analyze_vocals(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze vocal content"""
        print("  Analyzing vocals...")
        
        # Simplified vocal detection using spectral features
        # In production, use spleeter or demucs
        
        # Vocal-like frequencies (rough heuristic)
        S = np.abs(librosa.stft(y, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=sr)
        vocal_mask = (freqs >= 80) & (freqs <= 4000)
        vocal_energy = np.mean(S[vocal_mask, :], axis=0)
        
        # Threshold for vocal presence
        vocal_threshold = np.percentile(vocal_energy, 70)
        has_vocals = np.mean(vocal_energy > vocal_threshold) > 0.2
        
        times = librosa.frames_to_time(np.arange(len(vocal_energy)), sr=sr, hop_length=self.hop_length)
        vocal_presence = (vocal_energy > vocal_threshold).astype(float)
        
        return {
            "vocals": {
                "has_vocals": bool(has_vocals),
                "vocal_type": "lead" if has_vocals else None
            },
            "vocal_timeline": {
                "times_sec": times[::10].tolist(),  # Downsample
                "vocal_presence": vocal_presence[::10].tolist()
            }
        }
    
    def _analyze_stereo(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze stereo and spatial features"""
        print("  Analyzing stereo...")
        
        if y.ndim == 2 and y.shape[0] == 2:
            left = y[0]
            right = y[1]
            
            # Stereo width
            mid = (left + right) / 2
            side = (left - right) / 2
            width = np.abs(side) / (np.abs(mid) + 1e-10)
            width_mean = float(np.mean(width))
            
            # Correlation
            correlation = np.corrcoef(left, right)[0, 1] if len(left) > 1 else 0.0
        else:
            width_mean = 0.0
            correlation = 1.0
        
        return {
            "stereo": {
                "stereo_width_mean": width_mean,
                "correlation_mean": float(correlation),
                "mono_compatibility": 0.9
            }
        }
    
    def _compute_embeddings(self, y: np.ndarray, sr: int) -> Dict:
        """Compute learned embeddings (placeholder - requires external models)"""
        print("  Computing embeddings...")
        
        # Placeholder - in production, load CLAP, musicnn, etc.
        # For now, use simple feature aggregation
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        embedding = np.mean(mfcc, axis=1).tolist()
        
        return {
            "embeddings": {
                "feature_embedding": embedding,  # Placeholder
                "embedding_dim": 13
            },
            "semantic_tags": {
                "genre": [],  # Placeholder
                "mood": []
            }
        }
    
    def extract_song_state_at_time(self, analysis: Dict, time_sec: float, lookback_bars: int = 32, lookahead_bars: int = 0) -> Dict:
        """
        Extract song state at a specific time point.
        Used for extracting context during transitions.
        
        Args:
            analysis: Complete song analysis dictionary
            time_sec: Time point to extract state at
            lookback_bars: Number of bars to look back for trend analysis
            lookahead_bars: Number of bars to look ahead
        
        Returns:
            Dictionary with song state at the specified time
        """
        bpm = analysis.get('tempo', {}).get('bpm', 128)
        sec_per_bar = 60.0 / bpm * 4  # Assuming 4/4 time
        
        # Determine which section we're in
        sections = analysis.get('structure', {}).get('structure', {}).get('sections', [])
        current_section = None
        bars_into_section = 0
        
        for section in sections:
            if section['start_sec'] <= time_sec <= section['end_sec']:
                current_section = section['type']
                bars_into_section = int((time_sec - section['start_sec']) / sec_per_bar)
                break
        
        # Extract energy values around this time
        energy_curve = analysis.get('energy', {}).get('energy_curve', {})
        energy_times = energy_curve.get('times_sec', [])
        energy_values = energy_curve.get('rms_normalized', [])
        
        # Get energy at this time (interpolate if needed)
        current_energy = self._get_value_at_time(energy_times, energy_values, time_sec, default=0.5)
        
        # Compute energy trend (lookback period)
        lookback_start = max(0, time_sec - (lookback_bars * sec_per_bar))
        lookback_energies = [v for t, v in zip(energy_times, energy_values) 
                           if lookback_start <= t <= time_sec]
        
        if len(lookback_energies) > 1:
            energy_trend_slope = (lookback_energies[-1] - lookback_energies[0]) / len(lookback_energies)
            if energy_trend_slope > 0.01:
                energy_trend = "increasing"
            elif energy_trend_slope < -0.01:
                energy_trend = "decreasing"
            else:
                energy_trend = "stable"
        else:
            energy_trend = "stable"
        
        # Extract last N bars of energy (for ML input)
        last_32_bars_start = max(0, time_sec - (32 * sec_per_bar))
        last_32_bars_energy = []
        for t, v in zip(energy_times, energy_values):
            if t >= last_32_bars_start and t <= time_sec:
                last_32_bars_energy.append(v)
        
        # Pad or trim to 32 bars (8 samples per bar = 256 samples)
        target_samples = 256
        if len(last_32_bars_energy) > target_samples:
            indices = np.linspace(0, len(last_32_bars_energy)-1, target_samples, dtype=int)
            last_32_bars_energy = [last_32_bars_energy[i] for i in indices]
        else:
            last_32_bars_energy = last_32_bars_energy + [last_32_bars_energy[-1]] * (target_samples - len(last_32_bars_energy))
        
        # Extract spectral features at this time
        spectrum = analysis.get('spectrum', {})
        spectral_centroid_curve = spectrum.get('spectral_curve', {}).get('centroid_hz', [])
        spectral_times = spectrum.get('spectral_curve', {}).get('times_sec', [])
        
        current_centroid = self._get_value_at_time(spectral_times, spectral_centroid_curve, time_sec, 
                                                   default=spectrum.get('spectral_shape', {}).get('centroid_hz_mean', 2400))
        
        # Extract frequency band energies
        bands = spectrum.get('frequency_bands', {})
        bass_energy = bands.get('bass_60_250', {}).get('energy_mean', 0.5)
        mid_energy = bands.get('mid_500_2000', {}).get('energy_mean', 0.5)
        
        # Vocal presence
        vocals = analysis.get('vocals', {})
        vocal_timeline = vocals.get('vocal_timeline', {})
        vocal_times = vocal_timeline.get('times_sec', [])
        vocal_presence = vocal_timeline.get('vocal_presence', [])
        
        has_vocals_now = self._get_value_at_time(vocal_times, vocal_presence, time_sec, default=0.0) > 0.5
        
        return {
            "bpm": bpm,
            "key": analysis.get('harmony', {}).get('key', {}).get('estimated_key', 'Unknown'),
            "camelot": analysis.get('harmony', {}).get('key', {}).get('camelot', ''),
            "energy": float(current_energy),
            "section_type": current_section or "unknown",
            "bars_into_section": bars_into_section,
            "has_vocals": bool(has_vocals_now) if has_vocals_now else vocals.get('vocals', {}).get('has_vocals', False),
            "bass_energy": float(bass_energy),
            "mid_energy": float(mid_energy),
            "spectral_centroid_hz": float(current_centroid),
            "energy_trend": energy_trend,
            "last_32_bars_energy": [float(v) for v in last_32_bars_energy[:256]],  # Ensure 256 samples
            "time_sec": float(time_sec)
        }
    
    def _get_value_at_time(self, times: List[float], values: List[float], target_time: float, default: float = 0.0) -> float:
        """Get interpolated value at a specific time"""
        if not times or not values:
            return default
        
        if target_time <= times[0]:
            return values[0]
        if target_time >= times[-1]:
            return values[-1]
        
        # Find surrounding points
        for i in range(len(times) - 1):
            if times[i] <= target_time <= times[i + 1]:
                # Linear interpolation
                t0, t1 = times[i], times[i + 1]
                v0, v1 = values[i], values[i + 1]
                ratio = (target_time - t0) / (t1 - t0) if (t1 - t0) > 0 else 0
                return v0 + ratio * (v1 - v0)
        
        return default


def save_analysis(analysis: Dict, output_path: str):
    """Save analysis to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Saved analysis to: {output_path}")


if __name__ == "__main__":
    # Example usage
    analyzer = SongAnalyzer()
    # analysis = analyzer.analyze("path/to/song.wav")
    # save_analysis(analysis, "song_analysis.json")

