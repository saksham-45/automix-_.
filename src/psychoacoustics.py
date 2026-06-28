"""
Psychoacoustic Analysis Module

Implements human sound perception models:
- Frequency masking (Bark scale, simultaneous/temporal masking)
- Loudness perception (LUFS, ITU-R BS.1770)
- Auditory scene analysis (stream segregation, figure-ground)
"""
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from scipy import signal


class PsychoacousticAnalyzer:
    """
    Analyzes audio using psychoacoustic principles - how humans actually hear.
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.n_fft = 2048
        self.hop_length = 512
        
        # Bark scale frequencies (approximate)
        # 25 critical bands covering 0-20kHz
        self.bark_bands = self._calculate_bark_bands()
    
    def _calculate_bark_bands(self) -> np.ndarray:
        """Calculate Bark scale frequency bands."""
        # Bark scale: z = 13*arctan(0.00076*f) + 3.5*arctan((f/7500)^2)
        freqs = np.linspace(0, self.sr / 2, self.n_fft // 2 + 1)
        
        # Convert Hz to Bark
        bark = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)
        
        # Group into critical bands (approximately 1 Bark per band)
        n_bands = 25
        bark_bands = np.linspace(0, bark.max(), n_bands + 1)
        
        return bark_bands
    
    def analyze_frequency_masking(self, 
                                  y: np.ndarray,
                                  masker: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze frequency masking in audio.
        
        Args:
            y: Audio signal (potentially masked signal)
            masker: Optional masking signal (if analyzing two signals)
        
        Returns:
            Dict with masking analysis
        """
        # Limit to 10 seconds max for speed
        max_samples = 10 * self.sr
        if len(y) > max_samples:
            y = y[:max_samples]
        
        # Use smaller FFT for faster computation
        n_fft_fast = 1024  # Smaller FFT
        hop_length_fast = 256  # Larger hop = fewer frames
        
        # Compute STFT
        S = np.abs(librosa.stft(y, n_fft=n_fft_fast, hop_length=hop_length_fast))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft_fast)
        
        # Convert to Bark scale
        bark = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)
        
        # Calculate masking threshold for each critical band
        masking_threshold = self._calculate_masking_threshold(S, bark)
        
        # If masker provided, calculate how it masks the signal
        masker_mask = None
        if masker is not None:
            # Limit masker sample too
            if len(masker) > max_samples:
                masker = masker[:max_samples]
            S_masker = np.abs(librosa.stft(masker, n_fft=n_fft_fast, hop_length=hop_length_fast))
            masker_mask = self._calculate_masker_effect(S, S_masker, bark)
        
        return {
            'masking_threshold_db': masking_threshold.tolist(),
            'critical_bands': self.bark_bands.tolist(),
            'masker_effect': masker_mask.tolist() if masker_mask is not None else None,
            'masked_frequencies': self._identify_masked_frequencies(S, masking_threshold, freqs)
        }
    
    def _calculate_masking_threshold(self, 
                                     magnitude: np.ndarray, 
                                     bark: np.ndarray) -> np.ndarray:
        """
        Calculate simultaneous masking threshold using spreading function.
        OPTIMIZED: Simplified calculation for speed.
        """
        # Simplified masking model - use mean over time for speed
        # Average magnitude over time frames
        magnitude_mean = np.mean(magnitude, axis=1)  # Average across time
        
        n_bins = len(magnitude_mean)
        threshold = np.zeros(n_bins)
        
        # Convert magnitude to dB
        magnitude_db = librosa.amplitude_to_db(magnitude_mean + 1e-10, ref=np.max(magnitude_mean))
        
        # Simplified: only process every 4th bin for speed
        for i in range(0, n_bins, 4):
            if magnitude_db[i] < -60:  # Too quiet to mask
                continue
            
            # Simple spreading: mask nearby 5 bins
            spread = 5
            for j in range(max(0, i - spread), min(n_bins, i + spread + 1)):
                distance = abs(j - i)
                attenuation = -20 - 5 * distance  # Simplified attenuation
                threshold[j] = max(threshold[j], magnitude_db[i] + attenuation)
        
        # Expand threshold back to full size
        threshold_full = np.tile(threshold[:, np.newaxis], (1, magnitude.shape[1]))
        return threshold_full
    
    def _calculate_masker_effect(self,
                                 masked_spec: np.ndarray,
                                 masker_spec: np.ndarray,
                                 bark: np.ndarray) -> np.ndarray:
        """Calculate how masker affects the masked signal."""
        masker_threshold = self._calculate_masking_threshold(masker_spec, bark)
        masked_db = librosa.amplitude_to_db(masked_spec, ref=np.max)
        
        # Frequencies below masking threshold are masked
        mask_effect = np.maximum(0, masker_threshold - masked_db)
        
        return mask_effect
    
    def _identify_masked_frequencies(self,
                                    magnitude: np.ndarray,
                                    threshold: np.ndarray,
                                    freqs: np.ndarray) -> List[float]:
        """Identify which frequencies are likely masked."""
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Frequencies where signal is below threshold
        # Handle 1D or 2D threshold
        if threshold.ndim == 1:
            threshold_2d = threshold[:, np.newaxis]
        else:
            threshold_2d = threshold
            
        masked = magnitude_db < threshold_2d - 10  # 10dB below threshold = masked
        masked_ratio = np.mean(masked, axis=1)
        
        # Return frequencies with >50% masked
        masked_freqs = freqs[masked_ratio > 0.5].tolist()
        
        return masked_freqs
    
    def analyze_loudness_lufs(self, y: np.ndarray) -> Dict:
        """
        Analyze loudness using ITU-R BS.1770 (LUFS).
        
        LUFS is more accurate than RMS because it uses perceptual weighting.
        """
        # Use sample if audio is too long (for speed)
        max_samples = 60 * self.sr  # 60 seconds max
        if len(y) > max_samples:
            y = y[:max_samples]
        
        # ITU-R BS.1770 K-weighting: two cascaded biquads (high-shelf + RLB high-pass).
        # Previously the K-weighting filter was *computed but never applied* (the
        # reported value was an unweighted, HP-only mean-square -> not LUFS).
        fs = float(self.sr)

        def _biquad(x, b, a):
            return signal.lfilter(b, a, x, axis=0)

        # Stage 1: high-frequency shelving boost (~+4 dB above ~1.7 kHz)
        f0 = 1681.974450955533
        G = 3.999843853973347
        Qs = 0.7071752369554196
        K = np.tan(np.pi * f0 / fs)
        Vh = 10.0 ** (G / 20.0)
        Vb = Vh ** 0.4996667741545416
        a0s = 1.0 + K / Qs + K * K
        b_shelf = [(Vh + Vb * K / Qs + K * K) / a0s,
                   2.0 * (K * K - Vh) / a0s,
                   (Vh - Vb * K / Qs + K * K) / a0s]
        a_shelf = [1.0,
                   2.0 * (K * K - 1.0) / a0s,
                   (1.0 - K / Qs + K * K) / a0s]

        # Stage 2: RLB high-pass (~38 Hz)
        f0h = 38.13547087602444
        Qh = 0.5003270373238773
        Kh = np.tan(np.pi * f0h / fs)
        a0h = 1.0 + Kh / Qh + Kh * Kh
        b_hp = [1.0, -2.0, 1.0]
        a_hp = [1.0,
                2.0 * (Kh * Kh - 1.0) / a0h,
                (1.0 - Kh / Qh + Kh * Kh) / a0h]

        if len(y) > 1024:
            y_kw = _biquad(_biquad(y.astype(np.float64), b_shelf, a_shelf), b_hp, a_hp)
        else:
            y_kw = y.astype(np.float64)

        # BS.1770 sums the mean-square power of each channel (L,R weighted 1.0).
        # Build a per-sample summed-power series so the LRA windowing below is consistent.
        if y_kw.ndim > 1:
            ms_series = np.sum(y_kw ** 2, axis=1)
        else:
            ms_series = y_kw ** 2

        # Mean square calculation
        mean_square = float(np.mean(ms_series))
        
        # Convert to LUFS
        # LUFS = -0.691 + 10*log10(mean_square)
        # For reference: 0 LUFS = -23 dBFS (EBU R128)
        lufs = -0.691 + 10 * np.log10(mean_square + 1e-10)
        
        # Loudness range (LRA) - variation in loudness
        # Use 3-second windows, compute 10th and 95th percentiles
        window_samples = 3 * self.sr
        n_windows = len(ms_series) // window_samples

        window_lufs = []
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            window_ms = np.mean(ms_series[start:end])
            window_lufs_val = -0.691 + 10 * np.log10(window_ms + 1e-10)
            window_lufs.append(window_lufs_val)
        
        if len(window_lufs) > 0:
            window_lufs = np.array(window_lufs)
            lra = np.percentile(window_lufs, 95) - np.percentile(window_lufs, 10)
        else:
            lra = 0.0
        
        # Peak level
        peak = np.max(np.abs(y))
        peak_db = 20 * np.log10(peak + 1e-10)
        
        return {
            'integrated_lufs': float(lufs),
            'loudness_range_lu': float(lra),
            'peak_db': float(peak_db),
            'true_peak_db': float(peak_db),  # Simplified - would need oversampling for true peak
            'peak_to_loudness_ratio': float(peak_db - lufs)
        }
    
    def analyze_auditory_scene(self, y: np.ndarray) -> Dict:
        """
        Analyze auditory scene - stream segregation and figure-ground separation.
        """
        # Separate harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Harmonic components (melody, vocals, chords) - foreground
        harmonic_energy = np.mean(np.abs(y_harmonic) ** 2)
        
        # Percussive components (drums, transients) - can be foreground or background
        percussive_energy = np.mean(np.abs(y_percussive) ** 2)
        
        # Spectral centroid (brightness) - related to figure-ground
        # Higher centroid = more prominent/foreground
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        avg_centroid = np.mean(spectral_centroid)
        
        # Onset detection - transient events (foreground)
        onset_frames = librosa.onset.onset_detect(y=y, sr=self.sr, hop_length=self.hop_length)
        onset_rate = len(onset_frames) / (len(y) / self.sr)
        
        # Spectral contrast - distinguishes foreground from background
        # High contrast = clear foreground/background separation
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)[0]
        avg_contrast = np.mean(spectral_contrast)
        
        return {
            'harmonic_energy': float(harmonic_energy),
            'percussive_energy': float(percussive_energy),
            'harmonic_percussive_ratio': float(harmonic_energy / (percussive_energy + 1e-10)),
            'spectral_centroid_hz': float(avg_centroid),
            'onset_rate_per_sec': float(onset_rate),
            'spectral_contrast': float(avg_contrast),
            'foreground_prominence': float(avg_contrast / 1000)  # Normalized estimate
        }
    
    def predict_frequency_clash(self,
                               y_a: np.ndarray,
                               y_b: np.ndarray) -> Dict:
        """
        Predict frequency clashes between two audio signals during overlap.
        
        Returns:
            Dict with clash prediction and recommendations
        """
        import time, json
        
        
        # Use samples if audio is too long (for speed) - already sampled in caller
        # No need to sample again here
        
        
        # Analyze masking for both signals
        mask_a = self.analyze_frequency_masking(y_a)
        
        
        mask_b = self.analyze_frequency_masking(y_b)
        
        
        # Calculate mutual masking
        S_a = np.abs(librosa.stft(y_a, n_fft=self.n_fft, hop_length=self.hop_length))
        S_b = np.abs(librosa.stft(y_b, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # Energy in critical bands
        bark = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)
        
        # Group frequencies into critical bands
        n_bands = 25
        band_energies_a = np.zeros(n_bands)
        band_energies_b = np.zeros(n_bands)
        
        for i in range(len(bark)):
            band_idx = int((bark[i] / bark.max()) * (n_bands - 1))
            band_idx = min(band_idx, n_bands - 1)
            band_energies_a[band_idx] += np.mean(S_a[i] ** 2)
            band_energies_b[band_idx] += np.mean(S_b[i] ** 2)
        
        # Normalize
        band_energies_a = band_energies_a / (band_energies_a.max() + 1e-10)
        band_energies_b = band_energies_b / (band_energies_b.max() + 1e-10)
        
        # Clash occurs when both signals have high energy in same band
        clash_score = np.sum(band_energies_a * band_energies_b) / n_bands
        
        # Identify problematic frequency ranges
        clash_bands = []
        for i in range(n_bands):
            if band_energies_a[i] > 0.5 and band_energies_b[i] > 0.5:
                clash_bands.append(i)
        
        # Recommendations
        recommendations = []
        if clash_score > 0.3:
            recommendations.append('High frequency clash risk - use EQ cuts')
        if len(clash_bands) > 5:
            recommendations.append('Multiple frequency bands clashing - consider bass swap')
        
        # Estimate bass clash
        bass_bands = list(range(0, 5))  # Low frequencies
        bass_clash = np.sum([band_energies_a[i] * band_energies_b[i] for i in bass_bands])
        if bass_clash > 0.4:
            recommendations.append('Bass frequencies clashing - perform bass swap')
        
        
        return {
            'clash_score': float(clash_score),  # 0-1, higher = more clash
            'clashing_bands': clash_bands,
            'bass_clash': float(bass_clash),
            'recommendations': recommendations,
            'band_energies_a': band_energies_a.tolist(),
            'band_energies_b': band_energies_b.tolist()
        }

