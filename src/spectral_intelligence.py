"""
Spectral Intelligence Engine

Advanced spectral processing that exceeds human DJ capabilities:
- Frequency slot negotiation (surgical EQ carving)
- Harmonic resonance boosting (enhance compatible frequencies)
- Spectral morphing (smooth spectral envelope transitions)
- Masking-aware mixing (psychoacoustic clash prevention)

This module prevents frequency fights and creates seamless blends.
"""
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.ndimage import gaussian_filter1d


class SpectralIntelligenceEngine:
    """
    Intelligent spectral processing for superhuman mixing clarity.
    
    Human DJs use EQ knobs; this engine understands the full spectrum
    and makes surgical cuts/boosts to create perfect frequency separation.
    """
    
    def __init__(self, sr: int = 44100, n_fft: int = 4096):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        
        # Define critical frequency bands
        self.frequency_bands = {
            'sub_bass': (20, 60),
            'bass': (60, 200),
            'low_mid': (200, 500),
            'mid': (500, 2000),
            'upper_mid': (2000, 5000),
            'presence': (5000, 8000),
            'brilliance': (8000, 16000),
            'air': (16000, 22000)
        }
        
        # Critical frequencies for different instruments
        self.instrument_frequencies = {
            'kick': (40, 100),
            'bass': (60, 250),
            'snare_body': (150, 400),
            'snare_crack': (1500, 4000),
            'vocals_fundamental': (100, 400),
            'vocals_clarity': (2000, 5000),
            'hi_hats': (8000, 16000)
        }
    
    # ==================== SPECTRAL ANALYSIS ====================
    
    def analyze_spectrum(self, y: np.ndarray) -> Dict:
        """
        Comprehensive spectral analysis of audio.
        
        Returns detailed frequency content analysis including:
        - Band energies
        - Spectral peaks
        - Harmonic content
        - Dominant frequencies
        """
        # Get spectrogram
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # Average spectrum
        avg_spectrum = np.mean(S, axis=1)
        
        # Energy per band
        band_energies = {}
        for band_name, (low, high) in self.frequency_bands.items():
            mask = (freqs >= low) & (freqs < high)
            band_energies[band_name] = float(np.sum(avg_spectrum[mask] ** 2))
        
        # Normalize energies
        total_energy = sum(band_energies.values()) + 1e-10
        band_ratios = {k: v / total_energy for k, v in band_energies.items()}
        
        # Find spectral peaks (dominant frequencies)
        peaks, properties = signal.find_peaks(
            avg_spectrum, 
            height=np.max(avg_spectrum) * 0.1,
            distance=int(100 / (self.sr / self.n_fft))  # Min 100Hz apart
        )
        
        peak_freqs = freqs[peaks][:20]  # Top 20 peaks
        peak_amps = avg_spectrum[peaks][:20]
        
        # Spectral centroid (brightness)
        centroid = float(np.sum(freqs * avg_spectrum) / (np.sum(avg_spectrum) + 1e-10))
        
        # Spectral bandwidth
        bandwidth = float(np.sqrt(
            np.sum(((freqs - centroid) ** 2) * avg_spectrum) / (np.sum(avg_spectrum) + 1e-10)
        ))
        
        # Spectral flatness (tonality vs noise)
        geometric_mean = np.exp(np.mean(np.log(avg_spectrum + 1e-10)))
        arithmetic_mean = np.mean(avg_spectrum)
        flatness = float(geometric_mean / (arithmetic_mean + 1e-10))
        
        return {
            'band_energies': band_energies,
            'band_ratios': band_ratios,
            'peak_frequencies': peak_freqs.tolist(),
            'peak_amplitudes': peak_amps.tolist(),
            'spectral_centroid': centroid,
            'spectral_bandwidth': bandwidth,
            'spectral_flatness': flatness,
            'dominant_band': max(band_ratios, key=band_ratios.get),
            'total_energy': total_energy
        }
    
    # ==================== FREQUENCY SLOT NEGOTIATION ====================
    
    def negotiate_frequency_slots(self, 
                                  y_a: np.ndarray, 
                                  y_b: np.ndarray) -> Dict:
        """
        Intelligent frequency slot negotiation between two songs.
        
        Determines which song "owns" each frequency band during transition
        and creates surgical EQ curves to prevent clashes.
        
        This is what professional DJs try to do by ear - we do it precisely.
        """
        spectrum_a = self.analyze_spectrum(y_a)
        spectrum_b = self.analyze_spectrum(y_b)
        
        # Analyze conflicts per band
        conflicts = {}
        ownership = {}
        eq_curves = {}
        
        for band_name, (low, high) in self.frequency_bands.items():
            energy_a = spectrum_a['band_energies'].get(band_name, 0)
            energy_b = spectrum_b['band_energies'].get(band_name, 0)
            
            # Calculate conflict score (both songs competing for same space)
            total = energy_a + energy_b + 1e-10
            ratio_a = energy_a / total
            ratio_b = energy_b / total
            
            # Conflict is highest when both have similar energy
            conflict = 2 * min(ratio_a, ratio_b)
            conflicts[band_name] = float(conflict)
            
            # Determine ownership (which song gets this band)
            # Factor in: energy level, importance for each song
            importance_a = self._calculate_band_importance(spectrum_a, band_name)
            importance_b = self._calculate_band_importance(spectrum_b, band_name)
            
            # Weighted scoring
            score_a = energy_a * importance_a
            score_b = energy_b * importance_b
            
            if score_a > score_b * 1.2:  # A owns this band (with margin)
                ownership[band_name] = 'A'
            elif score_b > score_a * 1.2:  # B owns this band
                ownership[band_name] = 'B'
            else:  # Shared - need negotiation
                ownership[band_name] = 'shared'
        
        # Create EQ curves based on ownership
        eq_curves = self._create_negotiated_eq_curves(
            conflicts, ownership, spectrum_a, spectrum_b
        )
        
        return {
            'conflicts': conflicts,
            'ownership': ownership,
            'eq_curves': eq_curves,
            'overall_conflict': float(np.mean(list(conflicts.values()))),
            'critical_conflicts': {k: v for k, v in conflicts.items() if v > 0.6}
        }
    
    def _calculate_band_importance(self, spectrum: Dict, band_name: str) -> float:
        """Calculate how important a frequency band is for this song."""
        band_ratio = spectrum['band_ratios'].get(band_name, 0)
        
        # Boost importance if it's near the dominant band
        dominant = spectrum.get('dominant_band', '')
        if band_name == dominant:
            return band_ratio * 1.5
        
        # Bass and low-mid are critical for dance music
        if band_name in ['bass', 'low_mid', 'sub_bass']:
            return band_ratio * 1.3
        
        # Presence is critical for vocals
        if band_name in ['mid', 'upper_mid']:
            return band_ratio * 1.2
        
        return band_ratio
    
    def _create_negotiated_eq_curves(self, 
                                     conflicts: Dict,
                                     ownership: Dict,
                                     spectrum_a: Dict,
                                     spectrum_b: Dict,
                                     transition_points: int = 100) -> Dict:
        """Create EQ automation curves based on frequency slot negotiation."""
        eq_a = {band: np.ones(transition_points) for band in self.frequency_bands}
        eq_b = {band: np.ones(transition_points) for band in self.frequency_bands}
        
        t = np.linspace(0, 1, transition_points)
        
        for band_name, owner in ownership.items():
            conflict = conflicts.get(band_name, 0)
            
            if conflict < 0.2:
                # Low conflict - normal crossfade
                eq_a[band_name] = np.sqrt(1 - t)
                eq_b[band_name] = np.sqrt(t)
            elif owner == 'A':
                # A owns this band - fade A slower, B faster
                eq_a[band_name] = np.sqrt(1 - t * 0.8)  # A fades to 0.45
                eq_b[band_name] = np.sqrt(t) * 0.5  # B stays at 0.5x
            elif owner == 'B':
                # B owns this band - fade A faster, B takes over
                eq_a[band_name] = np.sqrt((1 - t) * 0.5)  # A drops to 0.5x
                eq_b[band_name] = np.sqrt(t * 1.2)  # B ramps up faster
                eq_b[band_name] = np.clip(eq_b[band_name], 0, 1.2)
            else:  # shared
                # Equal fade but with mid-dip to avoid conflict peak
                mid_point = transition_points // 2
                dip = np.ones(transition_points)
                dip[mid_point-10:mid_point+10] = 0.7  # 30% dip at midpoint
                dip = gaussian_filter1d(dip, 5)  # Smooth
                
                eq_a[band_name] = np.sqrt(1 - t) * dip
                eq_b[band_name] = np.sqrt(t) * dip
        
        return {
            'eq_a': {k: v.tolist() for k, v in eq_a.items()},
            'eq_b': {k: v.tolist() for k, v in eq_b.items()}
        }
    
    def apply_spectral_negotiation(self,
                                   y_a: np.ndarray,
                                   y_b: np.ndarray,
                                   eq_curves: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply negotiated EQ curves to audio segments.
        """
        n_samples = min(len(y_a), len(y_b))
        
        # Process through multiband EQ
        y_a_processed = self._apply_multiband_eq(
            y_a[:n_samples], 
            eq_curves.get('eq_a', {})
        )
        y_b_processed = self._apply_multiband_eq(
            y_b[:n_samples], 
            eq_curves.get('eq_b', {})
        )
        
        return y_a_processed, y_b_processed
    
    def _apply_multiband_eq(self, y: np.ndarray, eq_curves: Dict) -> np.ndarray:
        """Apply multiband EQ curves to audio."""
        if len(eq_curves) == 0:
            return y
        
        n_samples = len(y)
        result = np.zeros_like(y)
        
        for band_name, curve in eq_curves.items():
            if band_name not in self.frequency_bands:
                continue
            
            low, high = self.frequency_bands[band_name]
            
            # Create bandpass filter
            nyq = self.sr / 2
            low_norm = max(low / nyq, 0.001)
            high_norm = min(high / nyq, 0.999)
            
            if high_norm <= low_norm:
                continue
            
            try:
                sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
                band_audio = signal.sosfilt(sos, y)
            except:
                continue
            
            # Apply gain curve
            gain_curve = np.interp(
                np.linspace(0, len(curve) - 1, n_samples),
                np.arange(len(curve)),
                curve
            )
            
            if y.ndim == 1:
                result += band_audio * gain_curve
            else:
                result += band_audio * gain_curve[:, np.newaxis]
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 0.95:
            result = result * (0.95 / max_val)
        
        return result
    
    # ==================== HARMONIC RESONANCE BOOSTING ====================
    
    def find_harmonic_resonances(self, y_a: np.ndarray, y_b: np.ndarray) -> Dict:
        """
        Find frequencies where both songs resonate harmonically.
        
        These are frequencies that sound good together and can be boosted
        during the transition to create a sense of unity.
        """
        # Get spectrograms
        S_a = np.abs(librosa.stft(y_a, n_fft=self.n_fft, hop_length=self.hop_length))
        S_b = np.abs(librosa.stft(y_b, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # Average spectra
        avg_a = np.mean(S_a, axis=1)
        avg_b = np.mean(S_b, axis=1)
        
        # Find peaks in both
        peaks_a, _ = signal.find_peaks(avg_a, height=np.max(avg_a) * 0.1)
        peaks_b, _ = signal.find_peaks(avg_b, height=np.max(avg_b) * 0.1)
        
        freq_a = freqs[peaks_a]
        freq_b = freqs[peaks_b]
        
        # Find harmonically related frequencies
        resonances = []
        
        for f_a in freq_a[:30]:  # Top 30 peaks
            for f_b in freq_b[:30]:
                # Check for harmonic relationships
                ratio = f_a / f_b if f_b > 0 else 0
                
                # Look for simple ratios: 1:1, 2:1, 3:2, 4:3, 5:4
                harmonic_ratios = [1.0, 2.0, 0.5, 1.5, 0.667, 1.333, 0.75, 1.25]
                
                for hr in harmonic_ratios:
                    if abs(ratio - hr) < 0.05:  # 5% tolerance
                        resonances.append({
                            'freq_a': float(f_a),
                            'freq_b': float(f_b),
                            'ratio': float(ratio),
                            'harmonic_type': self._classify_harmonic(hr)
                        })
                        break
        
        # Sort by musical importance
        resonances = sorted(resonances, key=lambda x: x['freq_a'])[:20]
        
        return {
            'resonances': resonances,
            'resonance_count': len(resonances),
            'resonance_strength': len(resonances) / 20.0,  # Normalized
            'boost_frequencies': [r['freq_a'] for r in resonances[:10]]
        }
    
    def _classify_harmonic(self, ratio: float) -> str:
        """Classify harmonic relationship."""
        ratios = {
            1.0: 'unison',
            2.0: 'octave_up',
            0.5: 'octave_down',
            1.5: 'perfect_fifth',
            0.667: 'perfect_fourth',
            1.333: 'perfect_fourth_up',
            0.75: 'major_third_down',
            1.25: 'major_third_up'
        }
        
        for r, name in ratios.items():
            if abs(ratio - r) < 0.05:
                return name
        return 'other'
    
    def create_resonance_boost(self, 
                               resonances: Dict,
                               transition_samples: int,
                               boost_db: float = 3.0) -> Dict:
        """
        Create EQ curves that boost harmonic resonances during transition.
        """
        boost_freqs = resonances.get('boost_frequencies', [])
        
        if len(boost_freqs) == 0:
            return {'boost_curve': np.ones(transition_samples).tolist()}
        
        # Create peaked boost at transition midpoint
        t = np.linspace(0, 1, transition_samples)
        
        # Gaussian peak at 50% through transition
        boost_envelope = np.exp(-((t - 0.5) ** 2) / (2 * 0.15 ** 2))
        
        # Convert dB to linear gain
        boost_linear = 10 ** (boost_db * boost_envelope / 20)
        
        return {
            'boost_curve': boost_linear.tolist(),
            'boost_frequencies': boost_freqs,
            'peak_boost_db': boost_db
        }
    
    # ==================== SPECTRAL MORPHING ====================
    
    def create_spectral_morph(self, 
                              y_a: np.ndarray, 
                              y_b: np.ndarray,
                              morph_type: str = 'linear') -> Dict:
        """
        Create spectral morphing data for smooth envelope transition.
        
        This morphs the spectral "shape" of Song A into Song B,
        making the transition feel like a natural transformation.
        """
        # Get average spectral envelopes
        S_a = np.abs(librosa.stft(y_a, n_fft=self.n_fft, hop_length=self.hop_length))
        S_b = np.abs(librosa.stft(y_b, n_fft=self.n_fft, hop_length=self.hop_length))
        
        envelope_a = np.mean(S_a, axis=1)
        envelope_b = np.mean(S_b, axis=1)
        
        # Normalize envelopes
        envelope_a = envelope_a / (np.max(envelope_a) + 1e-10)
        envelope_b = envelope_b / (np.max(envelope_b) + 1e-10)
        
        # Create morph stages (10 stages)
        n_stages = 10
        morph_stages = []
        
        for i in range(n_stages + 1):
            t = i / n_stages
            
            if morph_type == 'linear':
                morph_ratio = t
            elif morph_type == 'smooth':
                morph_ratio = 0.5 * (1 - np.cos(np.pi * t))
            elif morph_type == 'early':
                morph_ratio = 1 - (1 - t) ** 2
            elif morph_type == 'late':
                morph_ratio = t ** 2
            else:
                morph_ratio = t
            
            # Interpolate envelopes
            morphed = envelope_a * (1 - morph_ratio) + envelope_b * morph_ratio
            morph_stages.append(morphed.tolist())
        
        return {
            'morph_stages': morph_stages,
            'n_stages': n_stages,
            'morph_type': morph_type,
            'envelope_a': envelope_a.tolist(),
            'envelope_b': envelope_b.tolist()
        }
    
    def apply_spectral_morph(self, 
                             y: np.ndarray, 
                             morph_data: Dict,
                             progress: float) -> np.ndarray:
        """
        Apply spectral morphing at a given progress point.
        
        Args:
            y: Input audio
            morph_data: Data from create_spectral_morph
            progress: 0.0 = start (Song A envelope), 1.0 = end (Song B envelope)
        """
        stages = morph_data.get('morph_stages', [])
        n_stages = morph_data.get('n_stages', len(stages) - 1)
        
        if n_stages == 0 or len(stages) < 2:
            return y
        
        # Get interpolated target envelope
        stage_idx = int(progress * n_stages)
        stage_idx = min(stage_idx, len(stages) - 2)
        
        stage_progress = (progress * n_stages) - stage_idx
        
        envelope_a = np.array(stages[stage_idx])
        envelope_b = np.array(stages[stage_idx + 1])
        target_envelope = envelope_a * (1 - stage_progress) + envelope_b * stage_progress
        
        # Apply envelope shaping via STFT
        S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        S_mag = np.abs(S)
        S_phase = np.angle(S)
        
        # Current envelope
        current_envelope = np.mean(S_mag, axis=1)
        current_envelope = current_envelope / (np.max(current_envelope) + 1e-10)
        
        # Calculate gain to morph towards target
        gain = target_envelope / (current_envelope + 1e-10)
        gain = np.clip(gain, 0.5, 2.0)  # Limit extreme changes
        
        # Apply gain
        S_mag_morphed = S_mag * gain[:, np.newaxis]
        
        # Reconstruct
        S_morphed = S_mag_morphed * np.exp(1j * S_phase)
        y_morphed = librosa.istft(S_morphed, hop_length=self.hop_length)
        
        # Match original length
        if len(y_morphed) > len(y):
            y_morphed = y_morphed[:len(y)]
        elif len(y_morphed) < len(y):
            y_morphed = np.pad(y_morphed, (0, len(y) - len(y_morphed)))
        
        return y_morphed
    
    # ==================== MASKING-AWARE MIXING ====================
    
    def analyze_masking(self, y_a: np.ndarray, y_b: np.ndarray) -> Dict:
        """
        Psychoacoustic analysis of frequency masking between two signals.
        
        Masking occurs when one sound makes another inaudible.
        We detect and prevent this.
        """
        # Get spectrograms
        S_a = np.abs(librosa.stft(y_a, n_fft=self.n_fft, hop_length=self.hop_length))
        S_b = np.abs(librosa.stft(y_b, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # Calculate masking curves based on psychoacoustic principles
        masking_analysis = []
        
        # Analyze in time chunks
        n_frames = min(S_a.shape[1], S_b.shape[1])
        chunk_size = max(1, n_frames // 10)
        
        for chunk_idx in range(0, n_frames, chunk_size):
            end_idx = min(chunk_idx + chunk_size, n_frames)
            
            chunk_a = np.mean(S_a[:, chunk_idx:end_idx], axis=1)
            chunk_b = np.mean(S_b[:, chunk_idx:end_idx], axis=1)
            
            # Calculate masking threshold
            masking_threshold = self._calculate_masking_threshold(chunk_a, freqs)
            
            # Find masked frequencies in B
            masked_in_b = chunk_b < masking_threshold
            mask_ratio = np.sum(masked_in_b) / len(masked_in_b)
            
            masking_analysis.append({
                'chunk_idx': chunk_idx,
                'mask_ratio': float(mask_ratio),
                'severe_masking': float(mask_ratio) > 0.3
            })
        
        # Overall masking score
        avg_masking = np.mean([m['mask_ratio'] for m in masking_analysis])
        
        return {
            'overall_masking': float(avg_masking),
            'temporal_analysis': masking_analysis,
            'masking_severity': 'high' if avg_masking > 0.4 else 'medium' if avg_masking > 0.2 else 'low',
            'recommended_action': self._recommend_masking_action(avg_masking)
        }
    
    def _calculate_masking_threshold(self, spectrum: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """
        Calculate frequency masking threshold using simplified psychoacoustic model.
        
        Based on simultaneous masking principles.
        """
        # Convert to dB
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        
        # Create masking curve (spreading function)
        threshold = np.zeros_like(spectrum)
        
        for i, (freq, level) in enumerate(zip(freqs, spectrum_db)):
            if level < -60:
                continue
            
            # Spreading function (simplified bark-scale)
            for j, f in enumerate(freqs):
                if f == 0:
                    continue
                
                # Distance in critical bands (simplified)
                distance = abs(np.log2(freq / f + 1e-10))
                
                # Masking spread
                if distance < 1:  # Within one octave
                    spread_db = level - 20 * distance
                elif distance < 2:  # Within two octaves
                    spread_db = level - 40 - 10 * (distance - 1)
                else:
                    spread_db = -100
                
                threshold[j] = max(threshold[j], 10 ** (spread_db / 20))
        
        return threshold
    
    def _recommend_masking_action(self, masking_score: float) -> str:
        """Recommend action based on masking analysis."""
        if masking_score > 0.5:
            return 'aggressive_eq_carving'
        elif masking_score > 0.3:
            return 'bass_swap_with_mid_cut'
        elif masking_score > 0.15:
            return 'gentle_eq_adjustment'
        else:
            return 'standard_crossfade'
    
    def create_anti_masking_eq(self, 
                               masking_data: Dict,
                               transition_samples: int) -> Dict:
        """
        Create EQ curves that prevent frequency masking.
        """
        severity = masking_data.get('masking_severity', 'low')
        
        t = np.linspace(0, 1, transition_samples)
        
        if severity == 'high':
            # Aggressive carving
            # Cut outgoing lows/mids earlier
            low_cut_a = 1.0 - t * 1.5  # Faster fade
            low_cut_a = np.clip(low_cut_a, 0.1, 1.0)
            
            # Boost incoming presence later
            presence_boost_b = np.where(t > 0.5, 1.0 + (t - 0.5) * 0.4, 1.0)
            
            return {
                'outgoing_low_cut': low_cut_a.tolist(),
                'outgoing_mid_cut': (1.0 - t * 0.6).tolist(),
                'incoming_presence_boost': presence_boost_b.tolist(),
                'technique': 'aggressive'
            }
        
        elif severity == 'medium':
            # Moderate adjustment
            return {
                'outgoing_low_cut': np.sqrt(1 - t).tolist(),
                'outgoing_mid_cut': (1.0 - t * 0.3).tolist(),
                'incoming_presence_boost': np.ones(transition_samples).tolist(),
                'technique': 'moderate'
            }
        
        else:
            # Standard crossfade
            return {
                'outgoing_low_cut': np.sqrt(1 - t).tolist(),
                'outgoing_mid_cut': np.sqrt(1 - t).tolist(),
                'incoming_presence_boost': np.ones(transition_samples).tolist(),
                'technique': 'standard'
            }
