"""
High-Accuracy Transition Detection Module

Detects transitions in DJ mixes by sensing NEW musical content appearing,
not just "audio changed." Uses human-like perception:

1. Beat/Bar-Aligned Analysis - Analyze at musical boundaries
2. Repetition Structure - Detect when repeating patterns break
3. Source Complexity - Detect when two songs overlap (layering)
4. Vocal/Melodic Novelty - Detect new melodic content appearing
5. Frequency Band Independence - Detect when bands decouple (bass swap)

The key insight: transitions involve FOREIGN content entering, not just change.
"""

import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, correlate
from scipy.stats import entropy
from scipy.spatial.distance import cosine


@dataclass
class DetectedTransition:
    """Represents a detected transition in a mix"""
    start_sec: float
    end_sec: float
    duration_sec: float
    confidence: float
    detection_method: str
    bar_number: Optional[int] = None
    novelty_score: Optional[float] = None
    bpm_change: Optional[float] = None
    source_count_change: Optional[float] = None


@dataclass 
class BarSegment:
    """A single bar of music (typically 4 beats)"""
    bar_index: int
    start_sec: float
    end_sec: float
    start_sample: int
    end_sample: int
    features: Optional[Dict] = None


class TransitionDetector:
    """
    High-accuracy transition detector using human-like perception.
    
    Key principle: Detect NEW content appearing, not just change.
    """
    
    def __init__(self, sr: int = 22050, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length
        
    def detect_all(self, audio_path: str) -> List[DetectedTransition]:
        """Detect all transitions in an audio file."""
        print("  Loading audio for analysis...")
        y, sr = librosa.load(audio_path, sr=self.sr)
        return self.detect_transitions(y, sr)
    
    def detect_transitions(self, y: np.ndarray, sr: int) -> List[DetectedTransition]:
        """Main detection method."""
        duration = len(y) / sr
        print(f"  Analyzing {duration/60:.1f} minutes of audio...")
        
        # Step 1: Beat and bar alignment (foundation for everything)
        print("  Step 1: Detecting beats and bars...")
        bars = self._detect_bars(y, sr)
        print(f"    Found {len(bars)} bars")
        
        if len(bars) < 8:
            print("    Warning: Too few bars detected, using fallback")
            return self._fallback_detection(y, sr, duration)
        
        # Step 2: Extract features for each bar
        print("  Step 2: Extracting bar features...")
        bars = self._extract_bar_features(y, sr, bars)
        
        # Step 3: Run all detection methods (bar-aligned)
        print("  Step 3: Running repetition analysis...")
        repetition_scores = self._detect_repetition_breaks(bars)
        
        print("  Step 4: Running complexity tracking...")
        complexity_scores = self._detect_complexity_changes(y, sr, bars)
        
        print("  Step 5: Running vocal/melodic novelty detection...")
        vocal_scores = self._detect_vocal_novelty(y, sr, bars)
        
        print("  Step 6: Running frequency band independence...")
        band_scores = self._detect_band_independence(y, sr, bars)
        
        # Step 7: Combine with smart voting
        print("  Step 7: Combining with musical context voting...")
        transitions = self._musical_voting(
            bars, 
            repetition_scores, 
            complexity_scores,
            vocal_scores,
            band_scores,
            duration
        )
        
        print(f"  Found {len(transitions)} transitions")
        return transitions
    
    # =========================================================================
    # STEP 1: BEAT/BAR-ALIGNED SEGMENTATION
    # =========================================================================
    
    def _detect_bars(self, y: np.ndarray, sr: int) -> List[BarSegment]:
        """
        Detect beats and group into bars (4 beats = 1 bar typically).
        
        This is the foundation - all analysis happens at bar boundaries.
        """
        # Detect tempo and beats
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=self.hop_length
        )
        
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)
        beat_samples = librosa.frames_to_samples(beat_frames, hop_length=self.hop_length)
        
        # Group beats into bars (4 beats per bar for 4/4 time)
        beats_per_bar = 4
        bars = []
        
        for i in range(0, len(beat_times) - beats_per_bar, beats_per_bar):
            bar_start_time = beat_times[i]
            bar_end_time = beat_times[i + beats_per_bar] if i + beats_per_bar < len(beat_times) else len(y) / sr
            bar_start_sample = beat_samples[i]
            bar_end_sample = beat_samples[i + beats_per_bar] if i + beats_per_bar < len(beat_samples) else len(y)
            
            bars.append(BarSegment(
                bar_index=len(bars),
                start_sec=bar_start_time,
                end_sec=bar_end_time,
                start_sample=int(bar_start_sample),
                end_sample=int(bar_end_sample)
            ))
        
        return bars
    
    def _extract_bar_features(self, y: np.ndarray, sr: int, bars: List[BarSegment]) -> List[BarSegment]:
        """Extract audio features for each bar."""
        for bar in bars:
            # Get audio segment for this bar
            segment = y[bar.start_sample:bar.end_sample]
            
            if len(segment) < 1024:
                bar.features = None
                continue
            
            # Compute features
            # 1. Chromagram (harmonic content)
            chroma = librosa.feature.chroma_cqt(y=segment, sr=sr, hop_length=self.hop_length)
            chroma_mean = np.mean(chroma, axis=1)
            
            # 2. MFCCs (timbral content)
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # 3. Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=segment))
            
            # 4. RMS energy
            rms = np.mean(librosa.feature.rms(y=segment))
            
            # 5. Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(segment))
            
            bar.features = {
                'chroma': chroma_mean,
                'mfcc': mfcc_mean,
                'spectral_centroid': spectral_centroid,
                'spectral_flatness': spectral_flatness,
                'rms': rms,
                'zcr': zcr,
                'combined': np.concatenate([chroma_mean, mfcc_mean, [spectral_centroid/10000, spectral_flatness, rms*10, zcr]])
            }
        
        return bars
    
    # =========================================================================
    # STEP 2: REPETITION STRUCTURE ANALYSIS
    # =========================================================================
    
    def _detect_repetition_breaks(self, bars: List[BarSegment]) -> np.ndarray:
        """
        Detect when repeating patterns break.
        
        Songs repeat every 4, 8, or 16 bars. When we compare bar N to bar N-8
        and they're suddenly different, something changed.
        """
        n_bars = len(bars)
        scores = np.zeros(n_bars)
        
        # Compare each bar to bars 4, 8, and 16 bars ago
        lookbacks = [4, 8, 16]
        
        for i, bar in enumerate(bars):
            if bar.features is None:
                continue
            
            dissimilarities = []
            
            for lookback in lookbacks:
                if i >= lookback and bars[i - lookback].features is not None:
                    # Compare combined features
                    current = bar.features['combined']
                    previous = bars[i - lookback].features['combined']
                    
                    # Cosine distance (0 = identical, 1 = completely different)
                    dist = cosine(current, previous)
                    dissimilarities.append(dist)
            
            if dissimilarities:
                # Take the maximum dissimilarity (if any lookback shows change)
                scores[i] = max(dissimilarities)
        
        # Smooth slightly
        scores = gaussian_filter1d(scores, sigma=1)
        
        # Normalize
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return scores
    
    # =========================================================================
    # STEP 3: SOURCE COMPLEXITY TRACKING
    # =========================================================================
    
    def _detect_complexity_changes(self, y: np.ndarray, sr: int, bars: List[BarSegment]) -> np.ndarray:
        """
        Detect when spectral complexity increases (two songs overlapping).
        
        During a transition, there are effectively TWO songs playing,
        which increases the spectral entropy and "fullness."
        """
        n_bars = len(bars)
        scores = np.zeros(n_bars)
        
        for i, bar in enumerate(bars):
            segment = y[bar.start_sample:bar.end_sample]
            
            if len(segment) < 1024:
                continue
            
            # Compute spectrogram
            S = np.abs(librosa.stft(segment, hop_length=self.hop_length))
            
            # 1. Spectral entropy (higher = more complex/random = possible overlap)
            S_norm = S / (S.sum(axis=0, keepdims=True) + 1e-8)
            spectral_entropy = np.mean([entropy(col + 1e-8) for col in S_norm.T])
            
            # 2. Spectral flatness (higher = more noise-like = more sources)
            flatness = np.mean(librosa.feature.spectral_flatness(S=S))
            
            # 3. Frequency band density (how many bands are active)
            band_activity = np.mean(S > np.percentile(S, 50))
            
            # Combine metrics
            scores[i] = 0.4 * spectral_entropy + 0.3 * flatness + 0.3 * band_activity
        
        # Compute rate of change (we want INCREASES in complexity)
        complexity_change = np.zeros(n_bars)
        window = 4  # Compare to 4 bars ago
        
        for i in range(window, n_bars):
            prev_avg = np.mean(scores[i-window:i])
            if prev_avg > 0:
                complexity_change[i] = max(0, (scores[i] - prev_avg) / prev_avg)
        
        # Normalize
        if complexity_change.max() > 0:
            complexity_change = complexity_change / complexity_change.max()
        
        return complexity_change
    
    # =========================================================================
    # STEP 4: VOCAL/MELODIC NOVELTY DETECTION
    # =========================================================================
    
    def _detect_vocal_novelty(self, y: np.ndarray, sr: int, bars: List[BarSegment]) -> np.ndarray:
        """
        Detect when NEW melodic/vocal content appears.
        
        This is the most reliable indicator - a new voice or melody
        means a new song is entering.
        """
        n_bars = len(bars)
        scores = np.zeros(n_bars)
        
        # Harmonic-percussive separation to isolate melodic content
        y_harmonic, _ = librosa.effects.hpss(y)
        
        # Extract harmonic features for each bar
        bar_harmonic_features = []
        
        for bar in bars:
            segment = y_harmonic[bar.start_sample:bar.end_sample]
            
            if len(segment) < 1024:
                bar_harmonic_features.append(None)
                continue
            
            # Chromagram (captures melody/harmony)
            chroma = librosa.feature.chroma_cqt(y=segment, sr=sr, hop_length=self.hop_length)
            chroma_mean = np.mean(chroma, axis=1)
            
            # MFCCs of harmonic content (captures vocal timbre)
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Pitch tracking (fundamental frequency changes)
            pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
            pitch_mean = np.mean(pitches[magnitudes > np.percentile(magnitudes, 80)])
            
            if np.isnan(pitch_mean):
                pitch_mean = 0
            
            bar_harmonic_features.append({
                'chroma': chroma_mean,
                'mfcc': mfcc_mean,
                'pitch': pitch_mean,
                'combined': np.concatenate([chroma_mean, mfcc_mean, [pitch_mean/1000]])
            })
        
        # Detect novelty: compare current bar to recent history
        history_window = 8  # Look back 8 bars
        
        for i in range(history_window, n_bars):
            if bar_harmonic_features[i] is None:
                continue
            
            current = bar_harmonic_features[i]['combined']
            
            # Build "expected" pattern from history
            history_features = []
            for j in range(i - history_window, i):
                if bar_harmonic_features[j] is not None:
                    history_features.append(bar_harmonic_features[j]['combined'])
            
            if len(history_features) < 2:
                continue
            
            history_mean = np.mean(history_features, axis=0)
            history_std = np.std(history_features, axis=0) + 1e-8
            
            # How many standard deviations away is the current bar?
            z_score = np.mean(np.abs((current - history_mean) / history_std))
            
            # High z-score = current bar is very different from recent history
            scores[i] = min(1.0, z_score / 3.0)  # Cap at 3 std devs
        
        return scores
    
    # =========================================================================
    # STEP 5: FREQUENCY BAND INDEPENDENCE
    # =========================================================================
    
    def _detect_band_independence(self, y: np.ndarray, sr: int, bars: List[BarSegment]) -> np.ndarray:
        """
        Detect when frequency bands change independently.
        
        During a bass swap transition, the bass changes while mids/highs stay the same.
        This indicates content from two different sources.
        """
        n_bars = len(bars)
        scores = np.zeros(n_bars)
        
        # Define frequency bands
        bands = [
            (20, 150, 'sub_bass'),
            (150, 400, 'bass'),
            (400, 2000, 'mids'),
            (2000, 8000, 'highs'),
        ]
        
        # Extract band energies for each bar
        bar_band_energies = []
        
        for bar in bars:
            segment = y[bar.start_sample:bar.end_sample]
            
            if len(segment) < 1024:
                bar_band_energies.append(None)
                continue
            
            # Compute spectrogram
            S = np.abs(librosa.stft(segment, hop_length=self.hop_length))
            freqs = librosa.fft_frequencies(sr=sr)
            
            band_energy = {}
            for low, high, name in bands:
                mask = (freqs >= low) & (freqs < high)
                if np.any(mask):
                    band_energy[name] = np.mean(S[mask, :])
                else:
                    band_energy[name] = 0
            
            bar_band_energies.append(band_energy)
        
        # Detect when bands change independently
        for i in range(4, n_bars):
            if bar_band_energies[i] is None or bar_band_energies[i-4] is None:
                continue
            
            current = bar_band_energies[i]
            previous = bar_band_energies[i-4]
            
            # Calculate change in each band
            changes = []
            for _, _, name in bands:
                if previous[name] > 0:
                    change = abs(current[name] - previous[name]) / (previous[name] + 1e-8)
                    changes.append(change)
                else:
                    changes.append(0)
            
            if len(changes) < 2:
                continue
            
            # Independence score: high when some bands change a lot while others don't
            # (standard deviation of changes)
            changes = np.array(changes)
            independence = np.std(changes)
            
            # Also check if the total energy is similar but distribution changed
            total_current = sum(current.values())
            total_previous = sum(previous.values())
            total_change = abs(total_current - total_previous) / (total_previous + 1e-8)
            
            # High independence + low total change = bass swap or similar
            if total_change < 0.3:  # Total energy similar
                scores[i] = independence
            else:
                scores[i] = independence * 0.5  # Reduce score if total energy changed
        
        # Normalize
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    # =========================================================================
    # STEP 6: MUSICAL CONTEXT VOTING
    # =========================================================================
    
    def _musical_voting(self, 
                        bars: List[BarSegment],
                        repetition_scores: np.ndarray,
                        complexity_scores: np.ndarray,
                        vocal_scores: np.ndarray,
                        band_scores: np.ndarray,
                        duration: float) -> List[DetectedTransition]:
        """
        Combine all detection methods with musical context awareness.
        
        Key insight: Real transitions happen at phrase boundaries (every 8 or 16 bars)
        and multiple indicators should agree.
        """
        n_bars = len(bars)
        
        # Weight the different methods
        weights = {
            'repetition': 2.0,   # Pattern breaks are very reliable
            'vocal': 2.0,       # New melodic content is very reliable
            'complexity': 1.5,  # Two sources overlapping
            'band': 1.0,        # Bass swaps
        }
        
        # Combine scores
        combined = np.zeros(n_bars)
        for i in range(n_bars):
            combined[i] = (
                weights['repetition'] * repetition_scores[i] +
                weights['vocal'] * vocal_scores[i] +
                weights['complexity'] * complexity_scores[i] +
                weights['band'] * band_scores[i]
            )
        
        # Normalize
        combined = combined / sum(weights.values())
        
        # Apply phrase boundary preference (transitions usually happen at bar 0, 8, 16, 24...)
        for i in range(n_bars):
            bar_in_phrase = i % 8
            # Boost scores at phrase boundaries (bar 0 or 4 of an 8-bar phrase)
            if bar_in_phrase == 0:
                combined[i] *= 1.3
            elif bar_in_phrase == 4:
                combined[i] *= 1.1
        
        # Find peaks (potential transitions)
        min_bars_between = 8  # Minimum 8 bars (~15-20 seconds) between transitions
        
        peaks, properties = find_peaks(
            combined,
            distance=min_bars_between,
            prominence=0.15,
            height=0.2
        )
        
        # If too few peaks, lower threshold
        if len(peaks) < 3 and len(bars) > 30:
            peaks, properties = find_peaks(
                combined,
                distance=min_bars_between,
                prominence=0.1,
                height=0.15
            )
        
        # Convert to transitions
        transitions = []
        
        for peak_idx in peaks:
            bar = bars[peak_idx]
            confidence = combined[peak_idx]
            
            # Determine which methods contributed
            methods = []
            if repetition_scores[peak_idx] > 0.2:
                methods.append('repetition')
            if vocal_scores[peak_idx] > 0.2:
                methods.append('vocal')
            if complexity_scores[peak_idx] > 0.2:
                methods.append('complexity')
            if band_scores[peak_idx] > 0.2:
                methods.append('band')
            
            # Estimate transition duration (typically 2-8 bars)
            trans_duration_bars = 4
            end_bar_idx = min(peak_idx + trans_duration_bars, n_bars - 1)
            
            transitions.append(DetectedTransition(
                start_sec=bar.start_sec,
                end_sec=bars[end_bar_idx].end_sec,
                duration_sec=bars[end_bar_idx].end_sec - bar.start_sec,
                confidence=confidence,
                detection_method='+'.join(methods) if methods else 'combined',
                bar_number=peak_idx,
                novelty_score=vocal_scores[peak_idx]
            ))
        
        # Filter by minimum gap (60 seconds between transitions)
        filtered = []
        min_gap = 60.0
        
        for trans in transitions:
            if not filtered or trans.start_sec - filtered[-1].start_sec >= min_gap:
                filtered.append(trans)
            elif trans.confidence > filtered[-1].confidence:
                filtered[-1] = trans
        
        return filtered
    
    def _fallback_detection(self, y: np.ndarray, sr: int, duration: float) -> List[DetectedTransition]:
        """Fallback detection when beat detection fails."""
        # Simple energy-based detection
        hop = self.hop_length
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        
        # Smooth heavily
        rms_smooth = gaussian_filter1d(rms, sigma=50)
        
        # Find significant changes
        rms_diff = np.abs(np.gradient(rms_smooth))
        rms_diff = gaussian_filter1d(rms_diff, sigma=20)
        
        if rms_diff.max() > rms_diff.min():
            rms_diff = (rms_diff - rms_diff.min()) / (rms_diff.max() - rms_diff.min())
        
        min_frames = int(60 * sr / hop)  # 60 seconds
        peaks, _ = find_peaks(rms_diff, distance=min_frames, height=0.2)
        
        times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
        
        transitions = []
        for t in times:
            transitions.append(DetectedTransition(
                start_sec=t,
                end_sec=min(t + 30, duration),
                duration_sec=30,
                confidence=0.5,
                detection_method='fallback_energy'
            ))
        
        return transitions
    
    def refine_transition_boundaries(
        self, 
        audio_path: str, 
        transition: DetectedTransition
    ) -> DetectedTransition:
        """Refine transition boundaries to align with beats."""
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length
        )
        
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
        
        if len(beat_times) > 0:
            start_idx = np.argmin(np.abs(beat_times - transition.start_sec))
            refined_start = beat_times[start_idx]
            
            end_idx = np.argmin(np.abs(beat_times - transition.end_sec))
            refined_end = beat_times[end_idx]
        else:
            refined_start = transition.start_sec
            refined_end = transition.end_sec
        
        return DetectedTransition(
            start_sec=refined_start,
            end_sec=refined_end,
            duration_sec=refined_end - refined_start,
            confidence=transition.confidence,
            detection_method=transition.detection_method,
            bar_number=transition.bar_number,
            novelty_score=transition.novelty_score,
            bpm_change=transition.bpm_change
        )
