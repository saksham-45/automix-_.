"""
Stem Morpher Engine — Progressive Stem-Level Audio Transformation

This module implements what no existing DJ software does: instead of simply fading
stems A→B by volume, it progressively *transforms* the audio content of each stem
from Song A into Song B. The listener hears the drums actually change pattern,
the bass actually shift notes, and the other instruments reshape their timbre
— all while the rest of the transition plays on top.

Architecture:
    StemMorpher is invoked between stem separation and orchestrate_mix. It receives
    stems_a and stems_b and returns *morphed* copies where the audio content of the
    chosen stems has been progressively warped toward the target.

Phases:
    1. Spectral Envelope Morphing (timbre transformation per stem)
    2. Onset-Level Drum Pattern Morphing (hit-by-hit replacement)
    3. Chroma-Based Harmonic Morphing (note-by-note pitch convergence)
    4. Phase Vocoder Cross-Synthesis (magnitude+phase interpolation)

Integration:
    superhuman_engine.py Stage 4.5 → between stem orchestration analysis and
    orchestrate_mix execution.

Author: automix engine
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.ndimage import gaussian_filter1d


class StemMorpher:
    """
    Progressive stem-level audio transformation engine.
    
    Transforms the actual audio content of a stem from Song A so it gradually
    sounds like the same stem from Song B. This goes far beyond volume crossfading:
    
    - Drums: individual hits swap one-by-one (hi-hats first, kick last)
    - Bass/Other: notes pitch-shift toward the target's chroma profile
    - All stems: spectral envelope morphs so the timbre transitions
    - Advanced: phase vocoder cross-synthesis blends the raw spectra
    """

    # Stem type → which morph techniques apply best
    STEM_TECHNIQUE_MAP = {
        'drums':  ['onset_replacement', 'spectral_envelope', 'multiband_crossfade'],
        'bass':   ['chroma_modulation', 'spectral_envelope', 'cross_synthesis'],
        'vocals': ['spectral_envelope', 'cross_synthesis'],
        'other':  ['chroma_modulation', 'spectral_envelope', 'cross_synthesis'],
    }

    # Priority order for "which stem to morph first" when strategy='best_match'
    MORPH_PRIORITY = ['drums', 'bass', 'other', 'vocals']

    def __init__(self, sr: int = 44100, n_fft: int = 4096):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = n_fft // 4

    # ====================================================================
    #  PUBLIC API
    # ====================================================================

    def analyze_stem_compatibility(
        self,
        stems_a: Dict[str, np.ndarray],
        stems_b: Dict[str, np.ndarray],
    ) -> Dict:
        """
        Score every stem pair (A.drums vs B.drums, etc.) on how well they'd
        morph into each other. Returns ordered list with best candidate first.
        
        Scoring dimensions:
            - spectral_similarity:  how close the timbral envelopes are
            - rhythmic_similarity:  onset density / pattern correlation
            - energy_ratio:         loudness balance (extreme mismatch = bad morph)
            - harmonic_overlap:     chroma correlation for pitched stems
        """
        common_stems = set(stems_a.keys()) & set(stems_b.keys())
        results: List[Dict] = []

        for stem_name in common_stems:
            sa = stems_a[stem_name]
            sb = stems_b[stem_name]
            if len(sa) == 0 or len(sb) == 0:
                continue

            # --- mono versions for analysis ---
            sa_mono = np.mean(sa, axis=1) if sa.ndim > 1 else sa
            sb_mono = np.mean(sb, axis=1) if sb.ndim > 1 else sb
            # take max 10s for speed
            max_len = min(len(sa_mono), len(sb_mono), 10 * self.sr)
            sa_mono = sa_mono[:max_len]
            sb_mono = sb_mono[:max_len]

            spec_sim = self._spectral_similarity(sa_mono, sb_mono)
            rhythm_sim = self._rhythmic_similarity(sa_mono, sb_mono)
            energy_a = float(np.sqrt(np.mean(sa_mono ** 2)))
            energy_b = float(np.sqrt(np.mean(sb_mono ** 2)))
            energy_ratio = min(energy_a, energy_b) / (max(energy_a, energy_b) + 1e-10)

            harmonic_overlap = 0.5
            if stem_name not in ('drums',):
                harmonic_overlap = self._chroma_correlation(sa_mono, sb_mono)

            # Weighted composite — drums care more about rhythm, melodic stems
            # care more about harmony.
            if stem_name == 'drums':
                score = (
                    0.20 * spec_sim +
                    0.45 * rhythm_sim +
                    0.20 * energy_ratio +
                    0.15 * harmonic_overlap
                )
            else:
                score = (
                    0.25 * spec_sim +
                    0.15 * rhythm_sim +
                    0.20 * energy_ratio +
                    0.40 * harmonic_overlap
                )

            results.append({
                'stem': stem_name,
                'score': float(score),
                'spectral_similarity': float(spec_sim),
                'rhythmic_similarity': float(rhythm_sim),
                'energy_ratio': float(energy_ratio),
                'harmonic_overlap': float(harmonic_overlap),
                'recommended_techniques': self.STEM_TECHNIQUE_MAP.get(
                    stem_name, ['spectral_envelope']
                ),
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return {
            'stem_rankings': results,
            'best_morph_candidate': results[0]['stem'] if results else None,
        }

    def create_morph_plan(
        self,
        compatibility: Dict,
        strategy: str = 'best_match',
        morph_depth: float = 0.8,
        techniques: Optional[List[str]] = None,
    ) -> Dict:
        """
        Build a morph plan: which stems to morph, which techniques, timing.
        
        Args:
            compatibility: output of analyze_stem_compatibility
            strategy: 'best_match' | 'drums_first' | 'all' | stem name
            morph_depth: 0.0 = no morphing, 1.0 = full content transformation
            techniques: override list of techniques (else use per-stem defaults)
        
        Returns:
            Plan dict with per-stem morph instructions.
        """
        rankings = compatibility.get('stem_rankings', [])
        if not rankings:
            return {'stems_to_morph': {}, 'morph_depth': morph_depth}

        stems_to_morph: Dict[str, Dict] = {}

        if strategy == 'all':
            for entry in rankings:
                stems_to_morph[entry['stem']] = {
                    'techniques': techniques or entry['recommended_techniques'],
                    'depth': morph_depth,
                    'score': entry['score'],
                }
        elif strategy == 'best_match':
            # Morph the best candidate at full depth; second best at half
            for i, entry in enumerate(rankings[:2]):
                depth = morph_depth if i == 0 else morph_depth * 0.5
                stems_to_morph[entry['stem']] = {
                    'techniques': techniques or entry['recommended_techniques'],
                    'depth': depth,
                    'score': entry['score'],
                }
        elif strategy == 'drums_first':
            for entry in rankings:
                if entry['stem'] == 'drums':
                    stems_to_morph['drums'] = {
                        'techniques': techniques or entry['recommended_techniques'],
                        'depth': morph_depth,
                        'score': entry['score'],
                    }
                    break
        else:
            # Strategy is a stem name
            for entry in rankings:
                if entry['stem'] == strategy:
                    stems_to_morph[strategy] = {
                        'techniques': techniques or entry['recommended_techniques'],
                        'depth': morph_depth,
                        'score': entry['score'],
                    }
                    break

        return {
            'stems_to_morph': stems_to_morph,
            'morph_depth': morph_depth,
            'strategy': strategy,
        }

    def apply_progressive_morph(
        self,
        stems_a: Dict[str, np.ndarray],
        stems_b: Dict[str, np.ndarray],
        morph_plan: Dict,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        The main entry point. Takes stems_a, applies progressive morphing
        based on the morph_plan, and returns the morphed stems_a dict
        (stems_b is unchanged).
        
        Each morphed stem gradually transforms from A → B content over its
        duration, so by the end it sounds close to stem B.
        
        Returns:
            (morphed_stems_a, morph_report)
        """
        morphed = {}
        report: Dict[str, Dict] = {}
        stems_to_morph = morph_plan.get('stems_to_morph', {})

        for stem_name, stem_audio in stems_a.items():
            if stem_name not in stems_to_morph or stem_name not in stems_b:
                morphed[stem_name] = stem_audio
                continue

            plan = stems_to_morph[stem_name]
            techniques = plan.get('techniques', ['spectral_envelope'])
            depth = float(plan.get('depth', 0.8))
            target = stems_b[stem_name]

            if len(stem_audio) == 0 or len(target) == 0:
                morphed[stem_name] = stem_audio
                continue

            stem_report: Dict = {
                'techniques_applied': [],
                'depth': depth,
            }

            # Apply techniques in priority order — each builds on the last
            current = stem_audio.copy()
            for tech in techniques:
                try:
                    if tech == 'spectral_envelope':
                        current = self._apply_spectral_envelope_morph(
                            current, target, depth
                        )
                        stem_report['techniques_applied'].append('spectral_envelope')

                    elif tech == 'onset_replacement' and stem_name == 'drums':
                        current = self._apply_onset_replacement(
                            current, target, depth
                        )
                        stem_report['techniques_applied'].append('onset_replacement')

                    elif tech == 'chroma_modulation' and stem_name != 'drums':
                        current = self._apply_chroma_modulation(
                            current, target, depth
                        )
                        stem_report['techniques_applied'].append('chroma_modulation')

                    elif tech == 'cross_synthesis':
                        current = self._apply_cross_synthesis(
                            current, target, depth
                        )
                        stem_report['techniques_applied'].append('cross_synthesis')

                    elif tech == 'multiband_crossfade':
                        current = self._apply_multiband_crossfade(
                            current, target, depth
                        )
                        stem_report['techniques_applied'].append('multiband_crossfade')

                except Exception as e:
                    stem_report[f'{tech}_error'] = str(e)

            morphed[stem_name] = current
            report[stem_name] = stem_report

        return morphed, report

    # ====================================================================
    #  PHASE 1:  SPECTRAL ENVELOPE MORPHING
    # ====================================================================

    def _apply_spectral_envelope_morph(
        self,
        source: np.ndarray,
        target: np.ndarray,
        depth: float,
    ) -> np.ndarray:
        """
        Reshape the spectral envelope of `source` toward `target` over time.
        
        Frame-by-frame:
          1. Compute STFT of source and target
          2. Extract smoothed spectral envelopes
          3. Create per-frame gain = lerp(1.0, target_env/source_env, progress*depth)
          4. Apply gain to source magnitude, keep source phase
          5. ISTFT to get morphed audio
        
        The morphing progress goes from 0 (pure source) at the start
        to `depth` (near target timbre) at the end.
        """
        stereo = source.ndim == 2
        if stereo:
            channels = []
            for ch in range(source.shape[1]):
                t_ch = target[:, ch] if target.ndim == 2 and ch < target.shape[1] else (
                    np.mean(target, axis=1) if target.ndim == 2 else target
                )
                channels.append(self._spectral_morph_mono(
                    source[:, ch], t_ch, depth
                ))
            return np.column_stack(channels)
        else:
            t_mono = np.mean(target, axis=1) if target.ndim == 2 else target
            return self._spectral_morph_mono(source, t_mono, depth)

    def _spectral_morph_mono(
        self, source: np.ndarray, target: np.ndarray, depth: float
    ) -> np.ndarray:
        """Core spectral envelope morph on mono signals."""
        n = min(len(source), len(target))
        source = source[:n].astype(np.float64)
        target = target[:n].astype(np.float64)

        S_src = librosa.stft(source, n_fft=self.n_fft, hop_length=self.hop_length)
        S_tgt = librosa.stft(target, n_fft=self.n_fft, hop_length=self.hop_length)

        n_frames = min(S_src.shape[1], S_tgt.shape[1])
        S_src = S_src[:, :n_frames]
        S_tgt = S_tgt[:, :n_frames]

        mag_src = np.abs(S_src)
        phase_src = np.angle(S_src)
        mag_tgt = np.abs(S_tgt)

        # Smooth envelopes (median across ~250ms windows ≈ 10 frames)
        env_window = max(3, min(11, n_frames // 10))
        if env_window % 2 == 0:
            env_window += 1

        from scipy.ndimage import median_filter
        env_src = median_filter(mag_src, size=(1, env_window))
        env_tgt = median_filter(mag_tgt, size=(1, env_window))

        # Progress curve: 0 at start → depth at end, S-shaped
        progress = np.linspace(0, 1, n_frames)
        progress = 0.5 * (1 - np.cos(np.pi * progress)) * depth

        # Per-frame: compute gain = lerp between 1.0 and target/source ratio
        eps = 1e-10
        gain_ratio = (env_tgt + eps) / (env_src + eps)
        # Clamp extreme gains to avoid distortion
        gain_ratio = np.clip(gain_ratio, 0.25, 4.0)

        # Interpolate: at progress=0 → gain=1.0, at progress=depth → gain=ratio
        gain = np.ones_like(gain_ratio)
        for frame_idx in range(n_frames):
            p = progress[frame_idx]
            gain[:, frame_idx] = 1.0 + (gain_ratio[:, frame_idx] - 1.0) * p

        mag_morphed = mag_src * gain
        S_morphed = mag_morphed * np.exp(1j * phase_src)

        y_morphed = librosa.istft(S_morphed, hop_length=self.hop_length, length=n)
        return y_morphed.astype(np.float32)

    # ====================================================================
    #  PHASE 2:  ONSET-LEVEL DRUM PATTERN MORPHING
    # ====================================================================

    def _apply_onset_replacement(
        self,
        source: np.ndarray,
        target: np.ndarray,
        depth: float,
    ) -> np.ndarray:
        """
        For drum stems: detect individual hits in both source and target,
        classify them (kick/snare/hihat/other), and progressively replace
        source hits with target hits.
        
        Replacement order (most natural):
            1. Hi-hats / cymbals (high frequency, least noticeable change)
            2. Toms / percussion
            3. Snare (mid, becomes noticeable)
            4. Kick (foundation, last to change)
        
        The `depth` parameter controls how many hits get replaced by the end:
            depth=1.0 → all hits replaced → sounds exactly like target drums
            depth=0.5 → ~half the hits replaced
        """
        stereo = source.ndim == 2
        if stereo:
            # Process L channel; apply same hit map to R
            s_mono = np.mean(source, axis=1)
            t_mono = np.mean(target, axis=1) if target.ndim == 2 else target
        else:
            s_mono = source
            t_mono = target if target.ndim == 1 else np.mean(target, axis=1)

        n = min(len(s_mono), len(t_mono))
        s_mono = s_mono[:n]
        t_mono = t_mono[:n]

        # Detect onsets + classify
        src_hits = self._detect_and_classify_hits(s_mono)
        tgt_hits = self._detect_and_classify_hits(t_mono)

        if len(src_hits) == 0:
            return source

        # Build replacement schedule based on depth and position
        result = source[:n].copy() if stereo else s_mono.copy()
        target_full = target[:n].copy() if len(target) >= n else np.pad(
            target, ((0, n - len(target)),) + ((0, 0),) * (target.ndim - 1)
        )

        # Sort hits by replacement priority
        priority_map = {'hihat': 0, 'other': 1, 'snare': 2, 'kick': 3}
        src_hits.sort(key=lambda h: (priority_map.get(h['type'], 1), h['sample']))

        # How many hits to replace
        n_replace = int(len(src_hits) * depth)

        for i, hit in enumerate(src_hits[:n_replace]):
            sample = hit['sample']
            hit_type = hit['type']

            # Progress-based replacement: earlier hits in the audio get less
            # replacement, later hits get more (gradual transformation)
            position_ratio = sample / n
            replacement_strength = min(1.0, position_ratio * 2.0 * depth)

            if replacement_strength < 0.1:
                continue

            # Find the best matching hit from target near the same position
            best_tgt = self._find_nearest_hit(tgt_hits, sample, hit_type, tolerance=0.15)

            if best_tgt is not None:
                self._splice_hit(
                    result, target_full, sample, best_tgt['sample'],
                    replacement_strength, hit_type
                )

        if stereo and result.ndim == 1:
            # If we collapsed to mono during processing, re-stereo
            # by scaling the original stereo by result/s_mono ratio
            ratio = np.zeros(n)
            nonzero = np.abs(s_mono[:n]) > 1e-10
            ratio[nonzero] = result[:n][nonzero] / s_mono[:n][nonzero]
            ratio[~nonzero] = 1.0
            ratio = gaussian_filter1d(np.clip(ratio, 0.0, 4.0), sigma=50)
            out = source[:n].copy()
            out *= ratio[:, np.newaxis] if out.ndim == 2 else ratio
            return out

        return result

    def _detect_and_classify_hits(self, y: np.ndarray) -> List[Dict]:
        """Detect percussive hits and classify as kick/snare/hihat/other."""
        onset_env = librosa.onset.onset_strength(
            y=y, sr=self.sr, hop_length=self.hop_length // 2
        )
        peaks, props = signal.find_peaks(
            onset_env,
            height=np.mean(onset_env) * 1.5,
            distance=int(self.sr / self.hop_length * 0.05),
            prominence=0.05,
        )
        if len(peaks) == 0:
            return []

        times = librosa.frames_to_time(
            peaks, sr=self.sr, hop_length=self.hop_length // 2
        )

        hits = []
        for t in times:
            sample = int(t * self.sr)
            if sample >= len(y):
                continue

            # 40ms analysis window for classification
            window = min(int(0.04 * self.sr), len(y) - sample)
            if window < 256:
                continue
            segment = y[sample:sample + window]

            fft_mag = np.abs(np.fft.rfft(segment))
            freqs = np.fft.rfftfreq(len(segment), 1 / self.sr)

            low_energy = np.sum(fft_mag[(freqs >= 20) & (freqs < 200)])
            mid_energy = np.sum(fft_mag[(freqs >= 200) & (freqs < 2000)])
            high_energy = np.sum(fft_mag[(freqs >= 2000) & (freqs < 16000)])
            total = low_energy + mid_energy + high_energy + 1e-10

            low_r = low_energy / total
            high_r = high_energy / total

            if low_r > 0.50:
                hit_type = 'kick'
            elif high_r > 0.45:
                hit_type = 'hihat'
            elif mid_energy / total > 0.40:
                hit_type = 'snare'
            else:
                hit_type = 'other'

            hits.append({
                'sample': sample,
                'type': hit_type,
                'energy': float(np.sqrt(np.mean(segment ** 2))),
                'duration_samples': window,
            })

        return hits

    def _find_nearest_hit(
        self, hits: List[Dict], sample: int, hit_type: str,
        tolerance: float = 0.15,
    ) -> Optional[Dict]:
        """Find the nearest hit of the same type within ±tolerance seconds."""
        tol_samples = int(tolerance * self.sr)
        best = None
        best_dist = float('inf')

        for h in hits:
            if h['type'] != hit_type:
                continue
            dist = abs(h['sample'] - sample)
            if dist < tol_samples and dist < best_dist:
                best = h
                best_dist = dist

        # Fallback: any hit near the position
        if best is None:
            for h in hits:
                dist = abs(h['sample'] - sample)
                if dist < tol_samples and dist < best_dist:
                    best = h
                    best_dist = dist

        return best

    def _splice_hit(
        self,
        result: np.ndarray,
        target_full: np.ndarray,
        src_sample: int,
        tgt_sample: int,
        strength: float,
        hit_type: str,
    ):
        """
        Splice a target hit into the result at the source's position.
        Uses a short crossfade window to avoid clicks.
        """
        hit_durations = {
            'kick': int(0.08 * self.sr),
            'snare': int(0.06 * self.sr),
            'hihat': int(0.03 * self.sr),
            'other': int(0.05 * self.sr),
        }
        duration = hit_durations.get(hit_type, int(0.05 * self.sr))
        cf_len = min(int(0.005 * self.sr), duration // 4)  # 5ms crossfade

        n = len(result) if result.ndim == 1 else result.shape[0]
        src_end = min(src_sample + duration, n)
        tgt_end = min(tgt_sample + duration, len(target_full) if target_full.ndim == 1 else target_full.shape[0])

        actual_dur = min(src_end - src_sample, tgt_end - tgt_sample)
        if actual_dur < cf_len * 2:
            return

        # Extract hit from target
        if target_full.ndim == 1:
            tgt_hit = target_full[tgt_sample:tgt_sample + actual_dur].copy()
        else:
            tgt_hit = target_full[tgt_sample:tgt_sample + actual_dur].copy()

        if result.ndim == 1:
            src_hit = result[src_sample:src_sample + actual_dur].copy()
        else:
            src_hit = result[src_sample:src_sample + actual_dur].copy()

        # Match energy
        src_energy = np.sqrt(np.mean(src_hit ** 2)) + 1e-10
        tgt_energy = np.sqrt(np.mean(tgt_hit ** 2)) + 1e-10
        tgt_hit *= (src_energy / tgt_energy)

        # Blend with crossfade
        blend = src_hit * (1.0 - strength) + tgt_hit * strength

        # Apply micro crossfade at edges
        if cf_len > 0:
            t = np.linspace(0, 1, cf_len)
            if blend.ndim == 1:
                blend[:cf_len] = src_hit[:cf_len] * (1 - t) + blend[:cf_len] * t
                blend[-cf_len:] = blend[-cf_len:] * (1 - t) + src_hit[-cf_len:] * t
            else:
                t2 = t[:, np.newaxis]
                blend[:cf_len] = src_hit[:cf_len] * (1 - t2) + blend[:cf_len] * t2
                blend[-cf_len:] = blend[-cf_len:] * (1 - t2) + src_hit[-cf_len:] * t2

        if result.ndim == 1:
            result[src_sample:src_sample + actual_dur] = blend
        else:
            result[src_sample:src_sample + actual_dur] = blend

    # ====================================================================
    #  PHASE 3:  CHROMA-BASED HARMONIC MORPHING
    # ====================================================================

    def _apply_chroma_modulation(
        self,
        source: np.ndarray,
        target: np.ndarray,
        depth: float,
    ) -> np.ndarray:
        """
        For pitched stems (bass, other, vocals): analyze the chroma (pitch class)
        content frame-by-frame, then apply micro pitch-shifts so the source's 
        dominant pitches gradually converge toward the target's dominant pitches.
        
        Algorithm:
            1. Compute CQT-based chroma for source and target
            2. For each analysis frame (~250ms):
               a. Find dominant pitch class in source
               b. Find dominant pitch class in target  
               c. Compute semitone shift needed (shortest path on pitch wheel)
               d. Scale shift by progress * depth
               e. Apply pitch shift to that frame's audio
        """
        stereo = source.ndim == 2
        if stereo:
            channels = []
            for ch in range(source.shape[1]):
                t_ch = target[:, ch] if target.ndim == 2 and ch < target.shape[1] else (
                    np.mean(target, axis=1) if target.ndim == 2 else target
                )
                channels.append(self._chroma_morph_mono(source[:, ch], t_ch, depth))
            return np.column_stack(channels)
        else:
            t_mono = np.mean(target, axis=1) if target.ndim == 2 else target
            return self._chroma_morph_mono(source, t_mono, depth)

    def _chroma_morph_mono(
        self, source: np.ndarray, target: np.ndarray, depth: float
    ) -> np.ndarray:
        """Core note-by-note chroma morphing for mono signals."""
        n = min(len(source), len(target))
        source = source[:n].astype(np.float64)
        target = target[:n].astype(np.float64)

        # Use larger hop for chroma analysis (~93ms per frame at 44100 sr)
        chroma_hop = self.hop_length * 2
        chroma_src = librosa.feature.chroma_cqt(
            y=source, sr=self.sr, hop_length=chroma_hop
        )
        chroma_tgt = librosa.feature.chroma_cqt(
            y=target, sr=self.sr, hop_length=chroma_hop
        )

        n_frames = min(chroma_src.shape[1], chroma_tgt.shape[1])

        # Progress curve: S-shaped morph
        progress = np.linspace(0, 1, n_frames)
        progress = 0.5 * (1 - np.cos(np.pi * progress)) * depth

        # Per-frame: compute needed pitch shift in semitones
        shift_per_frame = np.zeros(n_frames)

        for i in range(n_frames):
            src_class = int(np.argmax(chroma_src[:, i]))
            tgt_class = int(np.argmax(chroma_tgt[:, i]))

            # Don't shift if source frame has very low energy (silence)
            if np.max(chroma_src[:, i]) < 0.05:
                continue

            # Shortest path on the pitch wheel (-6 to +6 semitones)
            delta = (tgt_class - src_class) % 12
            if delta > 6:
                delta -= 12

            # Scale by progress
            shift_per_frame[i] = delta * progress[i]

        # Smooth the shift curve to avoid abrupt jumps between frames
        shift_per_frame = gaussian_filter1d(shift_per_frame, sigma=3)

        # Clamp to reasonable range to avoid artifacts
        shift_per_frame = np.clip(shift_per_frame, -4.0, 4.0)

        # Apply per-chunk pitch shifts
        result = source.copy()
        samples_per_frame = chroma_hop

        for i in range(n_frames):
            shift = shift_per_frame[i]
            if abs(shift) < 0.05:
                continue

            start = i * samples_per_frame
            end = min(start + samples_per_frame, n)
            chunk = source[start:end]

            if len(chunk) < 512:
                continue

            try:
                shifted = librosa.effects.pitch_shift(
                    chunk, sr=self.sr, n_steps=float(shift)
                )
                # Micro crossfade at boundaries (10ms)
                cf = min(int(0.01 * self.sr), len(shifted) // 4)
                if cf > 0:
                    t = np.linspace(0, 1, cf)
                    shifted[:cf] = result[start:start + cf] * (1 - t) + shifted[:cf] * t
                result[start:end] = shifted[:end - start]
            except Exception:
                pass

        return result.astype(np.float32)

    # ====================================================================
    #  PHASE 4:  PHASE VOCODER CROSS-SYNTHESIS
    # ====================================================================

    def _apply_cross_synthesis(
        self,
        source: np.ndarray,
        target: np.ndarray,
        depth: float,
    ) -> np.ndarray:
        """
        True spectral cross-synthesis using phase vocoder approach.
        
        Blends the magnitude spectrum of target with the phase spectrum of
        source. The result has the "identity" (timbre + notes) of the target
        but the "timing" and transient structure of the source.
        
        Progress-based: starts pure source, ends as cross-synthesis at `depth`.
        """
        stereo = source.ndim == 2
        if stereo:
            channels = []
            for ch in range(source.shape[1]):
                t_ch = target[:, ch] if target.ndim == 2 and ch < target.shape[1] else (
                    np.mean(target, axis=1) if target.ndim == 2 else target
                )
                channels.append(self._cross_synthesis_mono(source[:, ch], t_ch, depth))
            return np.column_stack(channels)
        else:
            t_mono = np.mean(target, axis=1) if target.ndim == 2 else target
            return self._cross_synthesis_mono(source, t_mono, depth)

    def _cross_synthesis_mono(
        self, source: np.ndarray, target: np.ndarray, depth: float
    ) -> np.ndarray:
        """Core phase vocoder cross-synthesis for mono."""
        n = min(len(source), len(target))
        source = source[:n].astype(np.float64)
        target = target[:n].astype(np.float64)

        S_src = librosa.stft(source, n_fft=self.n_fft, hop_length=self.hop_length)
        S_tgt = librosa.stft(target, n_fft=self.n_fft, hop_length=self.hop_length)

        n_frames = min(S_src.shape[1], S_tgt.shape[1])
        S_src = S_src[:, :n_frames]
        S_tgt = S_tgt[:, :n_frames]

        mag_src = np.abs(S_src)
        phase_src = np.angle(S_src)
        mag_tgt = np.abs(S_tgt)
        phase_tgt = np.angle(S_tgt)

        # Progress: S-curve, reaches `depth` at end
        progress = np.linspace(0, 1, n_frames)
        progress = 0.5 * (1 - np.cos(np.pi * progress)) * depth

        # For cross-synthesis: blend both magnitude AND phase
        # Magnitude: lerp source → target
        # Phase: weighted circular interpolation
        mag_morphed = np.zeros_like(mag_src)
        phase_morphed = np.zeros_like(phase_src)

        for i in range(n_frames):
            p = progress[i]
            mag_morphed[:, i] = mag_src[:, i] * (1 - p) + mag_tgt[:, i] * p

            # Circular phase interpolation (handles wrapping at ±π)
            phase_diff = phase_tgt[:, i] - phase_src[:, i]
            # Wrap to [-π, π]
            phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
            phase_morphed[:, i] = phase_src[:, i] + phase_diff * p

        S_morphed = mag_morphed * np.exp(1j * phase_morphed)
        y_morphed = librosa.istft(S_morphed, hop_length=self.hop_length, length=n)

        # Energy matching: keep the result at source's overall level
        src_rms = np.sqrt(np.mean(source ** 2)) + 1e-10
        res_rms = np.sqrt(np.mean(y_morphed ** 2)) + 1e-10
        y_morphed *= (src_rms / res_rms)

        return y_morphed.astype(np.float32)

    # ====================================================================
    #  BONUS:  MULTI-BAND CROSSFADE (per-frequency-band content morph)
    # ====================================================================

    def _apply_multiband_crossfade(
        self,
        source: np.ndarray,
        target: np.ndarray,
        depth: float,
    ) -> np.ndarray:
        """
        Split both stems into frequency bands, then crossfade each band
        independently with staggered timing. High frequencies morph first
        (cymbals change before kick), creating a natural progression.
        
        Bands:  sub (<100), low (100-400), mid (400-2k), high-mid (2k-6k),
                presence (6k-12k), air (12k+)
        """
        stereo = source.ndim == 2
        if stereo:
            channels = []
            for ch in range(source.shape[1]):
                t_ch = target[:, ch] if target.ndim == 2 and ch < target.shape[1] else (
                    np.mean(target, axis=1) if target.ndim == 2 else target
                )
                channels.append(self._multiband_morph_mono(source[:, ch], t_ch, depth))
            return np.column_stack(channels)
        else:
            t_mono = np.mean(target, axis=1) if target.ndim == 2 else target
            return self._multiband_morph_mono(source, t_mono, depth)

    def _multiband_morph_mono(
        self, source: np.ndarray, target: np.ndarray, depth: float
    ) -> np.ndarray:
        """Multi-band morph on mono: split, morph timing per band, recombine."""
        n = min(len(source), len(target))
        source = source[:n].astype(np.float64)
        target = target[:n].astype(np.float64)

        # Band edges (Hz) and morph start ratios (when that band starts morphing)
        # Higher bands morph earlier → hi-hats change before kick
        bands = [
            {'low': 0,     'high': 100,   'morph_start': 0.50},  # sub
            {'low': 100,   'high': 400,   'morph_start': 0.40},  # low
            {'low': 400,   'high': 2000,  'morph_start': 0.30},  # mid
            {'low': 2000,  'high': 6000,  'morph_start': 0.20},  # high-mid
            {'low': 6000,  'high': 12000, 'morph_start': 0.10},  # presence
            {'low': 12000, 'high': 20000, 'morph_start': 0.05},  # air
        ]

        result = np.zeros(n)
        nyq = self.sr / 2

        for band in bands:
            low_hz = band['low']
            high_hz = min(band['high'], nyq - 1)
            morph_start = band['morph_start']

            # Bandpass filter on both
            src_band = self._bandpass(source, low_hz, high_hz)
            tgt_band = self._bandpass(target, low_hz, high_hz)

            # Create time-varying crossfade curve for this band
            t = np.linspace(0, 1, n)
            # Morph starts at morph_start and reaches depth by end
            morph_progress = np.clip(
                (t - morph_start) / max(0.01, 1.0 - morph_start),
                0.0, 1.0
            ) * depth
            # S-curve for smoothness
            morph_progress = 0.5 * (1 - np.cos(np.pi * morph_progress))

            # Blend
            blended = src_band * (1.0 - morph_progress) + tgt_band * morph_progress
            result += blended

        return result.astype(np.float32)

    def _bandpass(self, y: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
        """Apply bandpass filter. Handles edge cases for low/highpass."""
        nyq = self.sr / 2
        low_n = max(low_hz / nyq, 0.001)
        high_n = min(high_hz / nyq, 0.999)

        if high_n <= low_n:
            return np.zeros_like(y)

        try:
            if low_n < 0.005:
                # Lowpass only
                sos = signal.butter(4, high_n, btype='low', output='sos')
            elif high_n > 0.995:
                # Highpass only
                sos = signal.butter(4, low_n, btype='high', output='sos')
            else:
                sos = signal.butter(4, [low_n, high_n], btype='band', output='sos')
            return signal.sosfiltfilt(sos, y)
        except Exception:
            return np.zeros_like(y)

    # ====================================================================
    #  ANALYSIS HELPERS
    # ====================================================================

    def _spectral_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between average spectral envelopes."""
        S_a = np.abs(librosa.stft(a, n_fft=self.n_fft, hop_length=self.hop_length))
        S_b = np.abs(librosa.stft(b, n_fft=self.n_fft, hop_length=self.hop_length))

        env_a = np.mean(S_a, axis=1)
        env_b = np.mean(S_b, axis=1)

        dot = np.dot(env_a, env_b)
        norm = np.linalg.norm(env_a) * np.linalg.norm(env_b) + 1e-10
        return float(np.clip(dot / norm, 0, 1))

    def _rhythmic_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compare onset density and pattern correlation."""
        hop = self.hop_length // 2
        onset_a = librosa.onset.onset_strength(y=a, sr=self.sr, hop_length=hop)
        onset_b = librosa.onset.onset_strength(y=b, sr=self.sr, hop_length=hop)

        # Align lengths
        min_len = min(len(onset_a), len(onset_b))
        if min_len < 4:
            return 0.5

        onset_a = onset_a[:min_len]
        onset_b = onset_b[:min_len]

        # Normalize
        onset_a = onset_a / (np.max(onset_a) + 1e-10)
        onset_b = onset_b / (np.max(onset_b) + 1e-10)

        # Correlation
        corr = np.corrcoef(onset_a, onset_b)[0, 1]
        corr = max(0, corr)  # only positive

        # Density similarity
        density_a = np.mean(onset_a > 0.3)
        density_b = np.mean(onset_b > 0.3)
        density_sim = 1.0 - abs(density_a - density_b)

        return float(0.6 * corr + 0.4 * density_sim)

    def _chroma_correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        """Correlation between average chroma profiles (harmonic similarity)."""
        try:
            chroma_a = librosa.feature.chroma_cqt(y=a, sr=self.sr)
            chroma_b = librosa.feature.chroma_cqt(y=b, sr=self.sr)
        except Exception:
            return 0.5

        avg_a = np.mean(chroma_a, axis=1)
        avg_b = np.mean(chroma_b, axis=1)

        # Normalize
        avg_a = avg_a / (np.max(avg_a) + 1e-10)
        avg_b = avg_b / (np.max(avg_b) + 1e-10)

        # Correlation
        corr = np.corrcoef(avg_a, avg_b)[0, 1]
        return float(max(0, corr))
