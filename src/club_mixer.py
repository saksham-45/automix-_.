"""
Club / Boiler-Room–style mixer.

Implements the research-grounded long-blend model from BOILER_ROOM_MIXING_PLAN.md:

  * A transition is three timestamps t1 / t2 / t3 (B audible → switch → A inaudible),
    a four-state crossfade (Only A → A-prevalent → B-prevalent → Only B).
  * The switch point t2 is placed on a DOWNBEAT at the start of a phrase (8-bar grid).
  * Tempo is matched by time-stretching B to A's tempo (downbeat-locked).
  * Equal-power volume crossfade for mids/highs; a beat-locked 3-band EQ BASS SWAP
    keeps low-end energy constant while handing the kick from A to B on the t2 downbeat.
  * Loudness is matched (K-weighted LUFS) and the output is limited to -1 dBTP.

Self-contained: reuses CrossfadeEngine / PsychoacousticAnalyzer only where useful.
References: Zehren et al. 2022; Vande Veire & De Bie 2018; Foote 2000 / Müller FMP.
"""
from __future__ import annotations

import numpy as np
import librosa
import scipy.signal as sig
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ----------------------------------------------------------------------------- #
# Phrase grid
# ----------------------------------------------------------------------------- #
@dataclass
class PhraseGrid:
    sr: int
    tempo: float                 # BPM (scalar)
    bar_dur: float               # seconds per 4/4 bar
    downbeats: np.ndarray        # downbeat times (seconds), bar-1 of each bar
    bar_energy: np.ndarray       # RMS per bar (len == len(downbeats)-1, padded)
    duration: float              # seconds
    sections: Optional[list] = None  # [{start_sec,end_sec,label:'low'|'high',energy}]


def _to_mono(y: np.ndarray) -> np.ndarray:
    return y if y.ndim == 1 else np.mean(y, axis=1)


def _estimate_downbeat_offset(onset_at_beats: np.ndarray) -> int:
    """Pick the beat phase (0..3) whose beats carry the most onset strength —
    the downbeat usually has the strongest accent. Robust, simple heuristic
    (far better than the codebase's naive 'every 4th beat from index 0')."""
    best_off, best_val = 0, -1.0
    for off in range(4):
        vals = onset_at_beats[off::4]
        if len(vals) == 0:
            continue
        m = float(np.mean(vals))
        if m > best_val:
            best_val, best_off = m, off
    return best_off


def _detect_sections(mono: np.ndarray, sr: int, beats: np.ndarray,
                     downbeats: np.ndarray, bar_dur: float,
                     bar_energy: np.ndarray, hop_length: int = 512) -> list:
    """Structural segmentation via a beat-synchronous self-similarity matrix +
    Foote Gaussian checkerboard novelty (Foote 2000 / Müller FMP). Boundaries are
    snapped to downbeats (discarded if >0.4 bar away) and labelled low/high energy
    so the planner can mix OUT on a low section (outro/breakdown) and IN on a low
    section (intro/breakdown). Falls back to one 'high' section on short input."""
    full = [{"start_sec": 0.0, "end_sec": len(mono) / sr, "label": "high", "energy": 1.0}]
    if len(beats) < 16 or len(downbeats) < 2:
        return full
    try:
        chroma = librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=hop_length)
        mfcc = librosa.feature.mfcc(y=mono, sr=sr, hop_length=hop_length, n_mfcc=13)
        feat = np.vstack([chroma, mfcc])
        bf = np.clip(librosa.time_to_frames(beats, sr=sr, hop_length=hop_length), 0, feat.shape[1] - 1)
        X = librosa.util.sync(feat, bf, aggregate=np.mean).T            # [n_beats, F]
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        S = X @ X.T                                                      # cosine SSM
        n = S.shape[0]
        L = min(16, n // 2)                                             # half-kernel (~4 bars)
        if L < 2:
            return full
        taper = np.outer(np.hanning(2 * L), np.hanning(2 * L))
        checker = np.ones((2 * L, 2 * L)); checker[:L, L:] = -1; checker[L:, :L] = -1
        kernel = taper * checker
        nov = np.zeros(n)
        for i in range(L, n - L):
            nov[i] = float(np.sum(S[i - L:i + L, i - L:i + L] * kernel))
        nov = np.maximum(nov, 0.0)
        if nov.max() > 0:
            nov /= nov.max()
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(nov, height=0.3, distance=16)            # >=4 bars apart
        # snap peak beat-times to downbeats within 0.4 bar
        bounds = {0.0}
        for t in beats[peaks]:
            d = float(downbeats[int(np.argmin(np.abs(downbeats - t)))])
            if abs(d - t) <= 0.4 * bar_dur:
                bounds.add(d)
        bnds = sorted(bounds) + [len(mono) / sr]
        med = float(np.median(bar_energy)) if len(bar_energy) else 0.0
        secs = []
        for s, e in zip(bnds, bnds[1:]):
            if e - s < bar_dur:
                continue
            idx = [i for i, db in enumerate(downbeats) if s <= db < e]
            en = float(np.mean([bar_energy[i] for i in idx])) if idx else 0.0
            secs.append({"start_sec": float(s), "end_sec": float(e),
                         "label": "high" if en >= med else "low", "energy": en})
        return secs or full
    except Exception:
        return full


def build_phrase_grid(y: np.ndarray, sr: int, hop_length: int = 512,
                      phrase_bars: int = 8) -> PhraseGrid:
    """Beats → downbeat phase → bars → (phrase-aligned) downbeat grid + per-bar energy."""
    mono = _to_mono(y).astype(np.float32)
    tempo, beats = librosa.beat.beat_track(y=mono, sr=sr, hop_length=hop_length, units='time')
    tempo = float(np.atleast_1d(tempo)[0]) or 120.0
    bar_dur = 4.0 * 60.0 / tempo
    duration = len(mono) / sr

    if len(beats) < 4:
        # Degenerate: synthesize a grid from tempo alone.
        n_db = max(1, int(duration / bar_dur))
        downbeats = np.arange(n_db) * bar_dur
        bar_energy = np.array([float(np.sqrt(np.mean(mono ** 2)) + 1e-9)] * n_db)
        return PhraseGrid(sr, tempo, bar_dur, downbeats, bar_energy, duration)

    # Downbeat phase via onset strength sampled at beat positions.
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop_length)
    bf = np.clip(librosa.time_to_frames(beats, sr=sr, hop_length=hop_length), 0, len(onset_env) - 1)
    offset = _estimate_downbeat_offset(onset_env[bf])

    downbeats = np.asarray(beats[offset::4], dtype=float)
    if len(downbeats) == 0:
        downbeats = np.asarray(beats[:1], dtype=float)

    # Per-bar RMS energy (for region/section selection).
    bar_energy = []
    for i in range(len(downbeats)):
        s = int(downbeats[i] * sr)
        e = int(downbeats[i + 1] * sr) if i + 1 < len(downbeats) else min(len(mono), s + int(bar_dur * sr))
        seg = mono[s:e]
        bar_energy.append(float(np.sqrt(np.mean(seg ** 2)) + 1e-9) if len(seg) else 1e-9)
    bar_energy = np.asarray(bar_energy)

    sections = _detect_sections(mono, sr, np.asarray(beats, dtype=float),
                                downbeats, bar_dur, bar_energy, hop_length)
    return PhraseGrid(sr, tempo, bar_dur, downbeats, bar_energy, duration, sections)


# ----------------------------------------------------------------------------- #
# Transition planning (t1 / t2 / t3 on the phrase grid)
# ----------------------------------------------------------------------------- #
@dataclass
class TransitionPlan:
    blend_bars: int
    bar_dur: float               # seconds (mix tempo, = A's bar)
    a_switch_sec: float          # t2 in A coordinates (a downbeat)
    a_t1_sec: float              # overlap start in A
    a_t3_sec: float              # overlap end in A
    b_in_sec: float              # overlap start in B (a downbeat, mix-in)
    transition_type: str         # relaxed | rolling | double_drop
    overlap_samples: int


def _nearest_downbeat(downbeats: np.ndarray, t: float) -> float:
    if len(downbeats) == 0:
        return t
    return float(downbeats[int(np.argmin(np.abs(downbeats - t)))])


def _energy_label(grid: PhraseGrid, start_sec: float, bars: int) -> str:
    """'high' (drop/core) vs 'low' (intro/breakdown/outro) for a region, by
    comparing local bar energy to the track median."""
    if len(grid.bar_energy) == 0:
        return 'low'
    i0 = int(np.argmin(np.abs(grid.downbeats - start_sec)))
    window = grid.bar_energy[i0:i0 + max(1, bars)]
    if len(window) == 0:
        window = grid.bar_energy[i0:i0 + 1]
    med = float(np.median(grid.bar_energy))
    return 'high' if float(np.mean(window)) >= med else 'low'


def plan_transition(grid_a: PhraseGrid, grid_b: PhraseGrid,
                    blend_bars: int = 16,
                    recent_types: Optional[list] = None) -> TransitionPlan:
    """Place t1/t2/t3 on phrase downbeats and pick a transition type.

    Mix-OUT region in A: a late downbeat (outro) leaving room for the blend.
    Mix-IN region in B: an early downbeat past any dead-air intro (beat present).
    """
    bar_dur = grid_a.bar_dur
    half = blend_bars / 2.0
    half_sec = half * bar_dur

    # --- A switch (t2): prefer the start of A's LAST low-energy section
    #     (outro/breakdown) with room; else the latest valid downbeat. ---
    a_valid = [db for db in grid_a.downbeats
               if db - half_sec >= 0 and db + half_sec <= grid_a.duration]
    a_switch = None
    if grid_a.sections:
        low_starts = [s["start_sec"] for s in grid_a.sections
                      if s.get("label") == "low"
                      and s["start_sec"] - half_sec >= 0
                      and s["start_sec"] + half_sec <= grid_a.duration]
        if low_starts:
            # snap to the nearest actual downbeat (section starts already are)
            a_switch = _nearest_downbeat(grid_a.downbeats, float(low_starts[-1]))
    if a_switch is None:
        a_switch = float(a_valid[-1]) if a_valid else _nearest_downbeat(
            grid_a.downbeats, max(0.0, grid_a.duration - half_sec))
    a_t1 = max(0.0, a_switch - half_sec)
    a_t3 = min(grid_a.duration, a_switch + half_sec)

    # --- B mix-in: prefer the start of B's FIRST low-energy section
    #     (intro/breakdown) with room; else first downbeat with real energy. ---
    b_in = None
    if grid_b.sections:
        for s in grid_b.sections:
            if s.get("label") == "low" and s["start_sec"] + blend_bars * bar_dur <= grid_b.duration:
                b_in = _nearest_downbeat(grid_b.downbeats, float(s["start_sec"]))
                break
    if b_in is None:
        b_in = float(grid_b.downbeats[0]) if len(grid_b.downbeats) else 0.0
        if len(grid_b.bar_energy):
            thr = 0.15 * float(np.median(grid_b.bar_energy) + 1e-9)
            for i, db in enumerate(grid_b.downbeats):
                if grid_b.bar_energy[min(i, len(grid_b.bar_energy) - 1)] > thr:
                    b_in = float(db)
                    break
    # Ensure B has room for the whole blend after b_in.
    if b_in + blend_bars * bar_dur > grid_b.duration:
        b_in = max(0.0, grid_b.duration - blend_bars * bar_dur)
        b_in = _nearest_downbeat(grid_b.downbeats, b_in)

    # --- Transition type from overlapped-segment energy (FSM anti-repeat). ---
    lbl_a = _energy_label(grid_a, a_t1, blend_bars)
    lbl_b = _energy_label(grid_b, b_in, blend_bars)
    if lbl_a == 'high' and lbl_b == 'high':
        ttype = 'double_drop'
    elif lbl_a == 'low' and lbl_b == 'low':
        ttype = 'relaxed'
    else:
        ttype = 'rolling'
    recent = recent_types or []
    if recent and recent[-1] == ttype:
        # avoid repeating the same type back-to-back
        ttype = {'double_drop': 'rolling', 'relaxed': 'rolling', 'rolling': 'relaxed'}[ttype]

    overlap_samples = int(round(blend_bars * bar_dur * grid_a.sr))
    return TransitionPlan(blend_bars, bar_dur, a_switch, a_t1, a_t3, b_in,
                          ttype, overlap_samples)


# ----------------------------------------------------------------------------- #
# Rendering
# ----------------------------------------------------------------------------- #
def _as_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.column_stack([y, y])
    if y.shape[1] == 1:
        return np.column_stack([y[:, 0], y[:, 0]])
    return y[:, :2]


def _apply_gain(seg: np.ndarray, g: np.ndarray) -> np.ndarray:
    return seg * (g[:, None] if seg.ndim == 2 else g)


def _band_split(seg: np.ndarray, sr: int, lo: float = 250.0, hi: float = 2500.0):
    """3-band split with perfect reconstruction (low + mid + high == seg)."""
    nyq = sr / 2.0
    sos_lo = sig.butter(4, min(lo / nyq, 0.99), btype='low', output='sos')
    sos_hi = sig.butter(4, min(hi / nyq, 0.99), btype='high', output='sos')
    low = sig.sosfiltfilt(sos_lo, seg, axis=0)
    high = sig.sosfiltfilt(sos_hi, seg, axis=0)
    mid = seg - low - high
    return low, mid, high


def _equal_power(n: int) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)
    return np.sqrt(0.5 * (1 + np.cos(np.pi * t))), np.sqrt(0.5 * (1 - np.cos(np.pi * t)))


def _fit_len(seg: np.ndarray, n: int) -> np.ndarray:
    """Truncate or zero-pad a stereo segment to exactly n samples."""
    if len(seg) >= n:
        return seg[:n]
    pad = np.zeros((n - len(seg), seg.shape[1]), dtype=seg.dtype)
    return np.vstack([seg, pad])


def _render_overlap(seg_a: np.ndarray, seg_b: np.ndarray, plan: "TransitionPlan", sr: int) -> np.ndarray:
    """The 4-state blend over one overlap window: equal-power mids/highs crossfade
    + beat-locked 3-band EQ bass swap on the t2 downbeat (constant low-end power).
    seg_a/seg_b must already be the same length (= plan.overlap_samples)."""
    N = len(seg_a)
    bar_samp = int(round(plan.bar_dur * sr))
    volA, volB = _equal_power(N)
    low_a, mid_a, high_a = _band_split(seg_a, sr)
    low_b, mid_b, high_b = _band_split(seg_b, sr)
    s2 = int(round((plan.blend_bars / 2.0) * bar_samp))   # switch (t2)
    sw = max(1, bar_samp)                                  # swap over ~1 bar, centered on t2
    x = np.clip((np.arange(N) - (s2 - sw / 2)) / sw, 0.0, 1.0)
    gA_low = np.sqrt(0.5 * (1 + np.cos(np.pi * x)))        # 1 -> 0
    gB_low = np.sqrt(0.5 * (1 - np.cos(np.pi * x)))        # 0 -> 1 (equal power)
    lows = _apply_gain(low_a, gA_low) + _apply_gain(low_b, gB_low)
    mids_highs = _apply_gain(mid_a + high_a, volA) + _apply_gain(mid_b + high_b, volB)
    return lows + mids_highs


# ----------------------------------------------------------------------------- #
# Stem-aware overlap (uses demucs) — surgically clean handoffs
# ----------------------------------------------------------------------------- #
_SEP = None


def _get_separator():
    """Lazy, shared StemSeparator (cached separation, MPS/CUDA-auto)."""
    global _SEP
    if _SEP is None:
        from src.stem_separator import StemSeparator
        _SEP = StemSeparator()
    return _SEP


def _swap_gates(N: int, s2: int, sw: int) -> Tuple[np.ndarray, np.ndarray]:
    """Equal-power swap gates centred on the switch sample s2 over ~sw samples."""
    x = np.clip((np.arange(N) - (s2 - sw / 2)) / sw, 0.0, 1.0)
    return np.sqrt(0.5 * (1 + np.cos(np.pi * x))), np.sqrt(0.5 * (1 - np.cos(np.pi * x)))


def _render_overlap_stems(seg_a: np.ndarray, seg_b: np.ndarray,
                          plan: "TransitionPlan", sr: int) -> np.ndarray:
    """Stem-aware blend (demucs). Cleaner than the EQ filter because each element
    is handled on its own separated stem — no filter bleed, no two-kick mud, and
    explicit vocal-clash avoidance:

      drums  : beat-locked swap on the t2 downbeat (A kit out, B kit in) — one kit at a time
      bass   : same swap (constant low-end power, no sub stacking)
      other  : equal-power crossfade across the whole overlap (melodic glue)
      vocals : A vocal gone by the switch, B vocal only after — never two leads at full

    Falls back to the EQ overlap if separation is unavailable.
    """
    try:
        sep = _get_separator()
        sa = sep.separate_stems(seg_a, sr)
        sb = sep.separate_stems(seg_b, sr)
    except Exception as e:
        print(f"  ⚠ stem blend unavailable ({e}); using EQ overlap")
        return _render_overlap(seg_a, seg_b, plan, sr)

    N = len(seg_a)
    bar_samp = int(round(plan.bar_dur * sr))
    s2 = int(round((plan.blend_bars / 2.0) * bar_samp))
    sw = max(1, bar_samp)
    gA, gB = _swap_gates(N, s2, sw)          # 1->0 , 0->1 across the switch
    volA, volB = _equal_power(N)

    def st(d, k):
        return _fit_len(_as_stereo(d.get(k, np.zeros((N, 2), dtype=np.float32))), N)

    drums = _apply_gain(st(sa, 'drums'), gA) + _apply_gain(st(sb, 'drums'), gB)
    bass = _apply_gain(st(sa, 'bass'), gA) + _apply_gain(st(sb, 'bass'), gB)
    other = _apply_gain(st(sa, 'other'), volA) + _apply_gain(st(sb, 'other'), volB)
    vocals = _apply_gain(st(sa, 'vocals'), gA) + _apply_gain(st(sb, 'vocals'), gB)
    return drums + bass + other + vocals


def _time_stretch(y_mono: np.ndarray, rate: float) -> np.ndarray:
    if abs(1.0 - rate) < 1e-3:
        return y_mono
    return librosa.effects.time_stretch(y_mono, rate=rate)


def _choose_tempo_rate(tempo_a: float, tempo_b: float, max_pct: float = 0.12) -> float:
    """Rate to stretch B so it plays at A's tempo. librosa rate>1 SPEEDS UP, so
    rate = tempo_a/tempo_b. Try half/double-time when the direct match is too far,
    and reject (return 1.0) if even the best option exceeds max_pct."""
    candidates = [tempo_a / tempo_b, tempo_a / (2.0 * tempo_b), (2.0 * tempo_a) / tempo_b]
    candidates = [c for c in candidates if c > 0]
    best = min(candidates, key=lambda c: abs(np.log2(c)))  # closest to no-stretch
    if abs(1.0 - best) > max_pct:
        return 1.0  # too far — leave B unstretched (caller still blends, just not BPM-locked)
    return float(best)


def _match_loudness(seg_a: np.ndarray, seg_b: np.ndarray, sr: int) -> np.ndarray:
    """Scale B to A's integrated loudness (K-weighted LUFS), clamped to +/-6 dB."""
    try:
        from src.psychoacoustics import PsychoacousticAnalyzer
        pa = PsychoacousticAnalyzer(sr=sr)
        la = pa.analyze_loudness_lufs(_to_mono(seg_a))['integrated_lufs']
        lb = pa.analyze_loudness_lufs(_to_mono(seg_b))['integrated_lufs']
        gain_db = max(-6.0, min(6.0, la - lb))
        if abs(gain_db) > 0.5:
            return seg_b * (10.0 ** (gain_db / 20.0))
    except Exception:
        pass
    return seg_b


def _headroom(y: np.ndarray, ceiling: float = 0.89125) -> np.ndarray:  # -1 dBTP
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return y * (ceiling / m) if m > ceiling else y


def render_club_mix(y_a: np.ndarray,
                    y_b: np.ndarray,
                    sr: int = 44100,
                    blend_bars: int = 16,
                    context_bars: int = 8,
                    phrase_bars: int = 8,
                    recent_types: Optional[list] = None,
                    use_stems: bool = False) -> Tuple[np.ndarray, Dict]:
    """Render a Boiler-Room–style long blend of A→B. Returns (stereo_audio, plan_info)."""
    grid_a = build_phrase_grid(y_a, sr, phrase_bars=phrase_bars)
    grid_b = build_phrase_grid(y_b, sr, phrase_bars=phrase_bars)

    # --- Tempo lock: stretch B to A's tempo (downbeat handoff stays on grid). ---
    rate = _choose_tempo_rate(grid_a.tempo, grid_b.tempo)
    yb_st = _as_stereo(y_b).astype(np.float32)
    if abs(1.0 - rate) > 1e-3:
        chans = [_time_stretch(yb_st[:, c], rate) for c in range(yb_st.shape[1])]
        m = min(len(c) for c in chans)
        yb_st = np.column_stack([c[:m] for c in chans])
        # rebuild B grid on the stretched audio so downbeats are correct
        grid_b = build_phrase_grid(yb_st, sr, phrase_bars=phrase_bars)
    ya_st = _as_stereo(y_a).astype(np.float32)

    plan = plan_transition(grid_a, grid_b, blend_bars=blend_bars, recent_types=recent_types)

    N = plan.overlap_samples
    bar_samp = int(round(plan.bar_dur * sr))

    # --- Overlap segments. ---
    a_t1 = int(round(plan.a_t1_sec * sr))
    seg_a = ya_st[a_t1:a_t1 + N]
    b_in = int(round(plan.b_in_sec * sr))
    seg_b = yb_st[b_in:b_in + N]
    seg_a, seg_b = _fit_len(seg_a, N), _fit_len(seg_b, N)

    # Loudness-match B to A across the overlap, then render the 4-state blend.
    seg_b = _match_loudness(seg_a, seg_b, sr)
    overlap = (_render_overlap_stems if use_stems else _render_overlap)(seg_a, seg_b, plan, sr)

    # --- Context (lead-in from A, lead-out from B). ---
    ctx = int(round(context_bars * plan.bar_dur * sr))
    lead_in = ya_st[max(0, a_t1 - ctx):a_t1]
    b_out = b_in + N
    lead_out = yb_st[b_out:min(len(yb_st), b_out + ctx)]

    # --- Click-free assembly with 50 ms equal-power splices. ---
    final = _splice([lead_in, overlap, lead_out], sr)
    final = _headroom(final).astype(np.float32)

    info = {
        'tempo_a': grid_a.tempo, 'tempo_b': grid_b.tempo, 'stretch_rate_b': rate,
        'bar_dur_sec': plan.bar_dur, 'blend_bars': plan.blend_bars,
        'transition_type': plan.transition_type,
        't1_sec_in_A': plan.a_t1_sec, 't2_switch_sec_in_A': plan.a_switch_sec,
        't3_sec_in_A': plan.a_t3_sec, 'b_mix_in_sec': plan.b_in_sec,
        'switch_on_downbeat': bool(np.min(np.abs(grid_a.downbeats - plan.a_switch_sec)) < 1e-6)
                              if len(grid_a.downbeats) else False,
        'output_sec': len(final) / sr,
    }
    return final, info


def _splice(parts, sr: int, xf_sec: float = 0.05) -> np.ndarray:
    """Concatenate stereo parts with short equal-power crossfades at the joins."""
    parts = [p for p in parts if p is not None and len(p) > 0]
    if not parts:
        return np.zeros((0, 2), dtype=np.float32)
    out = parts[0]
    xf = int(xf_sec * sr)
    for nxt in parts[1:]:
        k = min(xf, len(out) // 2, len(nxt) // 2)
        if k > 8:
            fo, fi = _equal_power(k)
            joined = out[-k:] * fo[:, None] + nxt[:k] * fi[:, None]
            out = np.vstack([out[:-k], joined, nxt[k:]])
        else:
            out = np.vstack([out, nxt])
    return out


def build_continuous_set(tracks,
                         sr: int = 44100,
                         blend_bars: int = 16,
                         phrase_bars: int = 8,
                         progress=None,
                         use_stems: bool = False) -> Tuple[np.ndarray, list]:
    """Chain N tracks into ONE continuous, gapless, club-mixed timeline (a DJ set):

        [track0 body → t1] [0→1 blend] [track1 body → t1] [1→2 blend] ... [last body]

    Each consecutive pair is beat-/phrase-locked and bass-swapped via the same
    research-grounded model as render_club_mix. `tracks` is a list of audio arrays
    (mono or stereo). Returns (stereo_audio, markers) where markers describe each
    track-body and transition span (sample offsets) for a gapless client / UI.

    This is the engine behind "paste a playlist → one continuous mixed stream":
    a transition never adds a gap because the whole set is a single timeline.
    """
    n = len(tracks)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32), []
    cur = _as_stereo(np.asarray(tracks[0])).astype(np.float32)
    if n == 1:
        out = _headroom(cur).astype(np.float32)
        return out, [{"type": "track", "index": 0, "start_sec": 0.0, "end_sec": len(out) / sr}]

    timeline = []          # stereo parts to splice
    markers = []
    resume = 0.0           # seconds into `cur` where its audible body should begin
    pos = 0.0              # running output position (seconds), approximate (pre-splice)

    for i in range(n - 1):
        nxt = _as_stereo(np.asarray(tracks[i + 1])).astype(np.float32)
        grid_a = build_phrase_grid(cur, sr, phrase_bars=phrase_bars)
        grid_b0 = build_phrase_grid(nxt, sr, phrase_bars=phrase_bars)

        # Tempo-lock B to A (downbeat handoff stays on grid).
        rate = _choose_tempo_rate(grid_a.tempo, grid_b0.tempo)
        if abs(1.0 - rate) > 1e-3:
            chans = [_time_stretch(nxt[:, c], rate) for c in range(nxt.shape[1])]
            m = min(len(c) for c in chans)
            nxt = np.column_stack([c[:m] for c in chans]).astype(np.float32)
            grid_b = build_phrase_grid(nxt, sr, phrase_bars=phrase_bars)
        else:
            grid_b = grid_b0

        plan = plan_transition(grid_a, grid_b, blend_bars=blend_bars)
        N = plan.overlap_samples

        # Body of current track: from its resume point up to t1.
        r0 = int(round(resume * sr))
        a_t1 = int(round(plan.a_t1_sec * sr))
        body = cur[max(0, r0):max(r0, a_t1)]
        timeline.append(body)
        markers.append({"type": "track", "index": i,
                        "start_sec": pos, "end_sec": pos + len(body) / sr})
        pos += len(body) / sr

        # Overlap blend (loudness-matched, bass-swapped, phrase-locked).
        seg_a = _fit_len(cur[a_t1:a_t1 + N], N)
        b_in = int(round(plan.b_in_sec * sr))
        seg_b = _fit_len(nxt[b_in:b_in + N], N)
        seg_b = _match_loudness(seg_a, seg_b, sr)
        overlap = (_render_overlap_stems if use_stems else _render_overlap)(seg_a, seg_b, plan, sr)
        timeline.append(overlap)
        markers.append({"type": "transition", "from": i, "to": i + 1,
                        "transition_type": plan.transition_type,
                        "start_sec": pos, "end_sec": pos + len(overlap) / sr})
        pos += len(overlap) / sr

        # Next track becomes current; it resumes just after the overlap.
        cur = nxt
        resume = plan.b_in_sec + plan.blend_bars * plan.bar_dur
        if progress:
            progress(i + 1, n)

    # Tail: last track from its resume point to the end.
    tail = cur[int(round(resume * sr)):]
    timeline.append(tail)
    markers.append({"type": "track", "index": n - 1,
                    "start_sec": pos, "end_sec": pos + len(tail) / sr})

    final = _headroom(_splice(timeline, sr)).astype(np.float32)
    return final, markers
