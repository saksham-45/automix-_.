"""
Royalty-free demo loops, synthesized on demand.

The public demo must not ship copyrighted music (that would defeat both the
security/legal posture and the research framing). Instead we generate three short,
four-on-the-floor electronic loops at distinct tempos so the beat/downbeat tracker
has clean onsets to lock onto and the tempo-matched transition is actually audible.

`ensure_demo_samples()` is idempotent: it writes the files only if missing, so it is
cheap to call at container build time and again at server startup as a fallback.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

# (filename, bpm, root note Hz) — different tempos make beat-matching visible.
_SPECS = [
    ("sample_a.wav", 124.0, 110.00),  # A2
    ("sample_b.wav", 126.0, 130.81),  # C3
    ("sample_c.wav", 128.0, 146.83),  # D3
]
_SECONDS = 48.0


def _synth_loop(bpm: float, root: float, seconds: float, sr: int) -> np.ndarray:
    rng = np.random.default_rng(int(bpm))          # deterministic per track
    n = int(seconds * sr)
    beat = 60.0 / bpm
    out = np.zeros(n, dtype=np.float32)

    # Four-on-the-floor kick: pitch-dropping sine with a fast amplitude decay.
    for k in range(int(seconds / beat)):
        i0 = int(k * beat * sr)
        env_n = int(0.18 * sr)
        if i0 + env_n > n:
            break
        env = np.exp(-np.linspace(0, 7, env_n))
        pitch = np.linspace(120, 45, env_n)
        out[i0:i0 + env_n] += 0.9 * env * np.sin(2 * np.pi * np.cumsum(pitch) / sr)

    # Offbeat closed hat: short filtered noise burst between kicks.
    for k in range(int(seconds / beat)):
        i0 = int((k + 0.5) * beat * sr)
        env_n = int(0.04 * sr)
        if i0 + env_n > n:
            break
        out[i0:i0 + env_n] += 0.18 * np.exp(-np.linspace(0, 9, env_n)) * rng.standard_normal(env_n)

    # Sub bass following a simple 8th-note root/fifth pattern.
    pat = [1.0, 1.0, 1.5, 1.0, 1.0, 1.5, 1.0, 0.75]
    eighth = beat / 2
    for k in range(int(seconds / eighth)):
        i0 = int(k * eighth * sr)
        seg_n = int(eighth * sr)
        if i0 + seg_n > n:
            break
        f = root * pat[k % len(pat)]
        env = np.minimum(1.0, np.exp(-np.linspace(0, 2.5, seg_n)) * 1.3)
        out[i0:i0 + seg_n] += 0.35 * env * np.sin(2 * np.pi * f * (np.arange(seg_n) / sr))

    # Bar-aligned chord stab (root + major third + fifth) for harmonic body.
    bar = beat * 4
    for k in range(int(seconds / bar)):
        i0 = int(k * bar * sr)
        env_n = int(0.6 * sr)
        if i0 + env_n > n:
            break
        env = np.exp(-np.linspace(0, 4, env_n))
        chord = sum(np.sin(2 * np.pi * root * 2 * r * (np.arange(env_n) / sr))
                    for r in (1.0, 1.26, 1.5))
        out[i0:i0 + env_n] += 0.12 * env * chord

    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.89).astype(np.float32)


def ensure_demo_samples(out_dir, sr: int = 44100) -> list:
    """Write the demo loops into out_dir if absent. Returns the list of paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, bpm, root in _SPECS:
        p = out_dir / name
        if not p.exists():
            sf.write(str(p), _synth_loop(bpm, root, _SECONDS, sr), sr, subtype="PCM_16")
        paths.append(str(p))
    return paths


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "data/demo"
    for p in ensure_demo_samples(d):
        print(p)
