# Production Plan — YouTube Playlist → Gapless Auto-Mixed Stream

> Goal: user pastes a **YouTube playlist URL**, gets a **continuous, gapless, beat-matched club-style stream** (no audible delay between tracks), like Apple Music AutoMix. Demucs stems are used for quality and must be **fast without quality loss**.

---

## 0. The honest latency truth (so we engineer the right thing)
- **First track has unavoidable startup latency**: we must at least download + analyze track 1 before audio can start (seconds). Apple has this too (they pre-analyze the catalog; we can't for arbitrary YouTube).
- **Steady-state can be truly gapless** if a **producer pipeline stays ahead of playback**: while track N plays, we already downloaded/analyzed/separated/pre-rendered the N→N+1 transition. The user never waits *between* songs. This look-ahead buffer is the core of "no delay."
- So "no delay at all" = **minimize first-track startup** (start playback the instant track-1 body is ready, render the transition during playback) + **never starve the buffer** afterwards.

---

## 1. Architecture

```
 Playlist URL
     │  yt-dlp --flat-playlist  → [video_id...]
     ▼
┌───────────────────────── PRODUCER (stays ≥1 transition ahead) ─────────────────────────┐
│  Prefetch(N+1): yt-dlp download audio  ──►  Analyze (beat/downbeat/key/structure, CACHED)│
│        │                                          │                                       │
│        ▼                                          ▼                                       │
│  Separate ONLY the mix regions (out-of-A, in-of-B) via demucs on MPS  ◄── stem CACHE     │
│        │                                                                                  │
│        ▼                                                                                  │
│  club_mixer.render_club_mix(A_tail, B_head)  → transition chunk (gapless-spliced)         │
│        │                                                                                  │
│        ▼                                                                                  │
│  Emit ordered chunks:  [A body] [A→B transition] [B body] [B→C transition] ...            │
└──────────────────────────────────────────────────────────────────────────────────────────┘
     │  chunks (PCM/Opus)
     ▼
  STITCHER  → continuous timeline, sample-accurate joins (no gaps, no clicks)
     │
     ▼
  SERVE   ── HLS (segmented .m4s + playlist)  OR  progressive Opus/WebM over HTTP
     │
     ▼
  CLIENT (browser <audio>/hls.js)  → one uninterrupted stream
```

The repo already has the skeleton: `mix_server.py` (Flask) + `src/stream_manager.py` (body chunks + transition chunks + continuous writer). We harden it into the producer/stitcher above and swap the transition step to `club_mixer`.

---

## 2. Demucs: fast, no quality compromise

The quality model stays **htdemucs** (no downgrade). Speed comes from doing *less work* and *never repeating it*, not from a worse model.

| Lever | Effect | Status |
|---|---|---|
| **Run on MPS** (Apple GPU) / CUDA | 5–10× vs CPU | device auto-select already in `utils.get_best_device`; **blocked only by broken torchaudio** (see §4) |
| **Separate only the mix regions** (~16–32 bars each, not the whole track) | 10–20× less audio through the model | already via `separate_segment`; club_mixer needs only the overlap windows |
| **Persistent stem cache** (content-hash → .npz, + in-memory LRU) | each segment separated **once**; look-ahead/re-render/repeat = free | ✅ implemented in `stem_separator.py` (verified: compute runs once, bit-identical reload) |
| **Silence-skip** | no model call on near-silent segments | ✅ implemented |
| Prefetch + pipeline overlap | separation for N+1 happens *during* N playback → hidden latency | producer design (§1) |
| fp16 on GPU (optional) | ~2× | flag-gated; off by default (MPS half is flaky on some torch builds) |
| `shifts=1`, `overlap=0.25` | already the fast inference settings | ✅ present |
| Bigger model only if idle | htdemucs_ft optionally when buffer is deep | future toggle |

Net: a 16-bar overlap at 128 BPM ≈ 30 s of audio per side → demucs on MPS handles that comfortably **within** the playback time of the current track, so stems never gate the stream once we're ahead.

---

## 3. Gapless guarantees
- **Sample-accurate stitching**: producer returns chunk sample-offsets; stitcher concatenates with the short equal-power splices already in `club_mixer._splice` (no clicks) and **no silence padding** (the Tier-3 fix removed the duplicate-head bug; the stream_manager temp-file race was also fixed).
- **Constant sample rate / channels** end-to-end (44.1k stereo); one continuous encoder (don't reopen per chunk).
- **Buffer policy**: keep ≥ ~20 s of rendered audio queued; producer wakes when buffer < threshold. Back-pressure if the client pauses.
- **Loudness continuity**: per-track K-weighted LUFS match (already in club_mixer) so levels don't jump across a long set.

---

## 4. Environment fix (the actual current blocker)
Your env: **Python 3.14 + torch 2.11.0 (nightly), MPS available, but `torchaudio` won't load** → demucs silently falls back, so you've effectively been mixing *without* stems. Fix = a clean, pinned, MPS-capable env:

```bash
# isolated venv, do NOT touch system python  (VERIFIED working: torch+torchaudio 2.2.2, MPS=True, demucs htdemucs loads & separates on MPS)
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.2.2 torchaudio==2.2.2          # matched pair, arm64 wheels, MPS
pip install -r requirements.txt
pip install --no-deps 'git+https://github.com/adefossez/demucs.git'
pip install dora-search einops julius openunmix     # demucs runtime deps (skipped by --no-deps; lameenc NOT needed, we don't encode mp3)
pip install "setuptools<81"                          # librosa needs pkg_resources, removed in setuptools 81+
```
(torch/torchaudio **must** be the same version. Python 3.12 has stable wheels; 3.14 does not yet for this stack.) Run the app with `./.venv/bin/python club_server.py`. After this, demucs runs on MPS and the cache + segment-only path make it fast (measured: ~7.6 s for an 8 s segment first call, **0.012 s** cached).

---

## 5. Deployment shapes
- **Local / personal (Mac, MPS, single listener)** — simplest. FastAPI/Flask + progressive Opus stream + on-disk caches. Good first target; runs on your machine.
- **Cloud / multi-user** — GPU instance(s), FastAPI, a job queue (per-session producer), object storage for caches, HLS for adaptive playback, autoscaling. Bigger build; needed only for real "deployable for others."

These differ a lot (concurrency, GPU, serving), so we pick one before building the serving layer.

---

## 6. Phases
- **P0 — Env fix** (§4): get demucs running on MPS (unblocks real stems). *Then verify a club_mixer blend WITH stems.*
- **P1 — Producer pipeline**: playlist resolve + prefetch + cached analysis + region-only cached separation + `club_mixer` transitions, emitting ordered chunks. (Hardens `stream_manager`.)
- **P2 — Gapless stitcher + server**: continuous encoder, buffer/back-pressure, HTTP stream; minimal web UI (paste URL → play).
- **P3 — Polish**: structure-aware mix points (SSM+Foote, plan B5), energy-arc track ordering (B6), HLS/adaptive, multi-user if cloud.

---

## 7. What's already done toward this
- Engine starts & runs (Tier 0–4 fixes); correct LUFS/equal-power/EQ.
- `club_mixer.py`: phrase-locked club blends (verified).
- `stem_separator.py`: **MPS-aware + persistent stem cache + silence-skip** (verified) — the demucs-efficiency core.
- `stream_manager.py`: concurrency/temp-file/leak bugs fixed (Tier 3).
