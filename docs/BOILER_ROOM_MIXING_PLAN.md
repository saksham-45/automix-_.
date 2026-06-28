# Boiler-Room–Style Club Mixing — Research-Grounded Implementation Plan

> Goal: make `autmox` produce **long, beat-matched, harmonically-mixed club blends** (techno / house / melodic-progressive, 4/4, ~120–135 BPM) — the underground/Boiler-Room aesthetic — instead of short radio-style cuts. This plan is grounded in a deep, adversarially-verified literature review (see **§7 References**) and mapped onto the existing codebase.

---

## 1. What the research actually says (the model we build to)

The academic auto-DJ literature converges on a small, concrete model. Every design decision below traces to a verified finding.

### 1.1 A transition is three timestamps — `t1 / t2 / t3`
A DJ transition is formally defined by three times (Zehren et al., *Computer Music Journal* 46(3), 2022):
- **t1** — incoming track **B becomes audible** (crossfade starts; A still dominant)
- **t2** — the **switch point**: B becomes louder / musically prevalent than A *(the single most critical mix point)*
- **t3** — outgoing track **A becomes inaudible** (transition ends)

This yields a **four-state model**: `Only A → A-prevalent → B-prevalent → Only B`. The whole engine is organized around computing good `t1/t2/t3` and rendering the four states.

### 1.2 The placement law (hard constraints)
Switch points are **not** placed freely. Per Zehren et al.:
- A switch point is **always on a strong beat** (beats 1 & 3 of a 4/4 bar).
- A switch point is **always on the downbeat of the first bar of a phrase** (a "period" = 4 bars, or a multiple).

→ Engineering rule: **snap candidate points to downbeats, and only allow phrase-multiple boundaries (4 / 8 bars).**

### 1.3 Where the mix regions are (track structure)
EDM is "invariably 4/4," built from 2-bar phrases combined into periods, organized intro / core / outro with **breakdowns** and **build-ups**. A practical 7-class taxonomy: `intro, build-up, drop, breakdown, outro, silence, end`.
- **Mix-IN** happens on the incoming track's **intro / breakdown** (low energy, beat present, no lead).
- **Mix-OUT** happens on the outgoing track's **outro / breakdown**.
- **Never collide two drops** unless deliberately doing a *double-drop*.

### 1.4 How to find phrase boundaries
Canonical method (Foote 2000 / Müller FMP; Vande Veire & De Bie 2018):
1. Chroma (and timbre/energy) **self-similarity matrix (SSM)**.
2. Correlate a **Gaussian checkerboard kernel** down the diagonal → a **novelty function**; peaks = section boundaries.
3. **Snap each boundary to the nearest downbeat**; discard peaks **>0.4 bars** away.
4. **Force boundaries onto phrase multiples** (e.g. multiples of 8 bars).

Most predictive features for mix points in EDM: **energy novelty, timbre, drum-onset count, harmony**.

### 1.5 Tempo handling
One published end-to-end system fixes a **single global mix tempo** and **time-stretches every track to it via WSOLA**, then beat-matches to the previous track before fading/EQ. We adopt this (global mix BPM in the 120–135 range), with the open caveat that large tempo deltas / half-time–double-time need guarding.

### 1.6 Transition-type taxonomy + EQ pattern + FSM
Three core club types, chosen by the **energy of the overlapped segments**:
- **double_drop** — two drops/high-energy sections aligned (peak-time move)
- **rolling** — main/core of A into the next track's drop/core
- **relaxed** — low-energy outro/breakdown into another low-energy section

Each type drives a **fixed EQ + volume automation pattern**. A **Finite State Machine** prevents the same type twice in a row (variety). Broader technique palette in the literature: beat-matched crossfade (baseline), **3-band EQ mixing**, **bass swap**, looping/drum-roll, **echo-out**, power-down, slam/cut.

### 1.7 Next-track selection = "mixability"
Pick the next track by **tempo + rhythm + harmony** compatibility (transition-level and/or track-level). Harmony uses the **Camelot wheel** (number 1–12 + letter; A=minor, B=major). Compatible moves: **±1 same letter**, **same number / opposite letter** (relative major↔minor). A continuous **harmonic-compatibility score** (Bibbó & Faraldo 2022, TIV-based) improved mixes in 73.7% of expert-judged cases → usable as an objective term.

### 1.8 What the research did NOT pin (we set sensible defaults)
Open, so we choose defaults from practitioner sources + DSP best practice (see §5):
exact blend length in bars/seconds, crossfade curve shape, EQ automation timing, bass-swap mechanics, LUFS target, BPM-match tolerance, half/double-time policy, +2/+7 Camelot energy moves.

### 1.9 Explicitly refuted (do NOT do)
- ❌ Modeling a transition as *just* "EQ + fader" (too simplistic — refuted 0-3).
- ❌ Treating beat-matched crossfade as THE canonical technique (it's a baseline only).

---

## 2. Design — the club-mix pipeline

```
Track A, Track B (audio + cached analysis)
      │
      ▼
[1] PhraseGrid(A), PhraseGrid(B)            ← beats → downbeat phase → bars → 8-bar phrases → energy/section labels
      │
      ▼
[2] mixability check (tempo/rhythm/Camelot) ← gate + score (for next-track selection)
      │
      ▼
[3] TransitionPlan: pick mix-OUT phrase in A (outro/breakdown) + mix-IN phrase in B (intro/breakdown)
      │   → compute t1, t2 (switch=phrase downbeat), t3 ; choose type {relaxed|rolling|double_drop} via FSM
      ▼
[4] Render:
      • tempo-match B→mixBPM (WSOLA/time-stretch), downbeat-lock to A
      • 4-state equal-power volume crossfade across t1..t3
      • 3-band EQ BASS-SWAP exactly on the t2 downbeat (kill incoming lows pre-t2, swap at t2)
      • loudness match (K-weighted LUFS) + −1 dBTP headroom
      ▼
   Club blend (stereo, 44.1k)
```

### 2.1 The `t1/t2/t3` math (phrase-locked)
Let `bar = 4 * 60 / mixBPM` seconds, phrase `P` bars (default **8**), blend length `L` bars (default **16**, range 8–32).
- **t2 (switch)** = a downbeat at the start of A's chosen mix-out phrase (and the matching downbeat of B's mix-in phrase).
- **t1** = `t2 − (L/2) bars` (B fades in for the first half).
- **t3** = `t2 + (L/2) bars` (A fades out over the second half).
- All three are quantized to the **beat grid** and clamped to phrase boundaries.

### 2.2 EQ bass-swap (the core club move)
Two kicks/basslines must never stack. Using a low/mid/high 3-band split (crossovers ~250 Hz and ~2.5 kHz):
- **t1 → t2:** A full; **B low band killed** (high-passed), B mids/highs fade in.
- **at t2 (downbeat):** **swap the low band** — A low band killed, B low band restored — on the beat so the kick "hands off" cleanly.
- **t2 → t3:** A mids/highs fade out; B full.

### 2.3 Transition-type selection (FSM)
Label the overlapped A-segment and B-segment energy as `low`/`high`:
- `low→low` ⇒ **relaxed** (long, gentle, 24–32 bars)
- `high→high` (both drops) ⇒ **double_drop** (short align, 4–8 bars, only if phase + key safe)
- otherwise ⇒ **rolling** (medium, 16 bars)
FSM forbids repeating the same type back-to-back across a set.

---

## 3. Mapping to the existing codebase

| Need | Existing seam | Action |
|---|---|---|
| Beat grid | `BeatAligner.get_beat_grid`, `advanced_beatmatcher` | reuse; add robust **downbeat-phase** estimate (current "every 4th beat" is naive) |
| Phrase/section boundaries | `structure_analyzer` (`downbeats`, `phrases`, `sections`, `best_mix_in/out_points`) | upgrade detection (SSM+Foote novelty, 0.4-bar snap, 8-bar quantize, EDM labels) |
| Harmonic compat | `harmonic_analyzer` (Camelot, `are_keys_compatible`) | reuse for mixability score |
| Equal-power curves | `crossfade_engine.create_equal_power_crossfade` | reuse for the 4-state fade |
| Loudness | `psychoacoustics.analyze_loudness_lufs` (K-weighting now correct) | reuse for level match |
| Technique routing | `transition_strategist`, `technique_executor` | register a new `club_blend` technique |
| Orchestration | `smart_mixer` / `superhuman_engine` | add a clean `create_club_mix` path; don't destabilize existing engine |

**Strategy:** ship a **self-contained `src/club_mixer.py`** that implements §2 end-to-end by *reusing* the primitives above, then expose it via a CLI and a `technique_executor` hook. This delivers the Boiler-Room capability without rewriting the tangled superhuman engine.

---

## 4. Phased implementation

- **Phase B2 — Grid + planner** (`club_mixer.py`): `PhraseGrid` (downbeat-phase estimation, 8-bar phrases, energy-section labels) + `plan_transition` (t1/t2/t3 on phrase downbeats, mix-region selection, type FSM). *Verify: mix points are bar/phrase-aligned.*
- **Phase B3 — Renderer**: tempo-match + downbeat-lock, 4-state equal-power crossfade, beat-locked 3-band EQ bass-swap, loudness match + headroom. *Verify: finite, −1 dBTP, switch lands on a downbeat.*
- **Phase B4 — Wire + test**: CLI `scripts/club_mix.py` + `club_blend` technique; tests on repo clips. *Verify end-to-end.*
- **Phase B5 (later) — Structure upgrade**: replace heuristic boundaries in `structure_analyzer` with SSM+Foote novelty; feed real intro/breakdown/outro labels into region selection.
- **Phase B6 (later) — Set sequencing**: "mixability"-driven next-track ordering to shape an energy arc (Camelot + tempo + energy).

---

## 5. Parameter defaults (research where pinned, best-practice otherwise)

| Parameter | Default | Basis |
|---|---|---|
| Time signature | 4/4 | research (EDM invariant) |
| Phrase length `P` | 8 bars | research (phrase-multiple rule) |
| Blend length `L` | 16 bars (relaxed 24–32, double-drop 4–8) | practitioner default; research = "long" |
| Downbeat-snap threshold | 0.4 bars | research (discard farther peaks) |
| Crossfade curve | equal-power (√cos) | DSP best practice (constant perceived loudness) |
| EQ crossovers | ~250 Hz, ~2.5 kHz | practitioner 3-band convention |
| Bass-swap timing | on the **t2 downbeat** | research (switch on downbeat) + practitioner |
| Tempo policy | stretch to global mix BPM | research (WSOLA global tempo) |
| BPM-match tolerance | ≤ ~6% stretch transparent; reject/half-time beyond | practitioner default (open in research) |
| Loudness target | match via K-weighted LUFS, then −1 dBTP ceiling | best practice (open in research) |
| Camelot moves | same; ±1 same letter; relative (same #, opp letter) | research (Mixed In Key) |

---

## 6. Risks / caveats
- **Genre transfer:** the most detailed numbers (175 BPM, 8-bar, 94.3% boundary acc.) come from a **Drum & Bass** system — re-tune for 120–135 house/techno (4/4 + 8-bar + Camelot findings *do* transfer).
- **Downbeat accuracy** is the foundation; naive "every-4th-beat" must be replaced or the whole grid drifts.
- **No real-time** here (offline render); a live engine is a separate effort (see prior Apple-AutoMix analysis).
- Verification is structural + signal-level (bar alignment, finite/−1 dBTP, no NaN); perceptual quality needs ears.

---

## 7. References (verified)
- Zehren, Alunno & Bientinesi — *Automatic Detection of Cue Points for the Emulation of DJ Mixing*, Computer Music Journal 46(3), 2022. (t1/t2/t3; downbeat/phrase placement law; EDM structure)
- Vande Veire & De Bie — *From raw audio to a seamless mix: an automated DJ system for Drum & Bass*, EURASIP J. ASMP, 2018. (SSM+novelty boundaries, WSOLA global tempo, 3 transition types + FSM)
- *Interpretability of Methods for Switch Point Detection in EDM*, MDPI Signals 5(4), 2024. (switch-point features: energy novelty, timbre, drum onsets, harmony)
- Müller, *FMP Notebooks* — Foote checkerboard-kernel novelty segmentation (Audiolabs Erlangen).
- Bibbó & Faraldo — *A New Compatibility Measure for Harmonic EDM Mixing*, ICWE 2022 (Springer LNCS). (continuous harmonic-compatibility score; 73.7%)
- Mixed In Key — *Harmonic Mixing Guide* (canonical Camelot ruleset).
- *Temporal Considerations in DJ Mix IR & Generation*, LIPIcs TIME 2025. (technique taxonomy; "mixability")
- EDMFormer, arXiv 2026 (EDM structure taxonomy; boundary-detection features).
