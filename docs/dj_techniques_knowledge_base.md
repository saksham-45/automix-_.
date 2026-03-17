# DJ Techniques Knowledge Base

Professional DJ mixing techniques compiled from textbooks, online resources, and practice.

## Energy Management Techniques

### 1. Phrase Matching / Phrase Alignment
- **Description**: Aligning transitions to 8/16/32-bar phrase boundaries for musical coherence
- **When to use**: When both tracks have clear phrase structures, harmonically compatible
- **Duration**: 16-32 bars
- **Energy direction**: Maintain or build
- **Key principle**: Musical phrases are typically 8, 16, or 32 bars. Matching phrases creates natural transitions

### 2. Double Drop
- **Description**: Timing two drops together for maximum energy peak
- **When to use**: High-energy sections (drops, choruses), compatible keys
- **Duration**: 8-16 bars
- **Energy direction**: Up (peak)
- **Key principle**: Both tracks hit their drop/chorus at the same time for explosive impact

### 3. Triple Drop
- **Description**: Rare technique - three drops synchronized
- **When to use**: Special moments, festival sets
- **Duration**: 16 bars
- **Energy direction**: Up (peak)

## Rhythm & Timing Techniques

### 4. Backspin / Spinback
- **Description**: Spinning out outgoing track (reverse effect or tape stop)
- **When to use**: Closing a set, transitioning out of a high-energy section
- **Duration**: 4-8 bars
- **Energy direction**: Down
- **Key principle**: Creates dramatic exit before incoming track

### 5. Rewind / Reverse
- **Description**: Reversing audio before transition
- **When to use**: Creative transitions, breakdown sections
- **Duration**: 4-8 bars
- **Energy direction**: Down
- **Key principle**: Builds tension through reverse audio

### 6. Power Down
- **Description**: Energy dip with filter sweep before building back up
- **When to use**: Transitioning to lower energy sections
- **Duration**: 8-16 bars
- **Energy direction**: Down then up

### 7. Power Up
- **Description**: Progressive energy increase with filtering and EQ automation
- **When to use**: Building from verse to chorus, breakdown to drop
- **Duration**: 16-32 bars
- **Energy direction**: Up (progressive)

## Harmonic & Melodic Techniques

### 8. Modulation / Key Change Transition
- **Description**: Smooth key change during transition
- **When to use**: When keys are related (circle of fifths, relative keys)
- **Duration**: 16-32 bars
- **Energy direction**: Maintain
- **Key principle**: Use harmonic analysis to find compatible key changes

### 9. Acapella Overlay / Acapella Layering
- **Description**: Layer acapella from one track over instrumental of another
- **When to use**: When one track has strong vocals, other has strong beat
- **Duration**: 16-32 bars
- **Energy direction**: Maintain or build
- **Key principle**: Vocals from Track A + Beat from Track B

### 10. Melodic Mixing
- **Description**: Transition focusing on melody compatibility
- **When to use**: Melodic tracks, compatible keys
- **Duration**: 24-32 bars
- **Energy direction**: Maintain
- **Key principle**: Melodies should complement, not clash

## Creative & Effects Techniques

### 11. Loop Transition
- **Description**: Use loops to create smooth mixing points. The system repeats a short loop from the end of the outgoing track, then crossfades into the incoming track.
- **When to use**: Any section, especially when phrases don't align; selected when structure/energy fit (verse, chorus, breakdown, maintain).
- **Duration**: Typically 16 bars; configurable via loop_length_bars and loop_repeats.
- **Energy direction**: Maintain
- **Key principle**: Looping a section creates a "safe" transition point. Now selected by the strategist when structure and energy fit, and uses real looping (not a simple crossfade).

### 12. Breakdown-to-Build Transition
- **Description**: Transition from breakdown to build section
- **When to use**: Track A in breakdown, Track B building up
- **Duration**: 16-24 bars
- **Energy direction**: Up (progressive)
- **Key principle**: Match the energy arc - breakdown → build → drop

### 13. Energy Build
- **Description**: Progressive energy increase during transition
- **When to use**: Building intensity before a drop or chorus
- **Duration**: 16-32 bars
- **Energy direction**: Up (progressive)
- **Key principle**: Gradual EQ, filter, and volume automation

## Specialized Techniques

### 14. Hot Cue Mixing
- **Description**: Using hot cues to jump between sections quickly
- **When to use**: When precise timing is needed
- **Duration**: Variable
- **Key principle**: Allows DJ to "skip" to optimal transition points

### 15. Transformer Cuts (Fast Cuts)
- **Description**: Rapid on/off cuts with fader (hip-hop technique)
- **When to use**: High-energy sections, beat-heavy tracks
- **Duration**: 2-4 bars
- **Energy direction**: Up (maintain energy)
- **Key principle**: Fast fader chops create rhythmic emphasis

### 16. Drop on the One
- **Description**: Cut the outgoing track and bring in the new track exactly on the first beat of a bar (instant cut on downbeat).
- **When to use**: High-energy sections (drop, chorus); energy direction up.
- **Duration**: ~2 bars (short).
- **Energy direction**: Up
- **Key principle**: Minimal crossfade (e.g. 50 ms); effective for hip-hop or radio-style transitions.

### 17. Back-and-Forth / Switch
- **Description**: Alternating A and B every N bars (e.g. 8 or 16) for a call-and-response feel, then settling on B.
- **When to use**: Verse/chorus sections, harmonically compatible, energy maintain.
- **Duration**: ~24 bars.
- **Energy direction**: Maintain
- **Key principle**: Short crossfades at each switch; final segment is B only.

### 18. Drum Roll / Percussion Bridge
- **Description**: Short percussive build or "roll" (high-pass sweep on A + repeated slice) then cut to B on a strong beat.
- **When to use**: Building energy; incoming section is build, drop, or chorus.
- **Duration**: ~8 bars.
- **Energy direction**: Up
- **Key principle**: First half: high-pass sweep on A and repeated slice for roll; second half: quick crossfade to B.

### 19. Thematic Handoff
- **Description**: Phrase-aligned blend when thematic or lyric compatibility is present (same audible behavior as phrase_match).
- **When to use**: When a future thematic_compatibility signal is available; structure preference verse, chorus, breakdown.
- **Duration**: ~24 bars.
- **Energy direction**: Maintain
- **Key principle**: Same execution as phrase_match; selected when lyrics/themes are added later.

## Implementation Priority

**High Priority** (Most commonly used):
1. phrase_match - Essential for professional mixing
2. backspin - Classic technique
3. double_drop - High impact
4. acapella_overlay - Creative and effective
5. modulation - Advanced harmonic mixing

**Medium Priority**:
6. energy_build - Progressive transitions
7. loop_transition - Flexible mixing tool (now selected when structure/energy fit; uses real looping)
8. breakdown_to_build - Energy arc matching
9. drop_on_the_one - Instant cut on downbeat
10. back_and_forth - Call-and-response switch
11. drum_roll - Percussion bridge
12. thematic_handoff - Phrase-aligned when thematic signal present

**Lower Priority** (Specialized):
13. rewind/reverse - Creative but less common
14. transformer_cuts - Genre-specific (hip-hop)

## Notes

- All techniques should respect musical structure (phrases, bars)
- Harmonic compatibility is crucial for smooth transitions
- Energy direction matching prevents awkward energy jumps
- Stem separation enables more creative techniques (acapella, partial mixing)
- Techniques can be combined (e.g., phrase_match + energy_build)
