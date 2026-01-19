# Fixing Playback Continuity (Song Entry Offset)

## Goal
Eliminate the "replay" artifact where the intro of a song is heard twice (once in the transition, and again when the song starts). Ensure the playlist flows as a continuous DJ mix by starting each song exactly where its incoming transition ends.

## The Problem
Currently, the pipeline is:
1. **Transition N:** Plays `[Song A End] mixed with [Song B Start]`.
2. **Chunk N+1 (Song B):** Plays `[Song B Full (starting at 0:00)]`.
3. **Result:** The start of Song B is heard twice. The stream jumps backward in time after the transition.

## Proposed Solution
Modify `StreamingMixer` to "Structure-Aware trimming" of the *incoming* song.

1. **In `get_audio_chunk(song_index)`:**
   - Check if there is a **previous transition** (Transition `song_index - 1`).
   - Load its metadata (`transition_{song_index-1}.json`).
   - Extract:
     - `point_b`: The timestamp in Song B where the transition started.
     - `duration`: The length of the transition.
   - Calculate `start_offset = point_b + duration` (minus a small handover overlap if needed).
   - **Trim Song B:** `song_audio = song_audio[int(start_offset * sr) :]`.

2. **Handle Fallback/Legacy:**
   - If no metadata exists, assume the transition used the first `transition_duration` seconds (default 16s).
   - Trim the first 16s.

## Implementation Details

### `api/streaming_mixer.py`

#### `get_audio_chunk`
- Add logic to retrieve `prev_transition_metadata`.
- Since `Chunk N` maps to `Song N`, we look for `Transition N-1`.
- `Transition N-1` connects `Song N-1` -> `Song N`.
- Use `batch_processor.get_transition(song_index - 1)` (which returns metadata tuple).
- Apply slicing: `song_audio = song_audio[start_sample:]`.

#### Safety Checks
- Ensure `start_offset` < `len(song_audio)`.
- If trimmed length is too short, warn or fallback.

## Verification
- **Manual Listen:** Verify that the end of the transition flows perfectly into the body of the next song without repeating the intro.
- **Waveform Inspection:** (Optional) Check that the seamless join is click-free.
