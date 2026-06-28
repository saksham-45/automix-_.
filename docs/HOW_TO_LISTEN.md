# How to Listen to Your AI DJ Mix

## Current Status

The mix rendering is running in the background. Here are your options:

### Option 1: Wait for Background Process (Recommended)
The full mix is being created automatically. Check if it's ready:

```bash
ls -lh ai_dj_mix.wav
```

If the file exists, you can play it:
```bash
open ai_dj_mix.wav  # macOS
# or use any audio player
```

### Option 2: Manual Rendering with Shorter Clips

If you want to test with shorter clips first, you can:

1. Download shorter versions of the songs (30-60 seconds):
```bash
yt-dlp -x --audio-format wav --postprocessor-args "ffmpeg:-t 60" -o song1.wav "YOUR_URL_1"
yt-dlp -x --audio-format wav --postprocessor-args "ffmpeg:-t 60" -o song2.wav "YOUR_URL_2"
```

2. Render using existing transition JSON:
```bash
python scripts/render_simple_mix.py \
  --song-a song1.wav \
  --song-b song2.wav \
  --transition-json youtube_transition.json \
  --output quick_test.wav
```

### Option 3: Use Audio Editing Software

You can manually apply the transition curves in any DAW:

1. Import both audio files
2. Use the volume automation from `youtube_transition.json`:
   - `curves.volume_a` and `curves.volume_b` for crossfade
   - `curves.bass_a/b`, `curves.mid_a/b`, `curves.high_a/b` for EQ
3. The transition is 4 bars at ~128 BPM (~7.5 seconds)

## Transition Details

- **Technique**: CUT
- **Duration**: 4 bars (~7.5 seconds)
- **Volume A**: Fades from 1.0 → 0.02
- **Volume B**: Fades from 0.0 → 0.99
- **Key Compatible**: YES (smooth harmonic transition)

## Quick Check

Check if the mix is ready:
```bash
ls -lh ai_dj_mix.wav && echo "Ready to play!" || echo "Still rendering..."
```

