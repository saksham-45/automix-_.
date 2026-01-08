#!/bin/bash
# Quick render using existing transition data
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "Creating quick mix from existing transition data..."
echo ""
echo "Step 1: Downloading audio clips (30 seconds each)..."
yt-dlp -x --audio-format wav --postprocessor-args "ffmpeg:-t 30" \
  -o /tmp/song_a.wav "https://youtu.be/YE93dD6jjoU?si=I4SeUbjVGivIbeCm" 2>&1 | tail -3

yt-dlp -x --audio-format wav --postprocessor-args "ffmpeg:-t 30" \
  -o /tmp/song_b.wav "https://youtu.be/MHryuYVyHhk?si=8oLFIgeREpwAGM1K" 2>&1 | tail -3

echo ""
echo "Step 2: Rendering mix..."
python scripts/render_simple_mix.py \
  --song-a /tmp/song_a.wav \
  --song-b /tmp/song_b.wav \
  --transition-json youtube_transition.json \
  --output quick_mix.wav

if [ -f quick_mix.wav ]; then
  echo ""
  echo "✓ Quick mix ready: quick_mix.wav"
  echo "  Size: $(ls -lh quick_mix.wav | awk '{print $5}')"
  echo ""
  echo "To listen: open quick_mix.wav"
else
  echo "Error: Mix file not created"
fi
