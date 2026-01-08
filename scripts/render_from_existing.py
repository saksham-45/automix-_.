#!/usr/bin/env python3
"""
Render mix using existing transition JSON and YouTube URLs.

Simpler version that re-downloads audio and renders using existing prediction.
"""
import sys
import subprocess
import json
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.render_simple_mix import render_simple_transition


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Render mix from existing transition JSON')
    parser.add_argument('--transition-json', default='youtube_transition.json',
                       help='Path to transition prediction JSON')
    parser.add_argument('--url1', required=True, help='YouTube URL for song 1')
    parser.add_argument('--url2', required=True, help='YouTube URL for song 2')
    parser.add_argument('--output', default='ai_dj_mix.wav',
                       help='Output audio file')
    
    args = parser.parse_args()
    
    temp_dir = Path(tempfile.mkdtemp(prefix='dj_render_'))
    audio1 = temp_dir / 'song_a.wav'
    audio2 = temp_dir / 'song_b.wav'
    
    try:
        print("Downloading audio...")
        subprocess.run(['yt-dlp', '-x', '--audio-format', 'wav', 
                       '-o', str(audio1), args.url1], 
                      capture_output=True, check=True)
        subprocess.run(['yt-dlp', '-x', '--audio-format', 'wav',
                       '-o', str(audio2), args.url2],
                      capture_output=True, check=True)
        
        print(f"\nRendering mix...")
        render_simple_transition(
            str(audio1),
            str(audio2),
            args.transition_json,
            args.output
        )
        
        print(f"\n✓ Mix ready: {args.output}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()

