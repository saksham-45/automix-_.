#!/usr/bin/env python3
"""Download two YouTube videos and create a mix"""
import sys
from pathlib import Path
import argparse

# Ensure temp_audio directory exists
temp_dir = Path('temp_audio')
temp_dir.mkdir(exist_ok=True)

# Use extracted download function
from src.youtube_downloader import download_youtube_audio

def main():
    parser = argparse.ArgumentParser(description='Create DJ mix from two YouTube URLs')
    parser.add_argument('url1', help='YouTube URL for song A (outgoing)')
    parser.add_argument('url2', help='YouTube URL for song B (incoming)')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Max duration to download from each video (seconds, default: 60)')
    parser.add_argument('--keep', action='store_true',
                       help='Keep downloaded audio files (default: delete after mixing)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DOWNLOADING YOUTUBE AUDIO")
    print("="*60)
    
    song_a_path = temp_dir / 'song_a.wav'
    song_b_path = temp_dir / 'song_b.wav'
    
    # Clean up old files
    if song_a_path.exists():
        song_a_path.unlink()
    if song_b_path.exists():
        song_b_path.unlink()
    
    try:
        # Download both songs
        # Song A: last 60 seconds (from_end=True)
        download_youtube_audio(args.url1, song_a_path, args.duration, from_end=True)
        # Song B: first 60 seconds (from_end=False)
        download_youtube_audio(args.url2, song_b_path, args.duration, from_end=False)
        
        print("\n" + "="*60)
        print("CREATING MIX")
        print("="*60)
        print("\nNow running create_mix.py...\n")
        
        # Run create_mix.py
        import create_mix
        # The create_mix.py will use temp_audio/song_a.wav and temp_audio/song_b.wav
        
    except KeyboardInterrupt:
        print("\n\n⚠ Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if not args.keep:
            print("\n" + "="*60)
            print("CLEANUP")
            print("="*60)
            if song_a_path.exists():
                song_a_path.unlink()
                print(f"  ✓ Deleted: {song_a_path.name}")
            if song_b_path.exists():
                song_b_path.unlink()
                print(f"  ✓ Deleted: {song_b_path.name}")
        else:
            print(f"\n✓ Audio files kept in: {temp_dir}/")

if __name__ == '__main__':
    main()
