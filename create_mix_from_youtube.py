#!/usr/bin/env python3
"""Download two YouTube videos and create a mix"""
import sys
import subprocess
import shutil
from pathlib import Path
import argparse

# Ensure temp_audio directory exists
temp_dir = Path('temp_audio')
temp_dir.mkdir(exist_ok=True)

def download_youtube_audio(url: str, output_path: Path, max_duration: int = 60) -> Path:
    """Download audio from YouTube URL, limit to max_duration seconds."""
    print(f"Downloading: {url}")
    print(f"  → {output_path.name} (max {max_duration}s)")
    
    # First download, then trim with ffmpeg for more reliable duration limiting
    temp_output = output_path.parent / f"temp_{output_path.name}"
    
    # Use yt-dlp to download audio
    cmd = [
        'yt-dlp',
        '-x',  # Extract audio
        '--audio-format', 'wav',
        '--audio-quality', '0',  # Best quality
        '--no-playlist',
        '-o', str(temp_output).replace('.wav', '.%(ext)s'),
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # yt-dlp outputs with pattern: temp_song_a.wav or temp_song_a.opus (then converts)
        # Find any file that matches our temp pattern
        temp_pattern = temp_output.stem
        candidates = list(output_path.parent.glob(f"{temp_pattern}.*"))
        actual_file = None
        
        # Prefer .wav, then any audio file
        for ext in ['.wav', '.opus', '.m4a', '.mp3']:
            candidate = output_path.parent / f"{temp_pattern}{ext}"
            if candidate.exists():
                actual_file = candidate
                break
        
        if not actual_file and candidates:
            actual_file = candidates[0]
        
        if actual_file and actual_file.exists():
            # Trim to max_duration using ffmpeg
            if max_duration > 0:
                print(f"  Trimming to {max_duration} seconds...")
                trim_cmd = [
                    'ffmpeg', '-y', '-i', str(actual_file),
                    '-t', str(max_duration),
                    '-acodec', 'pcm_s16le',
                    '-ar', '44100',
                    str(output_path)
                ]
                subprocess.run(trim_cmd, capture_output=True, check=True)
                actual_file.unlink()  # Delete temp file
            else:
                # Convert/rename to .wav if needed
                if actual_file.suffix != '.wav':
                    convert_cmd = [
                        'ffmpeg', '-y', '-i', str(actual_file),
                        '-acodec', 'pcm_s16le', '-ar', '44100',
                        str(output_path)
                    ]
                    subprocess.run(convert_cmd, capture_output=True, check=True)
                    actual_file.unlink()
                elif actual_file != output_path:
                    shutil.move(str(actual_file), str(output_path))
            
            if output_path.exists():
                print(f"  ✓ Downloaded: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
                return output_path
            else:
                raise FileNotFoundError(f"Output file not created: {output_path}")
        else:
            raise FileNotFoundError(f"Downloaded file not found: {output_path}")
            
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error downloading: {e}")
        print(f"  stderr: {e.stderr[:200]}")
        raise
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        raise

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
        download_youtube_audio(args.url1, song_a_path, args.duration)
        download_youtube_audio(args.url2, song_b_path, args.duration)
        
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

