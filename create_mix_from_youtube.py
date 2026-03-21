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


def resolve_to_youtube_url(url_or_query: str, prefer: str = "official audio") -> str:
    """If input is already a YouTube URL, return it (strip tracking params). Otherwise treat as search query and resolve to first result (official audio or lyrics) to avoid geo-blocks."""
    s = url_or_query.strip()
    if "youtu.be/" in s:
        return s.split("?")[0]
    if "youtube.com" in s and "watch?v=" in s:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(s)
        vid = (parse_qs(parsed.query).get("v") or [None])[0]
        if vid:
            return f"https://www.youtube.com/watch?v={vid}"
        return s
    # Search query: resolve to first "official audio" or "lyrics" result
    search = f'ytsearch1:{s} {prefer}'
    try:
        result = subprocess.run(
            ["yt-dlp", "-q", "--get-id", "--no-playlist", search],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            vid = result.stdout.strip().split("\n")[0].strip()
            return f"https://www.youtube.com/watch?v={vid}"
    except Exception:
        pass
    return url_or_query


def download_youtube_audio(url: str, output_path: Path, max_duration: int = 60, from_end: bool = False, prefer: str = "official audio") -> Path:
    """Download audio from YouTube URL, limit to max_duration seconds.
    
    Args:
        url: YouTube URL or search query (e.g. 'Tame Impala Let It Happen'). Queries are resolved to 'official audio' / lyrics to avoid geo-blocks.
        output_path: Path to save the audio file
        max_duration: Duration in seconds to extract
        from_end: If True, extract last N seconds. If False, extract first N seconds.
        prefer: When url is a search query, prefer 'official audio' or 'lyrics'.
    """
    url = resolve_to_youtube_url(url, prefer=prefer)
    print(f"Downloading: {url}")
    if from_end:
        print(f"  → {output_path.name} (last {max_duration}s)")
    else:
        print(f"  → {output_path.name} (first {max_duration}s)")
    
    # First download, then trim with ffmpeg for more reliable duration limiting
    temp_output = output_path.parent / f"temp_{output_path.name}"
    
    # Use yt-dlp to download audio (android client avoids YouTube SABR streaming issues)
    cmd = [
        'yt-dlp',
        '-x',  # Extract audio
        '--audio-format', 'wav',
        '--audio-quality', '0',  # Best quality
        '--no-playlist',
        '--extractor-args', 'youtube:player_client=android',
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
                if from_end:
                    # Get video duration first if we need the last N seconds
                    probe_cmd = [
                        'ffprobe', '-v', 'error',
                        '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1',
                        str(actual_file)
                    ]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                    total_duration = float(probe_result.stdout.strip())
                    
                    # Calculate start time (last N seconds)
                    start_time = max(0, total_duration - max_duration)
                    print(f"  Extracting last {max_duration}s (from {start_time:.1f}s to {total_duration:.1f}s)...")
                    
                    trim_cmd = [
                        'ffmpeg', '-y', '-i', str(actual_file),
                        '-ss', str(start_time),  # Seek to start position
                        '-t', str(max_duration),  # Extract N seconds
                        '-acodec', 'pcm_s16le',
                        '-ar', '44100',
                        str(output_path)
                    ]
                else:
                    # Extract first N seconds (existing behavior)
                    print(f"  Trimming to first {max_duration} seconds...")
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
    parser = argparse.ArgumentParser(
        description='Create DJ mix from two YouTube URLs or search queries. Prefers official audio / lyrics to avoid geo-blocks.'
    )
    parser.add_argument('url1', help='YouTube URL or search query for song A (outgoing), e.g. "Tame Impala Let It Happen"')
    parser.add_argument('url2', help='YouTube URL or search query for song B (incoming), e.g. "Tame Impala New Person Same Old Mistakes"')
    parser.add_argument('--duration', type=int, default=124, 
                       help='Max duration to download from each video (seconds, default: 124)')
    parser.add_argument('--prefer', choices=['official audio', 'lyrics'], default='official audio',
                       help='When using a search query, prefer "official audio" or "lyrics" (default: official audio)')
    parser.add_argument('--force-stem-orchestration', action='store_true',
                       help='Force stem orchestration instead of hybrid techniques')
    parser.add_argument('--conversation-type', type=str, default=None,
                       help='Stem handoff type: e.g. progressive_morph, melody_a_drums_vocals_b, bass_from_a_beat_from_b, vocal_overlay_handoff')
    parser.add_argument('--morph-depth', type=float, default=None,
                       help='Stem morphing depth 0.0-1.0 (default: from config, usually 0.8). Higher = more aggressive content transformation')
    parser.add_argument('--morph-strategy', type=str, default=None,
                       choices=['best_match', 'drums_first', 'all', 'drums', 'bass', 'vocals', 'other'],
                       help='Which stems to morph (default: from config, usually best_match)')
    parser.add_argument('--no-morph', action='store_true',
                       help='Disable stem morphing entirely')
    parser.add_argument('--transition', type=float, default=30.0,
                       help='Transition duration in seconds (default: 30). Longer = smoother morph.')
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
        # We need enough audio to play the whole transition AND some context.
        # If the user requests a 64s transition, we must download at least 64 + padding.
        target_duration = max(args.duration, int(args.transition) + 40)
        
        # Download both songs (prefer official audio / lyrics to avoid geo-blocks)
        # Song A: last N seconds (from_end=True)
        download_youtube_audio(args.url1, song_a_path, target_duration, from_end=True, prefer=args.prefer)
        # Song B: first N seconds (from_end=False)
        download_youtube_audio(args.url2, song_b_path, target_duration, from_end=False, prefer=args.prefer)
        
        print("\n" + "="*60)
        print("CREATING MIX")
        print("="*60)
        # Use same superhuman path as mix_runner / server (existing SmartMixer)
        import soundfile as sf
        from src.smart_mixer import SmartMixer
        from datetime import datetime
        mixer = SmartMixer()
        # Apply stem morphing overrides from CLI
        if mixer.superhuman_enabled and mixer.superhuman_engine is not None:
            morph_overrides = {}
            if args.no_morph:
                morph_overrides['stem_morphing_enabled'] = False
            if args.morph_depth is not None:
                morph_overrides['stem_morph_depth'] = args.morph_depth
            if args.morph_strategy is not None:
                morph_overrides['stem_morph_strategy'] = args.morph_strategy
            if morph_overrides:
                mixer.superhuman_engine.configure(**morph_overrides)
                print(f"  🧬 Stem morph overrides: {morph_overrides}")
        mixed_audio = mixer.create_superhuman_mix(
            str(song_a_path),
            str(song_b_path),
            transition_duration=args.transition,
            creativity_level=0.6,
            optimize_quality=True,
            force_stem_orchestration=getattr(args, 'force_stem_orchestration', False),
            conversation_type_override=getattr(args, 'conversation_type', None),
        )
        base_name = 'ai_dj_mix'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'{base_name}_{timestamp}.wav'
        sf.write(output_file, mixed_audio, mixer.sr)
        print(f"\nSaving mix to: {output_file}")
        print(f"Duration: {len(mixed_audio) / mixer.sr:.1f}s")
        # Move old mixes to data/old_mixes (same as create_mix.py)
        try:
            cwd = Path(__file__).resolve().parent
            old_mixes_dir = cwd / 'data' / 'old_mixes'
            old_mixes_dir.mkdir(parents=True, exist_ok=True)
            mix_files = sorted(
                [f for f in cwd.iterdir() if f.name.startswith(base_name) and f.suffix == '.wav'],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            for old_file in mix_files[1:]:
                try:
                    old_file.rename(old_mixes_dir / old_file.name)
                except Exception:
                    pass
        except Exception:
            pass
        print(f"\n✅ Mix complete: {output_file}")
        
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
