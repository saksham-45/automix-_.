#!/usr/bin/env python3
"""
YouTube audio downloader - reusable module for downloading YouTube audio.
Extracted from create_mix_from_youtube.py for reuse in continuous mixer.
"""
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_youtube_audio(url: str, output_path: Path, 
                          max_duration: int = 60, 
                          from_end: bool = False) -> Path:
    """
    Download audio from YouTube URL, limit to max_duration seconds.
    
    Args:
        url: YouTube URL
        output_path: Path to save the audio file
        max_duration: Duration in seconds to extract (0 = full video)
        from_end: If True, extract last N seconds. If False, extract first N seconds.
        
    Returns:
        Path to downloaded audio file
        
    Raises:
        subprocess.CalledProcessError: If download fails
        FileNotFoundError: If output file not created
    """
    print(f"Downloading: {url}")
    if max_duration > 0:
        if from_end:
            print(f"  → {output_path.name} (last {max_duration}s)")
        else:
            print(f"  → {output_path.name} (first {max_duration}s)")
    else:
        print(f"  → {output_path.name} (full video)")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # First download, then trim with ffmpeg for more reliable duration limiting
    temp_output = output_path.parent / f"temp_{output_path.name}"
    
    # Use yt-dlp to download audio.
    # Use android player_client to avoid SABR streaming issues on YouTube.
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
        # Add timeout to prevent hanging forever (120s for download)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
        
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
                    # Timeout for probe
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, timeout=60)
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
                
                # Timeout for ffmpeg
                subprocess.run(trim_cmd, capture_output=True, check=True, timeout=60)
                actual_file.unlink()  # Delete temp file
            else:
                # Convert/rename to .wav if needed (no duration limit)
                if actual_file.suffix != '.wav':
                    convert_cmd = [
                        'ffmpeg', '-y', '-i', str(actual_file),
                        '-acodec', 'pcm_s16le', '-ar', '44100',
                        str(output_path)
                    ]
                    subprocess.run(convert_cmd, capture_output=True, check=True, timeout=60)
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
            
    except subprocess.TimeoutExpired as e:
        print(f"  ✗ Download timed out: {e}")
        raise
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error downloading: {e}")
        print(f"  stderr: {e.stderr[:200]}")
        raise
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        raise


def download_youtube_audio_parallel(urls: List[Tuple[str, Path, int, bool]],
                                   max_workers: int = 4) -> Dict[str, Path]:
    """
    Download multiple YouTube audio files in parallel.
    
    Args:
        urls: List of tuples (url, output_path, max_duration, from_end)
        max_workers: Maximum number of parallel downloads
        
    Returns:
        Dictionary mapping URL to downloaded Path (or None if failed)
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_url = {
            executor.submit(download_youtube_audio, url, output_path, max_duration, from_end): url
            for url, output_path, max_duration, from_end in urls
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                output_path = future.result()
                results[url] = output_path
                print(f"  ✓ Completed: {output_path.name}")
            except Exception as e:
                print(f"  ✗ Failed to download {url}: {e}")
                results[url] = None
    
    return results
