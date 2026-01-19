#!/usr/bin/env python3
"""
YouTube streaming downloader - downloads only segments needed for transitions.
Optimized for real-time continuous mixing.
"""
import subprocess
import json
from pathlib import Path
from typing import Optional


def download_segment(url: str, 
                    output_path: Path,
                    start_time: Optional[float] = None,
                    duration: float = 60.0,
                    from_end: bool = False) -> Path:
    """
    Download only a specific segment from YouTube.
    
    This is the KEY function for streaming: download only what we need
    for transitions, not the entire song.
    
    Args:
        url: YouTube URL
        output_path: Where to save segment
        start_time: Start time in seconds (if None, uses from_end logic)
        duration: Segment duration in seconds
        from_end: If True, download last N seconds (requires total duration lookup)
        
    Returns:
        Path to downloaded segment
        
    Raises:
        subprocess.CalledProcessError: If download fails
        FileNotFoundError: If output file not created
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"  Downloading segment: {output_path.name}")
    if from_end:
        print(f"    → Last {duration}s")
    elif start_time is not None:
        print(f"    → From {start_time:.1f}s for {duration}s")
    else:
        print(f"    → First {duration}s")
    
    # If from_end, we need to know total duration first
    if from_end and start_time is None:
        # Get video info to find duration
        info_cmd = [
            'yt-dlp',
            '--dump-json',
            '--no-playlist',
            url
        ]
        
        try:
            result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
            video_info = json.loads(result.stdout)
            total_duration = video_info.get('duration', 0)
            
            if total_duration > 0:
                start_time = max(0, total_duration - duration)
                print(f"    Video duration: {total_duration:.1f}s, extracting from {start_time:.1f}s")
            else:
                # Fallback: assume we want last N seconds
                print(f"    ⚠ Could not determine duration, using first {duration}s")
                from_end = False
        except Exception as e:
            print(f"    ⚠ Could not get duration, using first {duration}s: {e}")
            from_end = False
    
    # Strategy: Download audio and extract segment using ffmpeg
    # For streaming optimization: We download in two steps:
    # 1. Download audio (yt-dlp handles format selection)
    # 2. Extract segment with ffmpeg (can seek without full download with -ss before -i)
    # 
    # Note: True HTTP range requests aren't easily possible with YouTube streams,
    # but ffmpeg with -ss before -i allows seeking without downloading full file first.
    
    temp_full = output_path.parent / f"temp_full_{output_path.stem}"
    
    # Try multiple download strategies to handle YouTube changes
    download_attempts = [
        # Strategy 1: Direct audio format
        [
            'yt-dlp',
            '-f', 'bestaudio[ext=m4a]/bestaudio/best',
            '--no-playlist',
            '--extract-audio',
            '--audio-format', 'wav',
            '--audio-quality', '0',
            '-o', str(temp_full).replace('.wav', '.%(ext)s'),
            '--no-warnings',
            url
        ],
        # Strategy 2: Use ffmpeg for conversion (if format fails)
        [
            'yt-dlp',
            '-f', 'bestaudio',
            '--no-playlist',
            '-o', str(temp_full).replace('.wav', '.%(ext)s'),
            '--no-warnings',
            url
        ],
        # Strategy 3: Simple best format
        [
            'yt-dlp',
            '-f', 'best[height<=720]',
            '--no-playlist',
            '-o', str(temp_full).replace('.wav', '.%(ext)s'),
            '--no-warnings',
            url
        ]
    ]
    
    download_success = False
    last_error = None
    
    for attempt_num, download_cmd in enumerate(download_attempts, 1):
        try:
            print(f"    Download attempt {attempt_num}/{len(download_attempts)}...")
            result = subprocess.run(download_cmd, capture_output=True, text=True, check=False, timeout=180)
            
            if result.returncode == 0:
                download_success = True
                break
            else:
                error_msg = (result.stderr or result.stdout or "Unknown error")[:300]
                print(f"    ⚠ Attempt {attempt_num} failed: {error_msg}")
                last_error = error_msg
                
        except subprocess.TimeoutExpired:
            print(f"    ⚠ Attempt {attempt_num} timed out")
            last_error = "Download timed out"
        except Exception as e:
            print(f"    ⚠ Attempt {attempt_num} error: {e}")
            last_error = str(e)
    
    if not download_success:
        raise Exception(f"All download attempts failed. Last error: {last_error}")
    
    try:
        # Find downloaded file (yt-dlp may use different extensions)
        actual_file = None
        for ext in ['.wav', '.opus', '.m4a', '.mp3', '.webm', '.mkv', '.mp4']:
            candidate = output_path.parent / f"temp_full_{output_path.stem}{ext}"
            if candidate.exists():
                actual_file = candidate
                break
        
        if not actual_file:
            # Try to find any temp file with the pattern
            candidates = list(output_path.parent.glob(f"temp_full_{output_path.stem}.*"))
            if candidates:
                actual_file = candidates[0]
        
        if not actual_file or not actual_file.exists():
            raise FileNotFoundError(f"Downloaded file not found after download")
        
        # Extract segment using ffmpeg
        # Using -ss before -i allows ffmpeg to seek without downloading full file first
        if start_time is not None:
            # Extract from specific start time
            # -ss before -i = input seeking (faster, may not be frame-accurate)
            trim_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', str(start_time),
                '-i', str(actual_file),
                '-t', str(duration),
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                str(output_path)
            ]
        else:
            # Extract first N seconds
            trim_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', str(actual_file),
                '-t', str(duration),
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                str(output_path)
            ]
        
        # Run ffmpeg (may fail if file format is incompatible - try to handle)
        trim_result = subprocess.run(trim_cmd, capture_output=True, text=True, check=False)
        
        if trim_result.returncode != 0:
            # If ffmpeg fails, try to convert to WAV first
            print(f"    ⚠ Direct trim failed, converting format first...")
            convert_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', str(actual_file),
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                str(actual_file.parent / f"converted_{actual_file.stem}.wav")
            ]
            subprocess.run(convert_cmd, capture_output=True, check=True)
            actual_file = actual_file.parent / f"converted_{actual_file.stem}.wav"
            
            # Retry trim
            if start_time is not None:
                trim_cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-ss', str(start_time),
                    '-i', str(actual_file),
                    '-t', str(duration),
                    '-acodec', 'pcm_s16le',
                    '-ar', '44100',
                    str(output_path)
                ]
            else:
                trim_cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', str(actual_file),
                    '-t', str(duration),
                    '-acodec', 'pcm_s16le',
                    '-ar', '44100',
                    str(output_path)
                ]
            subprocess.run(trim_cmd, capture_output=True, check=True)
        
        # Cleanup temp file
        try:
            actual_file.unlink()
        except:
            pass
        
        if output_path.exists():
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"    ✓ Segment downloaded: {size_mb:.1f} MB, {duration}s")
            return output_path
        else:
            raise FileNotFoundError(f"Output file not created: {output_path}")
            
    except FileNotFoundError as e:
        print(f"    ✗ {e}")
        raise
    except Exception as e:
        print(f"    ✗ Segment processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise
