#!/usr/bin/env python3
"""
Easy Manual Data Entry Tool

Add a mix with its tracklist in one command.
Usage:
    python scripts/add_mix_with_tracklist.py

Then paste the YouTube URL and tracklist when prompted.
"""
import sys
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_tracklist(text: str) -> list:
    """Parse tracklist text into timestamp entries."""
    timestamps = []
    
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Pattern: timestamp followed by track info
        # Matches: "00:00 Track", "1:23:45 - Track", "00:00 - Artist - Track"
        patterns = [
            r'^[#\d\.\)]*\s*\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*[-–—:]?\s*(.+)$',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                time_str = match.group(1)
                track = match.group(2).strip()
                
                if len(track) >= 3:
                    timestamps.append({
                        'time_str': time_str,
                        'track': track
                    })
                break
    
    return timestamps


def download_audio(url: str, output_dir: Path, name: str) -> str:
    """Download audio from YouTube."""
    output_path = output_dir / f"{name}.wav"
    
    if output_path.exists():
        print(f"  Audio already exists: {output_path.name}")
        return str(output_path)
    
    cmd = [
        'yt-dlp',
        '-x',
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '-o', str(output_path).replace('.wav', '.%(ext)s'),
        '--no-playlist',
        url
    ]
    
    print("  Downloading audio...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if output_path.exists():
        return str(output_path)
    
    # Try to find the file
    for f in output_dir.glob(f"{name}*"):
        if f.suffix == '.wav':
            return str(f)
    
    return None


def run_analysis(audio_path: str, transitions_path: Path, output_path: Path):
    """Run deep transition analysis."""
    cmd = [
        sys.executable,
        'scripts/deep_analyze_transitions.py',
        audio_path,
        '--transitions', str(transitions_path),
        '--output', str(output_path),
        '--context', '15'
    ]
    
    print("  Running deep analysis...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Count transitions
        if output_path.exists():
            with open(output_path) as f:
                data = json.load(f)
            return len(data)
    else:
        print(f"  Error: {result.stderr[:200]}")
    
    return 0


def get_video_info(url: str) -> dict:
    """Get video metadata."""
    cmd = ['yt-dlp', '--dump-json', '--no-download', url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        return json.loads(result.stdout)
    return {}


def main():
    print("="*60)
    print("ADD MIX WITH TRACKLIST")
    print("="*60)
    
    # Get URL
    print("\n1. Enter YouTube URL:")
    url = input("   > ").strip()
    
    if not url:
        print("No URL provided. Exiting.")
        return
    
    # Get video info
    print("\n2. Fetching video info...")
    info = get_video_info(url)
    title = info.get('title', 'Unknown')
    duration = info.get('duration', 0)
    
    print(f"   Title: {title}")
    print(f"   Duration: {duration//60}:{duration%60:02d}")
    
    # Get tracklist
    print("\n3. Paste the tracklist (press Enter twice when done):")
    print("   Format: 00:00 Track Name")
    print("   Example:")
    print("   0:00 Artist - Track One")
    print("   3:45 Another Artist - Track Two")
    print()
    
    lines = []
    empty_count = 0
    
    while True:
        try:
            line = input()
            if line.strip() == '':
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
                lines.append(line)
        except EOFError:
            break
    
    tracklist_text = '\n'.join(lines)
    
    # Parse tracklist
    timestamps = parse_tracklist(tracklist_text)
    
    if not timestamps:
        print("\nNo valid timestamps found. Please check format.")
        return
    
    print(f"\n   Found {len(timestamps)} tracks:")
    for ts in timestamps[:5]:
        print(f"     {ts['time_str']} - {ts['track'][:50]}")
    if len(timestamps) > 5:
        print(f"     ... and {len(timestamps)-5} more")
    
    # Confirm
    print("\n4. Proceed with analysis? (y/n)")
    confirm = input("   > ").strip().lower()
    
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Setup paths
    output_dir = Path("data/youtube_mixes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean title for filename
    safe_title = re.sub(r'[^\w\s-]', '', title)[:60].strip().replace(' ', '_')
    
    # Save tracklist
    transitions_path = output_dir / f"{safe_title}_transitions.txt"
    with open(transitions_path, 'w') as f:
        for ts in timestamps:
            f.write(f"{ts['time_str']} {ts['track']}\n")
    print(f"\n5. Saved tracklist: {transitions_path.name}")
    
    # Download audio
    print("\n6. Downloading audio...")
    audio_path = download_audio(url, output_dir, safe_title)
    
    if not audio_path:
        print("   Failed to download audio.")
        return
    
    print(f"   Downloaded: {Path(audio_path).name}")
    
    # Run analysis
    print("\n7. Analyzing transitions...")
    analysis_path = output_dir / f"{safe_title}_analysis.json"
    transitions_analyzed = run_analysis(audio_path, transitions_path, analysis_path)
    
    if transitions_analyzed > 0:
        print(f"   ✓ Analyzed {transitions_analyzed} transitions")
    else:
        print("   ✗ Analysis failed")
        return
    
    # Save metadata
    result = {
        'url': url,
        'title': title,
        'duration': duration,
        'tracks_found': len(timestamps),
        'transitions_analyzed': transitions_analyzed,
        'audio_path': audio_path,
        'transitions_path': str(transitions_path),
        'analysis_path': str(analysis_path),
        'added_at': datetime.now().isoformat()
    }
    
    result_path = output_dir / f"{safe_title}_result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Update total count
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"\nAdded {transitions_analyzed} new transitions")
    
    # Count total transitions
    total = 0
    for f in Path("data").rglob("*_analysis.json"):
        try:
            with open(f) as file:
                data = json.load(file)
                if isinstance(data, list):
                    total += len(data)
        except:
            pass
    
    print(f"Total transitions in dataset: {total}")
    print(f"Target for NN training: 500")
    print(f"Target for LSTM training: 2000")
    print(f"Progress: {total}/2000 ({total/2000*100:.1f}%)")


if __name__ == '__main__':
    main()

