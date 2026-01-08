#!/usr/bin/env python3
"""
Process PLAYGRND SERIES videos with tracklists in descriptions.
"""
import sys
import json
import re
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.youtube_pipeline import YouTubePipeline


def extract_tracklist_from_description(description: str):
    """Extract tracklist from video description."""
    lines = description.split('\n')
    tracklist = []
    
    for line in lines:
        line = line.strip()
        # Look for timestamp patterns: MM:SS or [MM:SS] or H:MM:SS
        # Pattern: optional bracket, 1-2 digits : 2 digits (optional : 2 digits), optional bracket, dash, track name
        match = re.match(r'\[?(\d{1,2}):(\d{2})(?::(\d{2}))?\]?\s*[-–—]?\s*(.+)', line)
        if match:
            first_num = int(match.group(1))
            second_num = int(match.group(2))
            third_num = int(match.group(3)) if match.group(3) else None
            
            if third_num is not None:
                # H:MM:SS format
                hours = first_num
                mins = second_num
                secs = third_num
                total_mins = hours * 60 + mins
                total_secs = secs
            else:
                # MM:SS format (most common)
                total_mins = first_num
                total_secs = second_num
            
            track_name = match.group(4).strip()
            if track_name and len(track_name) > 2:
                tracklist.append(f"{total_mins}:{total_secs:02d} {track_name}")
    
    return tracklist


def process_video(video_info: dict, pipeline: YouTubePipeline):
    """Process a single video."""
    vid_id = video_info['id']
    url = f"https://www.youtube.com/watch?v={vid_id}"
    title = video_info['title']
    safe_title = video_info['safe_title']
    
    print(f"\n{'='*60}")
    print(f"Processing: {title[:60]}")
    print(f"{'='*60}")
    print(f"URL: {url}")
    print(f"Expected tracks: {video_info['track_count']}")
    
    # Extract tracklist from description
    description = video_info['description']
    tracklist = extract_tracklist_from_description(description)
    
    if len(tracklist) < 15:
        print(f"  ✗ Only {len(tracklist)} tracks extracted, skipping")
        return None
    
    print(f"  ✓ Extracted {len(tracklist)} tracks")
    
    # Save tracklist file
    transitions_file = Path(f"data/youtube_mixes/{safe_title}_transitions.txt")
    transitions_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(transitions_file, 'w') as f:
        for track in tracklist:
            f.write(track + '\n')
    
    print(f"  ✓ Saved tracklist: {transitions_file.name}")
    
    # Download audio
    print(f"  Downloading audio...")
    audio_path = pipeline.download_audio(url, safe_title)
    
    if not audio_path:
        print(f"  ✗ Download failed")
        return None
    
    print(f"  ✓ Downloaded: {Path(audio_path).name}")
    
    # Run analysis
    print(f"  Running analysis...")
    analysis_success = pipeline.run_deep_analysis(audio_path, transitions_file)
    
    if analysis_success:
        # Count transitions
        analysis_file = transitions_file.parent / f"{safe_title}_analysis.json"
        if analysis_file.exists():
            with open(analysis_file) as f:
                data = json.load(f)
            count = len(data) if isinstance(data, list) else 0
            print(f"  ✓✓✓ SUCCESS! Analyzed {count} transitions")
            return {'transitions_analyzed': count, 'status': 'success'}
    
    print(f"  ✗ Analysis failed")
    return None


def main():
    # Load videos to process
    with open('data/youtube_mixes/playgrnd_to_process.json') as f:
        videos = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {len(videos)} VIDEOS FROM PLAYGRND SERIES")
    print(f"{'='*60}\n")
    
    pipeline = YouTubePipeline()
    results = []
    
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}]")
        result = process_video(video, pipeline)
        
        if result:
            results.append({
                'video': video['title'],
                'url': f"https://www.youtube.com/watch?v={video['id']}",
                'transitions': result.get('transitions_analyzed', 0)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(results)}/{len(videos)}")
    
    total_new = sum(r['transitions'] for r in results)
    print(f"Total new transitions: {total_new}")
    
    # Save results
    with open('data/youtube_mixes/playgrnd_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()

