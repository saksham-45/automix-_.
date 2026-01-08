#!/usr/bin/env python3
"""
Automatically find and process videos with tracklists from a YouTube channel.

Usage:
    python scripts/auto_find_tracklists.py "https://www.youtube.com/@channel/videos" --limit 10
    python scripts/auto_find_tracklists.py "https://www.youtube.com/watch?v=VIDEO_ID"
"""
import sys
import argparse
import subprocess
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.youtube_pipeline import AutomatedMixAnalyzer


def find_videos_with_tracklists(channel_url: str, limit: int = 10, min_tracks: int = 15):
    """
    Find videos with tracklists in comments.
    
    Args:
        channel_url: YouTube channel URL or single video URL
        limit: Max number of videos to check
        min_tracks: Minimum number of timestamps to consider valid tracklist
    """
    # Check if it's a single video or channel
    if 'watch?v=' in channel_url:
        video_ids = [channel_url.split('watch?v=')[1].split('&')[0]]
    else:
        # Get channel videos
        cmd = f'yt-dlp --flat-playlist --print "%(id)s %(title)s" "{channel_url}" 2>/dev/null | head -{limit}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        video_ids = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    video_ids.append(parts[0])
    
    print(f"\n{'='*60}")
    print(f"SEARCHING FOR VIDEOS WITH TRACKLISTS")
    print(f"{'='*60}")
    print(f"Checking {len(video_ids)} videos...\n")
    
    found = []
    
    for i, vid_id in enumerate(video_ids, 1):
        url = f"https://www.youtube.com/watch?v={vid_id}"
        print(f"[{i}/{len(video_ids)}] Checking video...")
        
        # Get video info and comments
        cmd = f'yt-dlp --write-comments --skip-download --no-warnings -o "temp_find" "{url}" 2>/dev/null'
        subprocess.run(cmd, shell=True, capture_output=True)
        
        try:
            with open('temp_find.info.json') as f:
                data = json.load(f)
            
            title = data.get('title', 'Unknown')
            duration = data.get('duration', 0)
            comments = data.get('comments', [])
            
            print(f"  Title: {title[:60]}")
            print(f"  Comments: {len(comments)}")
            
            # Find best tracklist
            best_count = 0
            best_tracklist = None
            best_likes = 0
            
            for c in sorted(comments, key=lambda x: x.get('like_count', 0), reverse=True)[:50]:
                text = c.get('text', '')
                # Look for tracklist patterns: [MM:SS] or MM:SS formats
                timestamps = re.findall(r'(\[?\d{1,2}:\d{2}(?::\d{2})?\]?)', text)
                if len(timestamps) > best_count:
                    best_count = len(timestamps)
                    best_tracklist = text
                    best_likes = c.get('like_count', 0)
            
            if best_count >= min_tracks:
                print(f"  ✓ FOUND! {best_count} timestamps ({best_likes} likes)")
                found.append({
                    'id': vid_id,
                    'url': url,
                    'title': title,
                    'duration': duration,
                    'track_count': best_count,
                    'tracklist': best_tracklist
                })
            else:
                print(f"  ✗ No tracklist (max {best_count} timestamps)")
                
        except Exception as e:
            print(f"  Error: {e}")
        
        subprocess.run('rm -f temp_find.info.json', shell=True)
    
    return found


def process_found_videos(found_videos: list):
    """Process all found videos with tracklists."""
    if not found_videos:
        print("\nNo videos with tracklists found!")
        return
    
    print(f"\n{'='*60}")
    print(f"FOUND {len(found_videos)} VIDEOS WITH TRACKLISTS")
    print(f"{'='*60}\n")
    
    analyzer = AutomatedMixAnalyzer()
    processed = 0
    
    for i, video in enumerate(found_videos, 1):
        print(f"\n[{i}/{len(found_videos)}] Processing: {video['title'][:50]}...")
        print(f"  URL: {video['url']}")
        print(f"  Tracks: {video['track_count']}")
        
        try:
            result = analyzer.process_url(video['url'])
            if result.get('status') == 'success':
                processed += 1
                print(f"  ✓ Successfully processed!")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"PROCESSED {processed}/{len(found_videos)} VIDEOS")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Automatically find and process videos with tracklists'
    )
    parser.add_argument('url', help='YouTube channel URL or video URL')
    parser.add_argument('--limit', '-l', type=int, default=10,
                       help='Max videos to check (for channels)')
    parser.add_argument('--min-tracks', '-m', type=int, default=15,
                       help='Minimum timestamps to consider valid tracklist')
    parser.add_argument('--process', '-p', action='store_true',
                       help='Automatically process found videos')
    parser.add_argument('--output', '-o', default='found_videos.json',
                       help='Save found videos to JSON file')
    
    args = parser.parse_args()
    
    # Find videos
    found = find_videos_with_tracklists(args.url, args.limit, args.min_tracks)
    
    # Save to file
    with open(args.output, 'w') as f:
        json.dump(found, f, indent=2)
    print(f"\n✓ Saved {len(found)} videos to: {args.output}")
    
    # Process if requested
    if args.process and found:
        process_found_videos(found)
    elif found:
        print(f"\nTo process these videos, run:")
        print(f"  python scripts/auto_find_tracklists.py {args.url} --process")


if __name__ == '__main__':
    main()

