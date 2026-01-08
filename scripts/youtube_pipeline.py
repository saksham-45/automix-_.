#!/usr/bin/env python3
"""
Automated YouTube DJ Mix Analyzer Pipeline

Fully automated:
1. Scrape video metadata & timestamps from YouTube
2. Download audio as WAV
3. Extract timestamps from description/chapters/comments
4. Run deep transition analysis
5. Generate training data

Usage:
    # Single video:
    python scripts/youtube_pipeline.py "https://youtube.com/watch?v=VIDEO_ID"
    
    # With trim (remove intro):
    python scripts/youtube_pipeline.py "https://youtube.com/watch?v=VIDEO_ID" --trim 30
    
    # Scrape channel (get list of videos):
    python scripts/youtube_pipeline.py --channel "https://youtube.com/@boilerroom" --list
    
    # Process multiple from channel:
    python scripts/youtube_pipeline.py --channel "https://youtube.com/@boilerroom" --max 5
"""
import re
import json
import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class YouTubePipeline:
    """
    Fully automated pipeline for processing YouTube DJ mixes.
    """
    
    def __init__(self, output_dir: str = "data/youtube_mixes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Patterns to identify DJ mixes vs other content
        self.mix_keywords = [
            'dj set', 'boiler room', 'mix', 'live set', 'b2b',
            'essential mix', 'fabric', 'dekmantel', 'resident advisor'
        ]
    
    def get_channel_videos(self, channel_url: str, max_videos: int = 50, 
                           sort_by: str = 'popular') -> List[Dict]:
        """
        Get list of videos from a YouTube channel.
        
        Args:
            channel_url: YouTube channel URL
            max_videos: Maximum number of videos to fetch
            sort_by: 'popular' for most viewed, 'recent' for newest
        """
        # For popular videos, use the channel's popular/videos?sort=p URL
        if sort_by == 'popular':
            url = f'{channel_url}/videos?view=0&sort=p'
            print(f"Fetching MOST POPULAR videos from channel...")
        else:
            url = f'{channel_url}/videos'
            print(f"Fetching recent videos from channel...")
        
        cmd = [
            'yt-dlp',
            '--flat-playlist',
            '--dump-json',
            '--playlist-end', str(max_videos * 2),  # Fetch more to filter later
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        videos = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    videos.append({
                        'id': data.get('id'),
                        'title': data.get('title', ''),
                        'url': f"https://www.youtube.com/watch?v={data.get('id')}",
                        'duration': data.get('duration'),
                        'view_count': data.get('view_count', 0)
                    })
                except json.JSONDecodeError:
                    continue
        
        return videos[:max_videos]
    
    def filter_dj_mixes(self, videos: List[Dict], min_duration: int = 1800) -> List[Dict]:
        """
        Filter videos to only include likely DJ mixes.
        
        Args:
            videos: List of video dicts
            min_duration: Minimum duration in seconds (default 30 min)
        """
        mixes = []
        
        for video in videos:
            title_lower = video['title'].lower()
            
            # Check duration (DJ sets are usually 30min+)
            duration = video.get('duration') or 0
            if duration < min_duration:
                continue
            
            # Check for mix keywords
            is_mix = any(kw in title_lower for kw in self.mix_keywords)
            
            # Boiler Room videos are almost always DJ sets
            if 'boiler room' in title_lower or is_mix:
                mixes.append(video)
        
        return mixes
    
    def get_video_metadata(self, url: str, fetch_comments: bool = True) -> Optional[Dict]:
        """Get full video metadata including description and comments."""
        # First get basic metadata
        cmd = ['yt-dlp', '--dump-json', '--no-download', url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return None
        
        metadata = json.loads(result.stdout)
        
        # Fetch comments if requested
        if fetch_comments:
            print("  Fetching comments (this may take a moment)...")
            comments = self._fetch_comments(url)
            metadata['_comments'] = comments
            print(f"  Fetched {len(comments)} comments")
        
        return metadata
    
    def _fetch_comments(self, url: str, max_comments: int = 100) -> List[str]:
        """Fetch top comments from a video using yt-dlp."""
        cmd = [
            'yt-dlp',
            '--skip-download',
            '--write-comments',
            '--extractor-args', f'youtube:max_comments=all,{max_comments},all,0',
            '--dump-json',
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        comments = []
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                if 'comments' in data:
                    for comment in data['comments']:
                        text = comment.get('text', '')
                        if text:
                            comments.append(text)
            except json.JSONDecodeError:
                pass
        
        return comments
    
    def extract_timestamps(self, metadata: Dict) -> Tuple[List[Dict], str]:
        """
        Extract timestamps from video metadata.
        
        Returns:
            (timestamps_list, source) where source is 'chapters', 'description', 'comments', or 'none'
        """
        # Priority 1: Chapter markers (highest quality)
        if metadata.get('chapters'):
            timestamps = []
            for ch in metadata['chapters']:
                timestamps.append({
                    'time_sec': int(ch['start_time']),
                    'time_str': self._sec_to_str(int(ch['start_time'])),
                    'track': ch['title'].strip()
                })
            return timestamps, 'chapters'
        
        # Priority 2: Description
        if metadata.get('description'):
            timestamps = self._parse_description(metadata['description'])
            if timestamps and len(timestamps) >= 3:
                return timestamps, 'description'
        
        # Priority 3: Comments - ANY timestamp = potential transition
        if metadata.get('_comments'):
            timestamps, comment_source = self._find_tracklist_in_comments(metadata['_comments'])
            if timestamps:
                return timestamps, comment_source
        
        return [], 'none'
    
    def _find_tracklist_in_comments(self, comments: List[str]) -> Tuple[List[Dict], str]:
        """
        Extract ALL timestamps from comments as potential transition points.
        
        Any timestamp where someone commented = something interesting happened
        = likely a transition, drop, or notable moment worth analyzing.
        """
        all_timestamps = {}  # time_sec -> description
        
        for comment in comments:
            # Find all timestamps in this comment
            # Pattern: timestamp optionally followed by text
            matches = re.findall(r'(\d{1,2}:\d{2}(?::\d{2})?)\s*(.*?)(?:\n|$)', comment)
            
            for time_str, description in matches:
                time_sec = self._str_to_sec(time_str)
                description = description.strip()[:100]  # Limit length
                
                # Keep the description (or use "transition" if empty)
                if time_sec not in all_timestamps:
                    all_timestamps[time_sec] = description if description else "transition"
                elif len(description) > len(all_timestamps[time_sec]):
                    all_timestamps[time_sec] = description
        
        if not all_timestamps:
            return [], 'none'
        
        # Convert to list and sort
        timestamps = [
            {
                'time_sec': time_sec,
                'time_str': self._sec_to_str(time_sec),
                'track': desc
            }
            for time_sec, desc in sorted(all_timestamps.items())
        ]
        
        # Filter timestamps that are too close together (within 30 sec)
        filtered = []
        last_time = -60
        for ts in timestamps:
            if ts['time_sec'] - last_time >= 30:
                filtered.append(ts)
                last_time = ts['time_sec']
        
        return filtered, 'comments'
    
    def _parse_description(self, text: str) -> List[Dict]:
        """Parse timestamps from video description."""
        patterns = [
            # Standard: "00:00 Track Name" or "1:23:45 Track Name"
            r'^(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—:]?\s*(.+?)$',
            # With brackets: "[00:00] Track Name"
            r'^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+?)$',
            # Numbered: "1. 00:00 Track Name"
            r'^\d+[\.\)]\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—:]?\s*(.+?)$',
        ]
        
        timestamps = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    time_str = match.group(1)
                    track = match.group(2).strip()
                    
                    # Skip very short track names (probably not a track)
                    if len(track) < 3:
                        continue
                    
                    # Skip common non-track lines
                    skip_words = ['subscribe', 'follow', 'http', 'www.', '@', '#']
                    if any(sw in track.lower() for sw in skip_words):
                        continue
                    
                    timestamps.append({
                        'time_sec': self._str_to_sec(time_str),
                        'time_str': time_str,
                        'track': track
                    })
                    break
        
        # Sort by time
        timestamps.sort(key=lambda x: x['time_sec'])
        
        # Validate: timestamps should be reasonably spaced
        if len(timestamps) >= 2:
            # Check if timestamps make sense (each at least 30 sec apart)
            valid = all(
                timestamps[i+1]['time_sec'] - timestamps[i]['time_sec'] >= 30
                for i in range(len(timestamps)-1)
            )
            if not valid:
                return []
        
        return timestamps
    
    def _str_to_sec(self, ts: str) -> int:
        """Convert "MM:SS" or "H:MM:SS" to seconds."""
        parts = [int(p) for p in ts.split(':')]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    
    def _sec_to_str(self, sec: int) -> str:
        """Convert seconds to timestamp string."""
        h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    
    def download_audio(self, url: str, output_name: str) -> Optional[str]:
        """Download audio as WAV."""
        output_path = self.output_dir / f"{output_name}.wav"
        
        # Skip if already exists
        if output_path.exists():
            print(f"  Audio already exists: {output_path}")
            return str(output_path)
        
        cmd = [
            'yt-dlp',
            '-x',                      # Extract audio
            '--audio-format', 'wav',   # Convert to WAV
            '--audio-quality', '0',    # Best quality
            '-o', str(output_path).replace('.wav', '.%(ext)s'),
            '--no-playlist',
            '--no-warnings',
            url
        ]
        
        print(f"  Downloading audio...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if output_path.exists():
            return str(output_path)
        
        # yt-dlp might add extension differently
        for f in self.output_dir.glob(f"{output_name}*"):
            if f.suffix == '.wav':
                return str(f)
        
        print(f"  Download failed: {result.stderr[:200]}")
        return None
    
    def trim_audio(self, audio_path: str, trim_sec: int) -> str:
        """Trim audio from start using ffmpeg."""
        if trim_sec <= 0:
            return audio_path
        
        input_path = Path(audio_path)
        output_path = input_path.parent / f"{input_path.stem}_trimmed.wav"
        
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(trim_sec),
            '-i', str(input_path),
            '-acodec', 'pcm_s16le',
            str(output_path)
        ]
        
        subprocess.run(cmd, capture_output=True)
        
        if output_path.exists():
            input_path.unlink()  # Remove original
            output_path.rename(input_path)  # Rename trimmed to original name
            return str(input_path)
        
        return audio_path
    
    def adjust_timestamps(self, timestamps: List[Dict], trim_sec: int) -> List[Dict]:
        """Adjust timestamps after trimming."""
        if trim_sec <= 0:
            return timestamps
        
        adjusted = []
        for ts in timestamps:
            new_sec = ts['time_sec'] - trim_sec
            if new_sec >= 0:
                adjusted.append({
                    'time_sec': new_sec,
                    'time_str': self._sec_to_str(new_sec),
                    'track': ts['track']
                })
        return adjusted
    
    def save_transitions_file(self, timestamps: List[Dict], path: Path):
        """Save timestamps in format expected by analyzer."""
        with open(path, 'w') as f:
            for ts in timestamps:
                f.write(f"{ts['time_str']} {ts['track']}\n")
    
    def run_deep_analysis(self, audio_path: str, transitions_path: Path) -> bool:
        """Run the deep transition analyzer."""
        output_path = str(transitions_path).replace('_transitions.txt', '_analysis.json')
        
        cmd = [
            sys.executable,
            'scripts/deep_analyze_transitions.py',
            audio_path,
            '--transitions', str(transitions_path),
            '--output', output_path,
            '--context', '15'
        ]
        
        print(f"  Running deep analysis...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Count transitions analyzed
            if Path(output_path).exists():
                with open(output_path) as f:
                    data = json.load(f)
                print(f"  ✓ Analyzed {len(data)} transitions")
                
                # Clean up audio file after successful analysis
                audio_file = Path(audio_path)
                if audio_file.exists() and audio_file.suffix == '.wav':
                    try:
                        audio_file.unlink()
                        print(f"  ✓ Deleted audio file: {audio_file.name}")
                    except Exception as e:
                        print(f"  ⚠ Could not delete {audio_file.name}: {e}")
                
                return True
        
        print(f"  Analysis error: {result.stderr[:200]}")
        return False
    
    def process_video(self, url: str, trim_start: int = 0) -> Dict:
        """
        Complete processing of a single video.
        
        Returns:
            Result dict with status and paths
        """
        print(f"\n{'='*60}")
        print("Processing video...")
        print(f"{'='*60}")
        
        # Step 1: Get metadata
        print("\n[1/5] Fetching metadata...")
        metadata = self.get_video_metadata(url)
        
        if not metadata:
            return {'status': 'error', 'error': 'Failed to fetch metadata', 'url': url}
        
        title = metadata.get('title', 'unknown')
        safe_title = re.sub(r'[^\w\s-]', '', title)[:60].strip().replace(' ', '_')
        duration = int(metadata.get('duration') or 0)
        
        print(f"  Title: {title}")
        print(f"  Duration: {duration//60}:{duration%60:02d}")
        
        # Step 2: Extract timestamps
        print("\n[2/5] Extracting timestamps...")
        timestamps, source = self.extract_timestamps(metadata)
        
        if not timestamps:
            print(f"  ✗ No timestamps found")
            return {
                'status': 'no_timestamps',
                'url': url,
                'title': title,
            }
        
        print(f"  ✓ Found {len(timestamps)} tracks from {source}")
        for ts in timestamps[:3]:
            print(f"    {ts['time_str']} - {ts['track'][:50]}")
        if len(timestamps) > 3:
            print(f"    ... and {len(timestamps)-3} more")
        
        # Step 3: Download audio
        print("\n[3/5] Downloading audio...")
        audio_path = self.download_audio(url, safe_title)
        
        if not audio_path:
            return {'status': 'error', 'error': 'Download failed', 'url': url, 'title': title}
        
        print(f"  ✓ Downloaded: {Path(audio_path).name}")
        
        # Step 4: Trim if needed
        print(f"\n[4/5] Trimming...")
        if trim_start > 0:
            audio_path = self.trim_audio(audio_path, trim_start)
            timestamps = self.adjust_timestamps(timestamps, trim_start)
            print(f"  ✓ Trimmed {trim_start}s from start")
        else:
            print(f"  No trimming needed")
        
        # Save transitions file
        transitions_path = self.output_dir / f"{safe_title}_transitions.txt"
        self.save_transitions_file(timestamps, transitions_path)
        
        # Step 5: Run analysis
        print("\n[5/5] Analyzing transitions...")
        analysis_success = self.run_deep_analysis(audio_path, transitions_path)
        
        result = {
            'status': 'success' if analysis_success else 'analysis_failed',
            'url': url,
            'title': title,
            'duration': duration,
            'tracks_found': len(timestamps),
            'transitions': len(timestamps) - 1,
            'timestamp_source': source,
            'audio_path': audio_path,
            'transitions_path': str(transitions_path),
            'analysis_path': str(transitions_path).replace('_transitions.txt', '_analysis.json'),
            'processed_at': datetime.now().isoformat()
        }
        
        # Save result metadata
        result_path = self.output_dir / f"{safe_title}_result.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n{'='*60}")
        if analysis_success:
            print("✓ SUCCESS!")
        else:
            print("⚠ Partial success (analysis failed)")
        print(f"{'='*60}")
        
        return result
    
    def process_channel(self, channel_url: str, max_videos: int = 10, 
                       trim_start: int = 0, sort_by: str = 'popular') -> List[Dict]:
        """
        Process multiple videos from a channel.
        """
        print(f"\n{'#'*60}")
        print(f"BOILER ROOM CHANNEL SCRAPER")
        print(f"{'#'*60}")
        
        # Get channel videos
        videos = self.get_channel_videos(channel_url, max_videos=max_videos * 2, sort_by=sort_by)
        print(f"\nFetched {len(videos)} videos from channel")
        
        # Filter to DJ mixes
        mixes = self.filter_dj_mixes(videos)
        print(f"Identified {len(mixes)} potential DJ mixes (30+ min)")
        
        # Limit to max
        mixes = mixes[:max_videos]
        
        print(f"\nWill process {len(mixes)} videos:")
        for i, mix in enumerate(mixes, 1):
            dur = int(mix.get('duration') or 0)
            print(f"  {i}. {mix['title'][:50]}... ({dur//60}:{dur%60:02d})")
        
        # Process each
        results = []
        for i, mix in enumerate(mixes, 1):
            print(f"\n\n{'#'*60}")
            print(f"VIDEO {i}/{len(mixes)}")
            print(f"{'#'*60}")
            
            result = self.process_video(mix['url'], trim_start=trim_start)
            results.append(result)
            
            # Summary
            if result['status'] == 'success':
                print(f"  → {result['transitions']} transitions extracted")
        
        # Final summary
        print(f"\n\n{'#'*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'#'*60}")
        
        success = sum(1 for r in results if r['status'] == 'success')
        no_ts = sum(1 for r in results if r['status'] == 'no_timestamps')
        errors = sum(1 for r in results if r['status'] == 'error')
        
        print(f"\nResults:")
        print(f"  ✓ Successful: {success}")
        print(f"  ⚠ No timestamps: {no_ts}")
        print(f"  ✗ Errors: {errors}")
        
        total_transitions = sum(r.get('transitions', 0) for r in results if r['status'] == 'success')
        print(f"\nTotal transitions extracted: {total_transitions}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Automated YouTube DJ Mix Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single video:
    python scripts/youtube_pipeline.py "https://youtube.com/watch?v=VIDEO_ID"
    
    # List videos from Boiler Room channel:
    python scripts/youtube_pipeline.py --channel "https://youtube.com/@boilerroom" --list
    
    # Process 5 videos from channel:
    python scripts/youtube_pipeline.py --channel "https://youtube.com/@boilerroom" --max 5
    
    # Process with intro trim:
    python scripts/youtube_pipeline.py "https://youtube.com/watch?v=VIDEO_ID" --trim 30
        """
    )
    
    parser.add_argument('url', nargs='?', help='YouTube video URL')
    parser.add_argument('--channel', '-c', help='YouTube channel URL')
    parser.add_argument('--list', '-l', action='store_true', help='List videos only (no processing)')
    parser.add_argument('--max', '-m', type=int, default=5, help='Max videos to process from channel')
    parser.add_argument('--popular', '-p', action='store_true', help='Sort by popularity (most viewed)')
    parser.add_argument('--trim', '-t', type=int, default=0, help='Seconds to trim from start')
    parser.add_argument('--output', '-o', default='data/youtube_mixes', help='Output directory')
    
    args = parser.parse_args()
    
    pipeline = YouTubePipeline(output_dir=args.output)
    
    if args.channel:
        sort_by = 'popular' if args.popular else 'recent'
        
        if args.list:
            # Just list videos
            videos = pipeline.get_channel_videos(args.channel, max_videos=args.max, sort_by=sort_by)
            mixes = pipeline.filter_dj_mixes(videos)
            
            print(f"\nDJ Mixes from channel ({len(mixes)} found):\n")
            for i, mix in enumerate(mixes, 1):
                dur = int(mix.get('duration') or 0)
                print(f"{i:2}. [{dur//60}:{dur%60:02d}] {mix['title']}")
                print(f"     {mix['url']}")
        else:
            # Process channel
            pipeline.process_channel(args.channel, max_videos=args.max, trim_start=args.trim, sort_by=sort_by)
    
    elif args.url:
        # Process single video
        pipeline.process_video(args.url, trim_start=args.trim)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

