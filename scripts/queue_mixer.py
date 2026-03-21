#!/usr/bin/env python3
"""
Queue-based continuous mixer CLI.
Simple interface for managing song queues and creating continuous mixes.
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.queue_manager import QueueManager
from scripts.continuous_mixer import ContinuousMixer


def cmd_add(args):
    """Add song to queue."""
    queue_manager = QueueManager(args.queue)
    
    if args.local:
        source = "local"
    else:
        source = "youtube"
    
    song_id = queue_manager.add_song(args.url, source=source)
    print(f"✓ Added song {song_id} to queue: {args.url}")


def cmd_list(args):
    """List songs in queue."""
    queue_manager = QueueManager(args.queue)
    songs = queue_manager.list_queue()
    
    if not songs:
        print("Queue is empty")
        return
    
    print(f"\nQueue ({len(songs)} songs):")
    print("-" * 60)
    for song in songs:
        status_icon = {
            "queued": "○",
            "downloading": "↓",
            "analyzing": "⚙",
            "mixed": "✓"
        }.get(song['status'], "?")
        
        print(f"  {status_icon} [{song['id']}] {song['source']}: {song['url']}")
        if song.get('metadata'):
            metadata = song['metadata']
            if 'title' in metadata:
                print(f"      Title: {metadata['title']}")


def cmd_remove(args):
    """Remove song from queue."""
    queue_manager = QueueManager(args.queue)
    
    if queue_manager.remove_song(args.song_id):
        print(f"✓ Removed song {args.song_id} from queue")
    else:
        print(f"✗ Song {args.song_id} not found in queue")


def cmd_clear(args):
    """Clear entire queue."""
    queue_manager = QueueManager(args.queue)
    queue_manager.clear_queue()
    print("✓ Queue cleared")


def cmd_mix(args):
    """Create continuous mix from queue."""
    mixer = ContinuousMixer(
        cache_dir=args.cache_dir,
        db_path=args.db_path,
        temp_dir=args.temp_dir
    )
    
    if args.streaming:
        # Use streaming mode (segment-based downloads)
        output_path = mixer.process_queue_streaming(
            queue_path=args.queue,
            output_path=args.output,
            segment_duration=args.duration,
            transition_duration=args.transition_duration,
            buffer_seconds=args.buffer_seconds
        )
    else:
        # Use traditional mode (full downloads)
        output_path = mixer.process_queue(
            queue_path=args.queue,
            output_path=args.output,
            segment_duration=args.duration,
            transition_duration=args.transition_duration,
            max_workers=args.workers
        )
    
    print(f"\n✓ Mix saved to: {output_path}")


def cmd_add_playlist(args):
    """Add songs from YouTube playlist to queue."""
    try:
        import yt_dlp
    except ImportError:
        print("✗ yt-dlp not installed. Install with: pip install yt-dlp")
        return
    
    print(f"Fetching playlist: {args.playlist_url}")
    
    # Extract playlist info
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'skip_download': True,
    }
    
    queue_manager = QueueManager(args.queue)
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(args.playlist_url, download=False)
            
            if 'entries' not in info:
                print("✗ Not a valid playlist")
                return
            
            added = 0
            for entry in info['entries']:
                if entry is None:
                    continue
                
                video_id = entry.get('id')
                if not video_id:
                    continue
                
                url = f"https://youtu.be/{video_id}"
                song_id = queue_manager.add_song(url, source="youtube")
                added += 1
            
            print(f"✓ Added {added} songs from playlist")
            
        except Exception as e:
            print(f"✗ Error fetching playlist: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Queue-based continuous mixer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add songs to queue
  python scripts/queue_mixer.py add "https://youtu.be/..."
  python scripts/queue_mixer.py add "song.mp3" --local
  
  # List queue
  python scripts/queue_mixer.py list
  
  # Remove song
  python scripts/queue_mixer.py remove 2
  
  # Clear queue
  python scripts/queue_mixer.py clear
  
  # Create mix
  python scripts/queue_mixer.py mix
  
  # Add from playlist
  python scripts/queue_mixer.py add-playlist "https://youtube.com/playlist?list=..."
        """
    )
    
    parser.add_argument('--queue', type=str, default='data/queue.json',
                       help='Queue file path (default: data/queue.json)')
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Add command
    parser_add = subparsers.add_parser('add', help='Add song to queue')
    parser_add.add_argument('url', help='YouTube URL or local file path')
    parser_add.add_argument('--local', action='store_true',
                           help='Treat URL as local file path')
    parser_add.set_defaults(func=cmd_add)
    
    # List command
    parser_list = subparsers.add_parser('list', help='List songs in queue')
    parser_list.set_defaults(func=cmd_list)
    
    # Remove command
    parser_remove = subparsers.add_parser('remove', help='Remove song from queue')
    parser_remove.add_argument('song_id', type=int, help='Song ID to remove')
    parser_remove.set_defaults(func=cmd_remove)
    
    # Clear command
    parser_clear = subparsers.add_parser('clear', help='Clear entire queue')
    parser_clear.set_defaults(func=cmd_clear)
    
    # Mix command
    parser_mix = subparsers.add_parser('mix', help='Create continuous mix from queue')
    parser_mix.add_argument('--output', type=str, default=None,
                           help='Output file path (default: auto-generate)')
    parser_mix.add_argument('--duration', type=int, default=60,
                           help='Segment duration per song (seconds, default: 60)')
    parser_mix.add_argument('--transition-duration', type=float, default=30.0,
                           help='Transition duration (seconds, default: 30.0)')
    parser_mix.add_argument('--streaming', action='store_true',
                           help='Use streaming mode (segment-based downloads, faster)')
    parser_mix.add_argument('--buffer-seconds', type=int, default=60,
                           help='Buffer size for streaming (default: 60, future use)')
    parser_mix.add_argument('--workers', type=int, default=4,
                           help='Parallel download workers (default: 4, traditional mode only)')
    parser_mix.add_argument('--cache-dir', type=str, default='data/cache',
                           help='Cache directory (default: data/cache)')
    parser_mix.add_argument('--db-path', type=str, default='data/music_analysis.db',
                           help='Database path (default: data/music_analysis.db)')
    parser_mix.add_argument('--temp-dir', type=str, default='temp_audio',
                           help='Temporary audio directory (default: temp_audio)')
    parser_mix.set_defaults(func=cmd_mix)
    
    # Add playlist command
    parser_playlist = subparsers.add_parser('add-playlist', 
                                           help='Add songs from YouTube playlist')
    parser_playlist.add_argument('playlist_url', help='YouTube playlist URL')
    parser_playlist.set_defaults(func=cmd_add_playlist)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
