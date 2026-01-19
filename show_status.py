#!/usr/bin/env python3
"""Show real-time status of a mix."""
import urllib.request
import json
import sys
import time

mix_id = sys.argv[1] if len(sys.argv) > 1 else '26eb2541'

def show_status():
    try:
        url = f'http://127.0.0.1:5001/api/mix/{mix_id}'
        with urllib.request.urlopen(url, timeout=5) as response:
            status = json.loads(response.read().decode())
            
            print("\r" + " " * 80 + "\r", end="")  # Clear line
            songs = status.get('songs_ready', 0)
            total_songs = status.get('total_songs_in_playlist', 0) or status.get('total_songs', 0)
            chunks = status.get('chunks_ready', 0)
            total_chunks = status.get('total_chunks', 0)
            processing = status.get('processing_status', 'N/A')
            
            bar_width = 40
            if total_songs > 0:
                filled = int(bar_width * songs / total_songs)
                bar = '█' * filled + '░' * (bar_width - filled)
                pct = (songs / total_songs * 100) if total_songs > 0 else 0
            else:
                bar = '░' * bar_width
                pct = 0
            
            print(f"[{bar}] {songs}/{total_songs} songs | {chunks}/{total_chunks} chunks | {processing[:20]}", end="", flush=True)
            
            return status.get('songs_ready', 0) > 0
    except:
        print("\r❌ Server not responding", end="", flush=True)
        return False

if __name__ == '__main__':
    print(f"Monitoring mix {mix_id} (Ctrl+C to stop)...")
    try:
        while True:
            show_status()
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n✅ Done")
