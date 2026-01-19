#!/usr/bin/env python3
"""Simple script to play the mix in real-time via HTTP streaming."""
import urllib.request
import wave
import io
import sys

mix_id = '26eb2541'
API_BASE = 'http://127.0.0.1:5001/api'

def play_chunk(chunk_idx):
    """Download and save a chunk to play."""
    url = f'{API_BASE}/stream/{mix_id}/audio?chunk={chunk_idx}'
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = response.read()
            if len(data) > 0:
                # Save to file
                filename = f'chunk_{chunk_idx}.wav'
                with open(filename, 'wb') as f:
                    f.write(data)
                
                # Get duration
                wav_file = wave.open(io.BytesIO(data), 'rb')
                duration = wav_file.getnframes() / wav_file.getframerate()
                wav_file.close()
                
                print(f"✅ Chunk {chunk_idx}: Saved ({duration:.1f}s) - Play: open {filename}")
                return True
            else:
                print(f"⏳ Chunk {chunk_idx}: Not ready yet")
                return False
    except Exception as e:
        print(f"❌ Chunk {chunk_idx}: {e}")
        return False

if __name__ == '__main__':
    chunk_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"Downloading chunk {chunk_idx}...")
    if play_chunk(chunk_idx):
        print(f"\n🎵 To play: open chunk_{chunk_idx}.wav")
