#!/usr/bin/env python3
"""
Test script for monitoring and testing the full 43-song playlist mix.
"""
import requests
import time
import sys

API_BASE = 'http://127.0.0.1:5001/api'
MIX_ID = '26eb2541'

def get_status():
    """Get current mix status."""
    try:
        resp = requests.get(f'{API_BASE}/mix/{MIX_ID}', timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        print(f"Error getting status: {e}")
        return None

def test_chunk(chunk_index):
    """Test if a chunk is available."""
    try:
        resp = requests.get(
            f'{API_BASE}/stream/{MIX_ID}/audio?chunk={chunk_index}',
            timeout=5
        )
        if resp.status_code == 200:
            return len(resp.content)
        return 0
    except:
        return 0

def main():
    print("="*70)
    print("FULL PLAYLIST MIX TEST - 43 SONGS")
    print("="*70)
    print()
    print(f"Mix ID: {MIX_ID}")
    print("Monitoring progress...")
    print()
    
    max_wait = 300  # 5 minutes
    start_time = time.time()
    last_chunks_ready = -1
    consecutive_no_change = 0
    
    while time.time() - start_time < max_wait:
        status = get_status()
        
        if not status:
            print("⚠ Could not get status. Is the server running?")
            time.sleep(5)
            continue
        
        # Extract status info
        processing_status = status.get('processing_status', 'unknown')
        first_batch_ready = status.get('first_batch_ready', False)
        chunks_ready = status.get('chunks_ready', 0)
        total_songs = status.get('total_songs', 0)
        total_transitions_needed = status.get('total_transitions_needed', 0)
        
        # Try to get total_chunks - may not be in old code
        total_chunks = status.get('total_chunks')
        if total_chunks is None:
            # Calculate it: 2*N - 1 for N songs
            total_chunks = 2 * total_songs - 1 if total_songs > 0 else 0
        
        songs_ready = status.get('songs_ready', status.get('total_songs', 0))
        transitions_ready = status.get('transitions_ready', 0)
        
        elapsed = int(time.time() - start_time)
        
        # Progress calculation
        progress_pct = (chunks_ready / total_chunks * 100) if total_chunks > 0 else 0
        
        # Status line
        status_line = (
            f"[{elapsed:3d}s] "
            f"Ready: {chunks_ready}/{total_chunks} chunks ({progress_pct:.1f}%) | "
            f"Songs: {songs_ready}/{total_songs} | "
            f"Transitions: {transitions_ready}/{total_transitions_needed} | "
            f"Status: {processing_status}"
        )
        
        # Check if we have progress
        if chunks_ready != last_chunks_ready:
            print(status_line)
            last_chunks_ready = chunks_ready
            consecutive_no_change = 0
            
            # Test first few chunks as they become available
            if chunks_ready > 0:
                for chunk_idx in [0, 1, 2]:
                    if chunk_idx < chunks_ready:
                        size = test_chunk(chunk_idx)
                        if size > 0:
                            print(f"    ✓ Chunk {chunk_idx}: {size} bytes available")
        else:
            consecutive_no_change += 1
            if consecutive_no_change % 10 == 0:  # Print every 20 seconds if no change
                print(status_line + " (waiting...)")
        
        # Check if first batch is ready
        if first_batch_ready and chunks_ready >= 5:
            print()
            print("="*70)
            print("✓ FIRST BATCH READY! Playback can start now!")
            print("="*70)
            print()
            print("Stream URL:")
            print(f"  {API_BASE}/stream/{MIX_ID}/audio?chunk=0")
            print()
            print("Web Interface:")
            print(f"  http://127.0.0.1:5001/")
            print()
        
        # Check if complete
        if processing_status == 'complete':
            print()
            print("="*70)
            print("✓✓✓ ALL CONTENT PROCESSED! ✓✓✓")
            print("="*70)
            print()
            print(f"Total chunks ready: {chunks_ready}/{total_chunks}")
            print(f"Songs: {songs_ready}/{total_songs}")
            print(f"Transitions: {transitions_ready}/{total_transitions_needed}")
            print()
            print("Expected stream length: ~3-4 hours (43 full songs + 42 transitions)")
            print()
            print("You can now play the complete mix from:")
            print(f"  {API_BASE}/stream/{MIX_ID}/audio?chunk=0")
            print()
            break
        
        time.sleep(2)
    
    # Final status
    print()
    print("="*70)
    print("FINAL STATUS")
    print("="*70)
    status = get_status()
    if status:
        print(f"Processing status: {status.get('processing_status')}")
        print(f"Chunks ready: {status.get('chunks_ready', 0)}")
        print(f"Total songs: {status.get('total_songs', 0)}")
        print()
        print("The mix is processing in the background.")
        print("You can start playing now - it will stream chunks as they become available!")
        print()
        print(f"Web Interface: http://127.0.0.1:5001/")
        print(f"Stream URL: {API_BASE}/stream/{MIX_ID}/audio?chunk=0")

if __name__ == '__main__':
    main()
