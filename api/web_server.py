#!/usr/bin/env python3
"""
Web API Server for real-time continuous mixer.

ALIGNMENT WITH PROJECT GOALS:
- RESTful API for playlist management
- Real-time audio streaming
- Shuffle and sequential mode support
- Integration with existing quality prediction system
- Production-ready error handling and validation
"""
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename

from api.streaming_mixer import RealTimeStreamingMixer
from src.next_song_selector import NextSongSelector
from src.database import MusicDatabase
from scripts.queue_manager import QueueManager


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = Path('data/playlists')
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

MIXES_FOLDER = Path('data/web_mixes')
MIXES_FOLDER.mkdir(parents=True, exist_ok=True)

# Active mixing sessions
active_mixes: Dict[str, RealTimeStreamingMixer] = {}

# Database connection
db_path = Path('data/music_analysis.db')
db = MusicDatabase(str(db_path)) if db_path.exists() else None


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_mixes': len(active_mixes)
    })


@app.route('/api/playlist/upload', methods=['POST'])
def upload_playlist():
    """
    Upload YouTube playlist URL.
    
    ALIGNMENT: Extracts songs from playlist, queues for analysis.
    
    Request:
        {
            "playlist_url": "https://youtube.com/playlist?list=...",
            "analyze": true/false  # Whether to analyze immediately
        }
    
    Response:
        {
            "playlist_id": "...",
            "songs": [...],
            "status": "queued" or "analyzing"
        }
    """
    try:
        data = request.get_json()
        playlist_url = data.get('playlist_url')
        
        if not playlist_url:
            return jsonify({'error': 'playlist_url required'}), 400
        
        # Extract playlist songs using yt-dlp
        import yt_dlp
        
        playlist_id = str(uuid.uuid4())[:8]
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'skip_download': True,
        }
        
        songs = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(playlist_url, download=False)
                
                if 'entries' not in info:
                    return jsonify({'error': 'Not a valid playlist'}), 400
                
                for i, entry in enumerate(info['entries']):
                    if entry is None:
                        continue
                    
                    video_id = entry.get('id')
                    if not video_id:
                        continue
                    
                    url = f"https://youtu.be/{video_id}"
                    title = entry.get('title', f'Track {i+1}')
                    
                    songs.append({
                        'id': str(uuid.uuid4())[:16],
                        'url': url,
                        'title': title,
                        'source': 'youtube',
                        'status': 'queued',
                        'playlist_id': playlist_id
                    })
                
                # Save playlist
                playlist_file = UPLOAD_FOLDER / f"{playlist_id}.json"
                with open(playlist_file, 'w') as f:
                    json.dump({
                        'playlist_id': playlist_id,
                        'playlist_url': playlist_url,
                        'songs': songs,
                        'created_at': datetime.now().isoformat()
                    }, f, indent=2)
                
                # Queue for analysis if requested
                if data.get('analyze', False):
                    # TODO: Background analysis task
                    pass
                
                return jsonify({
                    'playlist_id': playlist_id,
                    'songs': songs,
                    'status': 'queued',
                    'total_songs': len(songs)
                }), 200
                
            except Exception as e:
                return jsonify({'error': f'Failed to extract playlist: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/playlist/<playlist_id>', methods=['GET'])
def get_playlist(playlist_id: str):
    """Get playlist details."""
    playlist_file = UPLOAD_FOLDER / f"{playlist_id}.json"
    
    if not playlist_file.exists():
        return jsonify({'error': 'Playlist not found'}), 404
    
    with open(playlist_file, 'r') as f:
        playlist = json.load(f)
    
    return jsonify(playlist), 200


@app.route('/api/mix/start', methods=['POST'])
def start_mix():
    """
    Start continuous mix from playlist.
    
    ALIGNMENT: Initializes streaming mixer with shuffle or sequential mode.
    
    Request:
        {
            "playlist_id": "...",
            "mode": "shuffle" or "sequential",
            "segment_duration": 60,
            "transition_duration": 16.0
        }
    
    Response:
        {
            "mix_id": "...",
            "stream_url": "/api/stream/<mix_id>/audio",
            "status": "ready"
        }
    """
    try:
        data = request.get_json()
        playlist_id = data.get('playlist_id')
        mode = data.get('mode', 'sequential')  # 'shuffle' or 'sequential'
        segment_duration = data.get('segment_duration', 60)
        transition_duration = data.get('transition_duration', 16.0)
        
        if not playlist_id:
            return jsonify({'error': 'playlist_id required'}), 400
        
        if mode not in ['shuffle', 'sequential']:
            return jsonify({'error': 'mode must be shuffle or sequential'}), 400
        
        # Load playlist
        playlist_file = UPLOAD_FOLDER / f"{playlist_id}.json"
        if not playlist_file.exists():
            return jsonify({'error': 'Playlist not found'}), 404
        
        with open(playlist_file, 'r') as f:
            playlist_data = json.load(f)
        
        songs = playlist_data.get('songs', [])
        if len(songs) < 2:
            return jsonify({'error': 'Playlist must have at least 2 songs'}), 400
        
        # Initialize streaming mixer with batch processing
        mix_id = data.get('mix_id') or str(uuid.uuid4())[:8]
        
        mixer = RealTimeStreamingMixer(
            mode=mode,
            transition_duration=transition_duration,
            db_path=str(db_path) if db_path.exists() else None,
            batch_size=3,  # Process 3 songs per batch
            cache_dir=f'temp_audio/cache/{mix_id}'  # Isolated cache per mix
        )
        
        # Callback when first batch is ready (can start playback)
        def on_first_batch_ready():
            print(f"  ✓✓✓ First batch ready for mix {mix_id} - playback can start! ✓✓✓")
        
        # Start batch processing (first batch processed immediately, rest in background)
        mixer.start_playlist(songs, on_first_batch_ready=on_first_batch_ready)
        
        # Store active mix
        active_mixes[mix_id] = mixer
        
        return jsonify({
            'mix_id': mix_id,
            'stream_url': f'/api/stream/{mix_id}/audio',
            'control_url': f'/api/mix/{mix_id}',
            'status': 'processing',  # First batch processing
            'mode': mode,
            'total_songs': len(songs),
            'message': 'Processing first batch... Playback will start when ready (30-90 seconds)'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mix/<mix_id>', methods=['GET'])
def get_mix_status(mix_id: str):
    """Get mix status and current song info."""
    mixer = active_mixes.get(mix_id)
    
    if not mixer:
        return jsonify({'error': 'Mix not found'}), 404
    
    status = mixer.get_status()
    return jsonify(status), 200


@app.route('/api/mix/<mix_id>', methods=['DELETE'])
def stop_mix(mix_id: str):
    """Stop active mix."""
    mixer = active_mixes.get(mix_id)
    
    if not mixer:
        return jsonify({'error': 'Mix not found'}), 404
    
    mixer.stop()
    del active_mixes[mix_id]
    
    return jsonify({'status': 'stopped'}), 200


@app.route('/api/mix/<mix_id>/next', methods=['POST'])
def get_next_song(mix_id: str):
    """
    Get next song (for shuffle mode).
    
    ALIGNMENT: Uses NextSongSelector for intelligent next song selection.
    """
    mixer = active_mixes.get(mix_id)
    
    if not mixer:
        return jsonify({'error': 'Mix not found'}), 404
    
    if mixer.mode != 'shuffle':
        return jsonify({'error': 'Not in shuffle mode'}), 400
    
    next_song = mixer.get_next_song()
    
    if not next_song:
        return jsonify({'error': 'No next song available'}), 404
    
    return jsonify(next_song), 200


@app.route('/api/stream/<mix_id>/audio', methods=['GET'])
def stream_audio(mix_id: str):
    """
    Stream audio chunks for playback.
    
    ALIGNMENT: Progressive streaming with triple buffering.
    """
    #region agent log
    import time, json
    log_path = '/Users/saksham/untitled folder 7/.cursor/debug.log'
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"web_server.py:297","message":"stream_audio endpoint called","data":{"mix_id":mix_id,"chunk":request.args.get('chunk', '0')},"timestamp":int(time.time()*1000)}) + '\n')
    #endregion
    
    mixer = active_mixes.get(mix_id)
    
    if not mixer:
        #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"web_server.py:305","message":"Mix not found","data":{"mix_id":mix_id,"active_mixes":list(active_mixes.keys())},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        return jsonify({'error': 'Mix not found'}), 404
    
    chunk_index = int(request.args.get('chunk', 0))
    
    #region agent log
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"web_server.py:310","message":"Calling get_audio_chunk","data":{"mix_id":mix_id,"chunk_index":chunk_index},"timestamp":int(time.time()*1000)}) + '\n')
    #endregion
    
    def generate():
        """Generate audio chunks."""
        chunk = mixer.get_audio_chunk(chunk_index)
        
        #region agent log
        chunk_size = len(chunk) if chunk else 0
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"web_server.py:320","message":"get_audio_chunk returned","data":{"mix_id":mix_id,"chunk_index":chunk_index,"chunk_size_bytes":chunk_size},"timestamp":int(time.time()*1000)}) + '\n')
        #endregion
        
        if chunk is None:
            # End of stream
            return b''
        
        yield chunk
    
    return Response(
        generate(),
        mimetype='audio/wav',
        headers={
            'Content-Type': 'audio/wav',
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/stream/<mix_id>/next', methods=['GET'])
def get_next_chunk_info(mix_id: str):
    """Get info about next chunk (for pre-buffering)."""
    mixer = active_mixes.get(mix_id)
    
    if not mixer:
        return jsonify({'error': 'Mix not found'}), 404
    
    next_chunk = mixer.get_next_chunk_info()
    
    return jsonify(next_chunk), 200


@app.route('/')
def index():
    """Serve frontend HTML."""
    frontend_path = Path(__file__).parent.parent / 'frontend' / 'index.html'
    if frontend_path.exists():
        return send_file(str(frontend_path))
    else:
        return jsonify({'error': 'Frontend not found'}), 404


@app.route('/mix')
def mix_page():
    """Serve mix player page."""
    mix_path = Path(__file__).parent.parent / 'frontend' / 'mix.html'
    if mix_path.exists():
        return send_file(str(mix_path))
    else:
        return jsonify({'error': 'Mix page not found'}), 404


# Note: The catch-all route below must be last to not interfere with API routes


@app.route('/demo')
def serve_demo():
    """Serve the generated demo mix."""
    import os
    if os.path.exists('temp_audio/demo.wav'):
        return send_file('temp_audio/demo.wav', mimetype='audio/wav')
    return "Demo file not found", 404

if __name__ == '__main__':
    print("="*60)
    print("DJ MIXER WEB API SERVER")
    print("="*60)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Mixes folder: {MIXES_FOLDER}")
    print(f"Database: {db_path if db_path.exists() else 'Not found'}")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True, use_reloader=False)
