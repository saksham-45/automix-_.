#!/usr/bin/env python3
"""Flask server: YouTube Playlist AutoMixer"""
import sys
import logging
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_from_directory

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stream_manager import manager
from src.settings import CACHE_DIR, HOST, PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mix_server")

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"), static_folder=str(PROJECT_ROOT / "static"))
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024  # 1MB for requests

@app.route("/")
def index():
    return render_template("player.html")

@app.route("/api/playlist", methods=["POST"])
def start_playlist():
    url = (request.json.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL required"}), 400
    
    try:
        morph_settings = request.json.get("morph_settings", {})
        sid = manager.start_session(url)
        session = manager.get_session(sid)
        if session and morph_settings:
            session.morph_depth = float(morph_settings.get("depth", session.morph_depth))
            session.morph_strategy = morph_settings.get("strategy", session.morph_strategy)
            session.enable_morphing = bool(morph_settings.get("enabled", session.enable_morphing))
            
        return jsonify({"session_id": sid})
    except Exception as e:
        logger.exception("start_playlist failed")
        return jsonify({"error": "Failed to start playlist"}), 500

@app.route("/api/session/<sid>")
def get_session_status(sid):
    session = manager.get_session(sid)
    if not session:
        return jsonify({"error": "Session not found"}), 404
        
    # Drain the queue to get new chunks
    new_chunks = []
    while not session.chunks_queue.empty():
        chunk = session.chunks_queue.get()
        if chunk is None: # Sentinel
            session.status = "finished"
            continue
        session.processed_chunks.append(chunk)

    # Convert chunks to API format
    chunks_data = []
    for c in session.processed_chunks:
        chunks_data.append({
            "id": c["id"],
            "type": c["type"],
            "title": c.get("title", ""),
            "duration": c["duration"],
            "url": f"/api/audio/{Path(c['path']).name}"
        })
        
    return jsonify({
        "status": session.status,
        "chunks": chunks_data,
        "continuous_url": f"/api/session/{sid}/continuous" if getattr(session, "continuous_ready", False) else None
    })

@app.route("/api/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(CACHE_DIR, filename)

@app.route("/api/session/<sid>/continuous")
def serve_continuous(sid):
    session = manager.get_session(sid)
    if not session or not getattr(session, "continuous_ready", False):
        return jsonify({"error": "Continuous stream not ready"}), 404
    return send_from_directory(CACHE_DIR, Path(session.continuous_path).name, mimetype="audio/wav")

if __name__ == "__main__":
    logger.info("Starting server on %s:%s", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
