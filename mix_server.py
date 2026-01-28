#!/usr/bin/env python3
"""Flask server: paste two YouTube links, get a mix WAV to play on the page."""
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

# Project root = directory containing this file
PROJECT_ROOT = Path(__file__).resolve().parent
MIXES_DIR = PROJECT_ROOT / "static" / "mixes"
MIXES_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"), static_folder=str(PROJECT_ROOT / "static"))
app.config["MAX_CONTENT_LENGTH"] = 1024  # form field size limit (URLs only)

# In-memory job store: job_id -> {"status": "pending"|"done"|"error", "path": Path|None, "error": str|None}
jobs = {}
_lock = threading.Lock()


def _run_mix(job_id: str, url1: str, url2: str) -> None:
    path = MIXES_DIR / f"{job_id}.wav"
    try:
        from scripts.mix_runner import run_youtube_mix
        run_youtube_mix(url1, url2, path)
        with _lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["path"] = path
    except Exception as e:
        with _lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass


@app.route("/")
def index():
    return render_template("mix_index.html")


@app.route("/mix", methods=["POST"])
def start_mix():
    url1 = (request.form.get("url1") or "").strip()
    url2 = (request.form.get("url2") or "").strip()
    if not url1 or not url2:
        return jsonify({"error": "Both URL 1 and URL 2 are required"}), 400
    job_id = str(uuid.uuid4())
    with _lock:
        jobs[job_id] = {"status": "pending", "path": None, "error": None}
    thread = threading.Thread(target=_run_mix, args=(job_id, url1, url2))
    thread.daemon = True
    thread.start()
    return jsonify({"job_id": job_id})


@app.route("/mix/status/<job_id>")
def mix_status(job_id):
    with _lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    out = {"status": job["status"]}
    if job["status"] == "done":
        out["output_url"] = f"/mix/output/{job_id}"
    if job["status"] == "error":
        out["error"] = job["error"]
    return jsonify(out)


@app.route("/mix/output/<job_id>")
def mix_output(job_id):
    with _lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done" or not job.get("path"):
        return jsonify({"error": "Mix not ready or not found"}), 404
    path = Path(job["path"])
    if not path.exists():
        return jsonify({"error": "File missing"}), 404
    return send_file(path, mimetype="audio/wav", as_attachment=False, download_name=f"mix_{job_id[:8]}.wav")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
