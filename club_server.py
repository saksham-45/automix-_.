#!/usr/bin/env python3
"""
Club AutoMixer server — paste a YouTube playlist (or use the bundled sample), get
ONE continuous, gapless, Boiler-Room–style mixed set with TRUE no-delay start.

Run (use the project venv with matched torch/torchaudio + librosa + demucs):
    ./.venv/bin/python club_server.py
Then open http://localhost:5005

Streaming model (P3): the mixed timeline is sample-contiguous by construction, so
the producer emits it as fixed-size PCM chunks AS IT RENDERS (via build_continuous_set's
on_part callback). The web client schedules chunks back-to-back with the Web Audio
API and starts playback on chunk 0 — playback begins seconds in, while the rest of
the set is still rendering ahead of the playhead. No upfront full-set wait.
"""
import sys
import uuid
import threading
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from flask import Flask, jsonify, request, send_file, render_template

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.club_mixer import build_continuous_set, _as_stereo  # noqa: E402
from src.settings import SR, CACHE_DIR, HOST, PORT, DOWNLOAD_TIMEOUT_SEC  # noqa: E402

app = Flask(__name__, template_folder=str(ROOT / "templates"))
CHUNK_DIR = CACHE_DIR / "chunks"
DL_DIR = CACHE_DIR / "tracks"
for d in (CHUNK_DIR, DL_DIR):
    d.mkdir(parents=True, exist_ok=True)

SESSIONS: dict = {}
CHUNK_SEC = 4.0
_PER_TRACK_CAP = 150
_SAMPLE_SOURCES = [
    ("mixes/drake_weeknd_mix.wav", 50.0),
    ("mixes/goingbad_sunflower_mix.wav", 30.0),
    ("mixes/lookback_wakeup_mix.wav", 40.0),
]
_SAMPLE_DUR = 45.0


# --------------------------------------------------------------------------- #
# track collection
# --------------------------------------------------------------------------- #
def _load_stereo(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=False)
    return y.T if y.ndim == 2 else y


def _resolve_playlist(url: str) -> list:
    out = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--get-id", "--no-warnings", url],
        capture_output=True, text=True, timeout=60)
    return [f"https://www.youtube.com/watch?v={v.strip()}" for v in out.stdout.splitlines() if v.strip()]


def _download(url: str) -> str:
    vid = url.rsplit("=", 1)[-1].rsplit("/", 1)[-1]
    dst = DL_DIR / f"{vid}.wav"
    if dst.exists():
        return str(dst)
    tmp = DL_DIR / f"{vid}.src.%(ext)s"
    subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "wav", "--no-playlist",
         "--extractor-args", "youtube:player_client=android",
         "--download-sections", f"*0-{_PER_TRACK_CAP}", "-o", str(tmp), url],
        capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT_SEC, check=True)
    cand = sorted(DL_DIR.glob(f"{vid}.src.*"))
    if not cand:
        raise FileNotFoundError(f"download failed: {url}")
    subprocess.run(["ffmpeg", "-y", "-i", str(cand[0]), "-ar", str(SR), str(dst)],
                   capture_output=True, check=True)
    cand[0].unlink(missing_ok=True)
    return str(dst)


def _collect_tracks(sources, is_playlist, s) -> list:
    if not is_playlist and list(sources) == ["__SAMPLE__"]:
        s["status"] = "loading sample tracks"
        tracks = []
        for f, start in _SAMPLE_SOURCES:
            p = ROOT / f
            if p.exists():
                y, _ = sf.read(str(p), start=int(start * SR), frames=int(_SAMPLE_DUR * SR))
                tracks.append(y)
        return tracks
    if is_playlist:
        s["status"] = "resolving playlist"
        urls = _resolve_playlist(sources)
        tracks = []
        for i, u in enumerate(urls):
            s["status"] = f"downloading {i+1}/{len(urls)}"
            try:
                tracks.append(_load_stereo(_download(u)))
            except Exception as e:
                print(f"skip {u}: {e}")
        return tracks
    return [_load_stereo(p) for p in sources]


# --------------------------------------------------------------------------- #
# progressive producer
# --------------------------------------------------------------------------- #
def _produce(sid: str, sources, is_playlist: bool, blend_bars: int, use_stems: bool):
    s = SESSIONS[sid]
    s["chunks"] = []          # ordered list of chunk file paths
    s["chunk_sec"] = CHUNK_SEC
    CH = int(CHUNK_SEC * SR)
    buf = {"pcm": np.zeros((0, 2), dtype=np.float32)}

    def _flush(final=False):
        while len(buf["pcm"]) >= CH or (final and len(buf["pcm"]) > 0):
            take = buf["pcm"][:CH]
            buf["pcm"] = buf["pcm"][CH:]
            idx = len(s["chunks"])
            p = CHUNK_DIR / f"{sid}_{idx:05d}.wav"
            # float subtype preserves >1.0 sums; 0.9 uniform safety gain (consistent
            # across chunks so no level steps), final master is the source level.
            sf.write(p, np.clip(take * 0.9, -1.0, 1.0), SR, subtype="FLOAT")
            s["chunks"].append(str(p))
            if final and len(buf["pcm"]) == 0:
                break

    def _on_part(part):
        buf["pcm"] = np.vstack([buf["pcm"], _as_stereo(np.asarray(part)).astype(np.float32)])
        _flush(False)

    try:
        tracks = _collect_tracks(sources, is_playlist, s)
        if not tracks:
            raise RuntimeError("no playable tracks")
        s["status"] = "mixing"

        def _prog(i, n):
            s["status"] = f"mixing {i}/{n}"

        _, markers = build_continuous_set(
            tracks, sr=SR, blend_bars=blend_bars, use_stems=use_stems,
            on_part=_on_part, progress=_prog)
        _flush(final=True)
        s.update(status="ready", markers=markers, n_tracks=len(tracks),
                 chunk_count=len(s["chunks"]),
                 duration=len(s["chunks"]) * CHUNK_SEC)
    except Exception as e:
        import traceback; traceback.print_exc()
        s.update(status="error", error=str(e))


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("club_player.html")


@app.route("/api/set/start", methods=["POST"])
def start():
    data = request.get_json(force=True) or {}
    url = (data.get("url") or "").strip()
    files = data.get("files") or []
    blend_bars = int(data.get("blend_bars", 16))
    use_stems = bool(data.get("use_stems", True))
    if not url and not files:
        return jsonify({"error": "provide a playlist url or files[]"}), 400
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = {"status": "starting", "chunks": [], "chunk_sec": CHUNK_SEC}
    threading.Thread(target=_produce, args=(sid, url or files, bool(url), blend_bars, use_stems),
                     daemon=True).start()
    return jsonify({"session_id": sid})


@app.route("/api/set/<sid>")
def status(sid):
    s = SESSIONS.get(sid)
    if not s:
        return jsonify({"error": "not found"}), 404
    return jsonify({
        "status": s.get("status"),
        "error": s.get("error"),
        "chunk_sec": s.get("chunk_sec", CHUNK_SEC),
        "chunks_ready": len(s.get("chunks", [])),      # grows during render
        "chunk_count": s.get("chunk_count"),           # set only when finished
        "duration": s.get("duration"),
        "n_tracks": s.get("n_tracks"),
        "markers": s.get("markers") if s.get("status") == "ready" else None,
    })


@app.route("/api/set/<sid>/chunk/<int:n>")
def chunk(sid, n):
    s = SESSIONS.get(sid)
    if not s:
        return jsonify({"error": "not found"}), 404
    chunks = s.get("chunks", [])
    if n >= len(chunks):
        # not produced yet (client should retry) or past the end
        code = 416 if s.get("status") in ("ready", "error") else 404
        return ("", code)
    return send_file(chunks[n], mimetype="audio/wav", conditional=True)


if __name__ == "__main__":
    print(f"Club AutoMixer on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
