#!/usr/bin/env python3
"""
Club AutoMixer server (P2) — paste a YouTube playlist (or local files), get ONE
continuous, gapless, Boiler-Room–style mixed set to listen to.

Run (use the project venv that has matched torch/torchaudio + librosa):
    ./.venv/bin/python club_server.py
Then open http://localhost:5005

Design: the whole set is rendered as a SINGLE timeline by
src.club_mixer.build_continuous_set, so playback is gapless by construction (no
between-track delay — there are no track boundaries to gap). Progressive/
look-ahead streaming while it builds is the next refinement (see
PRODUCTION_STREAMING_PLAN.md, P3).
"""
import sys
import uuid
import threading
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from flask import Flask, jsonify, request, send_file, Response, render_template

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.club_mixer import build_continuous_set  # noqa: E402
from src.settings import SR, CACHE_DIR, HOST, PORT, DOWNLOAD_TIMEOUT_SEC  # noqa: E402

app = Flask(__name__, template_folder=str(ROOT / "templates"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DL_DIR = CACHE_DIR / "tracks"
DL_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS: dict = {}
_PER_TRACK_CAP = 150  # seconds downloaded per track (bounds time; plenty for a blend)

# Bundled demo tracks (file, start_sec) so "Try sample" works with no network.
_SAMPLE_SOURCES = [
    ("mixes/drake_weeknd_mix.wav", 50.0),
    ("mixes/goingbad_sunflower_mix.wav", 30.0),
    ("mixes/lookback_wakeup_mix.wav", 40.0),
]
_SAMPLE_DUR = 40.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_stereo(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=False)
    return y.T if y.ndim == 2 else y


def _resolve_playlist(url: str) -> list:
    """Return a list of watch URLs from a playlist (or a single video)."""
    out = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--get-id", "--no-warnings", url],
        capture_output=True, text=True, timeout=60,
    )
    ids = [x.strip() for x in out.stdout.splitlines() if x.strip()]
    return [f"https://www.youtube.com/watch?v={vid}" for vid in ids]


def _download(url: str) -> str:
    """Download (cached by video id) the first _PER_TRACK_CAP seconds as wav."""
    vid = url.rsplit("=", 1)[-1].rsplit("/", 1)[-1]
    dst = DL_DIR / f"{vid}.wav"
    if dst.exists():
        return str(dst)
    tmp = DL_DIR / f"{vid}.src.%(ext)s"
    subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "wav", "--no-playlist",
         "--extractor-args", "youtube:player_client=android",
         "--download-sections", f"*0-{_PER_TRACK_CAP}",
         "-o", str(tmp), url],
        capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT_SEC, check=True,
    )
    cand = sorted(DL_DIR.glob(f"{vid}.src.*"))
    if not cand:
        raise FileNotFoundError(f"download failed for {url}")
    # normalize to wav/sr
    subprocess.run(["ffmpeg", "-y", "-i", str(cand[0]), "-ar", str(SR), str(dst)],
                   capture_output=True, check=True)
    cand[0].unlink(missing_ok=True)
    return str(dst)


def _produce(sid: str, sources, is_playlist: bool, blend_bars: int, use_stems: bool = True):
    s = SESSIONS[sid]
    try:
        # Bundled-sample shortcut (no network): slice a few demo tracks.
        if not is_playlist and list(sources) == ["__SAMPLE__"]:
            s["status"] = "loading sample tracks"
            tracks = []
            for f, start in _SAMPLE_SOURCES:
                p = ROOT / f
                if not p.exists():
                    continue
                y, _ = sf.read(str(p), start=int(start * SR), frames=int(_SAMPLE_DUR * SR))
                tracks.append(y)
            if not tracks:
                raise RuntimeError("no sample audio found")
            s["status"] = "mixing"
            mixed, markers = build_continuous_set(
                tracks, sr=SR, blend_bars=blend_bars, use_stems=use_stems,
                progress=lambda i, n: s.update(status=f"mixing {i}/{n}"))
            out = CACHE_DIR / f"set_{sid}.wav"
            sf.write(out, mixed, SR)
            s.update(status="ready", audio=str(out), markers=markers,
                     duration=len(mixed) / SR, n_tracks=len(tracks))
            return

        if is_playlist:
            s["status"] = "resolving playlist"
            urls = _resolve_playlist(sources)
            if not urls:
                raise RuntimeError("no tracks found in playlist")
            paths = []
            for i, u in enumerate(urls):
                s["status"] = f"downloading {i+1}/{len(urls)}"
                try:
                    paths.append(_download(u))
                except Exception as e:
                    print(f"skip {u}: {e}")
        else:
            paths = list(sources)
        if len(paths) < 1:
            raise RuntimeError("no playable tracks")

        s["status"] = "loading tracks"
        tracks = [_load_stereo(p) for p in paths]

        def _prog(i, n):
            s["status"] = f"mixing {i}/{n}"
        s["status"] = "mixing"
        mixed, markers = build_continuous_set(tracks, sr=SR, blend_bars=blend_bars,
                                              use_stems=use_stems, progress=_prog)

        out = CACHE_DIR / f"set_{sid}.wav"
        sf.write(out, mixed, SR)
        s.update(status="ready", audio=str(out), markers=markers,
                 duration=len(mixed) / SR, n_tracks=len(tracks))
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
    SESSIONS[sid] = {"status": "starting"}
    threading.Thread(
        target=_produce, args=(sid, url or files, bool(url), blend_bars, use_stems), daemon=True
    ).start()
    return jsonify({"session_id": sid})


@app.route("/api/set/<sid>")
def status(sid):
    s = SESSIONS.get(sid)
    if not s:
        return jsonify({"error": "not found"}), 404
    return jsonify({
        "status": s.get("status"),
        "error": s.get("error"),
        "duration": s.get("duration"),
        "n_tracks": s.get("n_tracks"),
        "markers": s.get("markers") if s.get("status") == "ready" else None,
        "audio_url": f"/api/set/{sid}/audio" if s.get("status") == "ready" else None,
    })


@app.route("/api/set/<sid>/audio")
def audio(sid):
    s = SESSIONS.get(sid)
    if not s or s.get("status") != "ready":
        return jsonify({"error": "not ready"}), 404
    # send_file supports Range requests -> the <audio> element can seek/stream.
    return send_file(s["audio"], mimetype="audio/wav", conditional=True)


if __name__ == "__main__":
    print(f"Club AutoMixer on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
