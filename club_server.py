#!/usr/bin/env python3
"""
Club AutoMixer server — paste a YouTube playlist (or use the bundled sample), get
ONE continuous, gapless, Boiler-Room–style mixed set with TRUE no-delay start.

Run locally (use the project venv with matched torch/torchaudio + librosa + demucs):
    PORT=5005 ./.venv/bin/python club_server.py
Then open http://localhost:5005

In a container / on Hugging Face Spaces it listens on $PORT (default 7860) and is
served by gunicorn with a SINGLE worker + threads (see Dockerfile) — single worker
because session state and the chunk cache live in this process's memory.

Streaming model (P3): the mixed timeline is sample-contiguous by construction, so
the producer emits it as fixed-size PCM chunks AS IT RENDERS (via build_continuous_set's
on_part callback). The web client schedules chunks back-to-back with the Web Audio
API and starts playback on chunk 0 — playback begins seconds in, while the rest of
the set is still rendering ahead of the playhead. No upfront full-set wait.

Memory model: tracks are loaded off disk lazily (one at a time) by build_continuous_set,
and the set is NEVER held fully in RAM — each rendered chunk is flushed to disk and
streamed. Idle sessions and their chunk files are reaped on a TTL so disk/RAM stay bounded.

Security: only YouTube URLs are accepted (SSRF guard); video ids are whitelisted to the
canonical 11-char form before use in filenames (path-traversal guard); playlist length,
per-track seconds and download size are hard-capped; the start endpoint is rate-limited.
"""
import re
import sys
import time
import uuid
import threading
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
import librosa
from flask import Flask, jsonify, request, send_file, render_template

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.club_mixer import build_continuous_set, _as_stereo  # noqa: E402
from src.demo_samples import ensure_demo_samples  # noqa: E402
from src.settings import (  # noqa: E402
    SR, CACHE_DIR, HOST, PORT, DOWNLOAD_TIMEOUT_SEC, YTDLP_EXTRACTOR_ARGS,
    YT_COOKIES_FILE, ALLOW_YOUTUBE, MAX_PLAYLIST_TRACKS, PER_TRACK_CAP_SEC,
    MAX_DOWNLOAD_MB, SESSION_TTL_SEC, MAX_SESSIONS, RATE_LIMIT,
)

app = Flask(__name__, template_folder=str(ROOT / "templates"))
CHUNK_DIR = CACHE_DIR / "chunks"
DL_DIR = CACHE_DIR / "tracks"
DEMO_DIR = ROOT / "data" / "demo"
for d in (CHUNK_DIR, DL_DIR):
    d.mkdir(parents=True, exist_ok=True)
# Synthesize the royalty-free demo loops if they aren't present (idempotent).
ensure_demo_samples(DEMO_DIR, sr=SR)

# Optional per-IP rate limiting on the expensive endpoint. No-op if flask-limiter
# isn't installed, so the server still runs in a minimal environment.
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(get_remote_address, app=app, default_limits=[])
    _rate_limit = limiter.limit(RATE_LIMIT)
except Exception:  # flask-limiter missing or failed to init
    def _rate_limit(fn):
        return fn

SESSIONS: dict = {}
_SLOCK = threading.Lock()
CHUNK_SEC = 4.0
_VIDEO_ID_RE = re.compile(r"[0-9A-Za-z_-]{11}")
_ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com",
                  "music.youtube.com", "youtu.be", "www.youtu.be"}

# Bundled, offline demo tracks (no network / no copyright-download path). Short
# royalty-free loops that ship with the app so the demo always works even when
# YouTube is blocked.
_SAMPLE_SOURCES = [
    ("data/demo/sample_a.wav", 0.0),
    ("data/demo/sample_b.wav", 0.0),
    ("data/demo/sample_c.wav", 0.0),
]
_SAMPLE_DUR = 80.0   # long enough that track bodies are audible between transitions


# --------------------------------------------------------------------------- #
# validation / security helpers
# --------------------------------------------------------------------------- #
def _is_youtube_url(url: str) -> bool:
    """SSRF guard: only accept http(s) URLs whose host is a known YouTube domain,
    so the download subprocess can never be pointed at an internal/arbitrary host."""
    try:
        u = urlparse(url)
    except Exception:
        return False
    return u.scheme in ("http", "https") and (u.hostname or "").lower() in _ALLOWED_HOSTS


def _safe_vid(raw: str) -> str:
    """Extract the canonical 11-char YouTube id from a url/id and reject anything
    else. Used to build cache filenames -> path-traversal / injection guard."""
    tail = raw.rsplit("=", 1)[-1].rsplit("/", 1)[-1]
    m = _VIDEO_ID_RE.fullmatch(tail) or _VIDEO_ID_RE.search(raw)
    if not m:
        raise ValueError(f"no valid video id in: {raw[:80]}")
    return m.group(0)


def _ytdlp_base() -> list:
    """Common yt-dlp flags incl. optional cookies (the datacenter-IP bot mitigation)."""
    base = ["yt-dlp", "--no-warnings",
            "--extractor-args", YTDLP_EXTRACTOR_ARGS]
    if YT_COOKIES_FILE and Path(YT_COOKIES_FILE).is_file():
        base += ["--cookies", YT_COOKIES_FILE]
    return base


# --------------------------------------------------------------------------- #
# track collection (returns LAZY loaders, never all decoded audio at once)
# --------------------------------------------------------------------------- #
def _load_stereo(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=False)
    return y.T if y.ndim == 2 else y


def _resolve_playlist(url: str) -> list:
    out = subprocess.run(
        _ytdlp_base() + ["--flat-playlist", "--get-id", url],
        capture_output=True, text=True, timeout=60)
    ids = [v.strip() for v in out.stdout.splitlines() if v.strip()][:MAX_PLAYLIST_TRACKS]
    return [f"https://www.youtube.com/watch?v={v}" for v in ids]


def _download(url: str) -> str:
    vid = _safe_vid(url)
    dst = DL_DIR / f"{vid}.wav"
    if dst.exists():
        return str(dst)
    tmp = DL_DIR / f"{vid}.src.%(ext)s"
    subprocess.run(
        _ytdlp_base() + [
            "-x", "--audio-format", "wav", "--no-playlist",
            "--max-filesize", f"{MAX_DOWNLOAD_MB}M",
            "--download-sections", f"*0-{PER_TRACK_CAP_SEC}",
            "-o", str(tmp), f"https://www.youtube.com/watch?v={vid}"],
        capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT_SEC, check=True)
    cand = sorted(DL_DIR.glob(f"{vid}.src.*"))
    if not cand:
        raise FileNotFoundError(f"download failed: {vid}")
    subprocess.run(["ffmpeg", "-y", "-i", str(cand[0]), "-ar", str(SR), str(dst)],
                   capture_output=True, check=True)
    cand[0].unlink(missing_ok=True)
    return str(dst)


def _collect_loaders(sources, is_playlist, s) -> list:
    """Return a list of zero-arg loaders (thunks). build_continuous_set calls them
    one at a time, so at most two tracks are ever decoded in RAM simultaneously.
    Downloading (disk-bound) still happens up front; decoding (RAM-bound) is lazy."""
    if not is_playlist and list(sources) == ["__SAMPLE__"]:
        s["status"] = "loading sample tracks"
        loaders = []
        for f, start in _SAMPLE_SOURCES:
            p = ROOT / f
            if p.exists():
                loaders.append(lambda p=p, start=start: sf.read(
                    str(p), start=int(start * SR), frames=int(_SAMPLE_DUR * SR))[0])
        return loaders
    if is_playlist:
        s["status"] = "resolving playlist"
        urls = _resolve_playlist(sources)
        paths = []
        for i, u in enumerate(urls):
            s["status"] = f"downloading {i+1}/{len(urls)}"
            try:
                paths.append(_download(u))
            except Exception as e:
                print(f"skip {u}: {e}")
        return [lambda p=p: _load_stereo(p) for p in paths]
    return [lambda p=p: _load_stereo(p) for p in sources]


# --------------------------------------------------------------------------- #
# session reaper — bound RAM + disk on a small box
# --------------------------------------------------------------------------- #
def _reap_sessions():
    """Evict idle/old sessions and delete their chunk files. Also enforce a hard
    cap on concurrent sessions by evicting the oldest. Keeps RAM and disk bounded
    so the process can't be made to grow without limit."""
    now = time.time()
    with _SLOCK:
        items = sorted(SESSIONS.items(), key=lambda kv: kv[1].get("touched", 0))
        stale = [sid for sid, s in items if now - s.get("touched", now) > SESSION_TTL_SEC]
        overflow = [sid for sid, _ in items[:max(0, len(items) - MAX_SESSIONS + 1)]]
        for sid in set(stale) | set(overflow):
            s = SESSIONS.pop(sid, None)
            if not s:
                continue
            for cp in s.get("chunks", []):
                try:
                    Path(cp).unlink(missing_ok=True)
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
# progressive producer
# --------------------------------------------------------------------------- #
def _produce(sid: str, sources, is_playlist: bool, blend_bars: int, use_stems: bool):
    s = SESSIONS[sid]
    s["chunks"] = []          # ordered list of chunk file paths
    s["markers"] = []         # grows as parts are produced (for live UI)
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
            s["touched"] = time.time()
            if final and len(buf["pcm"]) == 0:
                break

    def _on_part(part, marker=None):
        buf["pcm"] = np.vstack([buf["pcm"], _as_stereo(np.asarray(part)).astype(np.float32)])
        if marker:
            s["markers"].append(marker)
        _flush(False)

    try:
        loaders = _collect_loaders(sources, is_playlist, s)
        if not loaders:
            raise RuntimeError("no playable tracks")
        s["status"] = "mixing"

        def _prog(i, n):
            s["status"] = f"mixing {i}/{n}"
            s["touched"] = time.time()

        _, markers = build_continuous_set(
            loaders, sr=SR, blend_bars=blend_bars, use_stems=use_stems,
            on_part=_on_part, progress=_prog)
        _flush(final=True)
        s["markers"] = markers     # authoritative full list
        s.update(status="ready", n_tracks=len(loaders),
                 chunk_count=len(s["chunks"]),
                 duration=len(s["chunks"]) * CHUNK_SEC, touched=time.time())
    except Exception as e:
        import traceback; traceback.print_exc()
        s.update(status="error", error=str(e), touched=time.time())


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("club_player.html")


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "sessions": len(SESSIONS)})


@app.route("/api/set/start", methods=["POST"])
@_rate_limit
def start():
    _reap_sessions()
    data = request.get_json(force=True, silent=True) or {}
    url = (data.get("url") or "").strip()
    files = data.get("files") or []
    blend_bars = max(8, min(32, int(data.get("blend_bars", 16))))
    use_stems = bool(data.get("use_stems", False))   # default OFF: fast EQ-swap demo
    is_sample = list(files) == ["__SAMPLE__"]
    if not url and not files:
        return jsonify({"error": "provide a playlist url or files[]"}), 400
    if url:
        if not ALLOW_YOUTUBE:
            return jsonify({"error": "YouTube ingestion is disabled on this server"}), 403
        if not _is_youtube_url(url):
            return jsonify({"error": "only youtube.com / youtu.be URLs are accepted"}), 400
    # Reject arbitrary server-side file paths from the client (only the bundled sample).
    if files and not is_sample:
        return jsonify({"error": "arbitrary file paths are not accepted"}), 400
    sid = uuid.uuid4().hex[:12]
    with _SLOCK:
        SESSIONS[sid] = {"status": "starting", "chunks": [], "chunk_sec": CHUNK_SEC,
                         "touched": time.time()}
    threading.Thread(target=_produce, args=(sid, url or files, bool(url), blend_bars, use_stems),
                     daemon=True).start()
    return jsonify({"session_id": sid})


@app.route("/api/set/<sid>")
def status(sid):
    s = SESSIONS.get(sid)
    if not s:
        return jsonify({"error": "not found"}), 404
    s["touched"] = time.time()
    return jsonify({
        "status": s.get("status"),
        "error": s.get("error"),
        "chunk_sec": s.get("chunk_sec", CHUNK_SEC),
        "chunks_ready": len(s.get("chunks", [])),      # grows during render
        "chunk_count": s.get("chunk_count"),           # set only when finished
        "duration": s.get("duration"),
        "n_tracks": s.get("n_tracks"),
        "markers": s.get("markers") or None,   # live, grows during render
    })


@app.route("/api/set/<sid>/chunk/<int:n>")
def chunk(sid, n):
    s = SESSIONS.get(sid)
    if not s:
        return jsonify({"error": "not found"}), 404
    s["touched"] = time.time()
    chunks = s.get("chunks", [])
    if n >= len(chunks):
        # not produced yet (client should retry) or past the end
        code = 416 if s.get("status") in ("ready", "error") else 404
        return ("", code)
    return send_file(chunks[n], mimetype="audio/wav", conditional=True)


if __name__ == "__main__":
    print(f"Club AutoMixer on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
