# Deploying automix (free, cloud)

The app is a single Flask service that renders a continuous DJ set and streams it as
it renders. It is packaged as a container (`Dockerfile`) that runs on any CPU box. The
recommended free home is a **Hugging Face Spaces "Docker" Space** (2 vCPU / 16 GB RAM /
50 GB disk on the free tier — enough RAM to run demucs on CPU).

## What's in the box

| Concern | How it's handled |
|---|---|
| **Memory** | Tracks are decoded lazily, one at a time (`build_continuous_set` accepts loaders); the full set is **never** held in RAM — chunks are flushed to disk as they render. Idle sessions + their chunk files are reaped on a TTL. |
| **Security** | Only `youtube.com`/`youtu.be` URLs accepted (SSRF guard); video ids whitelisted to the canonical 11-char form before use in filenames (path-traversal guard); playlist length, per-track seconds, and download size hard-capped; `gunicorn` (not the Flask dev server); per-IP rate limit on `/api/set/start`. |
| **Cost** | Stem-free by default (no torch needed to run); demucs is opt-in. Demo audio is synthesized, so the demo needs no network and no copyrighted assets. |
| **Research framing** | README carries the method + citations; only royalty-free audio ships. |

## Option A — Hugging Face Spaces (recommended)

1. Create a new **Space** → SDK: **Docker** → **Blank**.
2. Push this repo to the Space's git remote (or connect the GitHub repo):
   ```bash
   git remote add space https://huggingface.co/spaces/<you>/automix
   git push space HEAD:main
   ```
   The Space reads the YAML front-matter at the top of `README.md`
   (`sdk: docker`, `app_port: 7860`) and builds the `Dockerfile`.
3. First build takes a few minutes (demucs + CPU torch). When it's up, open the Space.
   **Try sample** works immediately.

### Make YouTube work on a datacenter IP (optional)

YouTube bot-throttles datacenter IPs (all cloud hosts), so server-side downloads often
fail with *"Sign in to confirm you're not a bot."* To mitigate, supply your own cookies:

1. Export cookies for `youtube.com` in Netscape format (e.g. the *Get cookies.txt*
   browser extension).
2. In the Space: **Settings → Variables and secrets → New secret**
   - Name: `YT_COOKIES`
   - Value: paste the entire `cookies.txt` contents
3. Restart. The entrypoint writes the secret to a file and points yt-dlp at it.

If you'd rather disable YouTube entirely (samples only), set a **variable**
`ALLOW_YOUTUBE=false`.

## Option B — any Docker host (Fly.io, Render, a VPS)

```bash
# Full image (includes demucs / CPU torch):
docker build -t automix .
# Or a small, torch-free image (EQ transitions only, ~5x smaller):
docker build -t automix-slim --build-arg INSTALL_STEMS=false .

docker run -p 7860:7860 \
  -e ALLOW_YOUTUBE=true \
  -e YT_COOKIES="$(cat cookies.txt)" \
  automix
# open http://localhost:7860
```

## Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `PORT` | `7860` | Listen port (HF Spaces expects 7860). |
| `ALLOW_YOUTUBE` | `true` | Set `false` to disable YouTube ingestion (samples only). |
| `MAX_PLAYLIST_TRACKS` | `12` | Cap tracks per set. |
| `PER_TRACK_CAP_SEC` | `150` | Seconds fetched per track. |
| `MAX_DOWNLOAD_MB` | `60` | yt-dlp `--max-filesize`. |
| `SESSION_TTL_SEC` | `1800` | Evict idle sessions after this long. |
| `MAX_SESSIONS` | `24` | Concurrent session cap. |
| `RATE_LIMIT` | `30 per hour` | Per-IP limit on `/api/set/start`. |
| `YT_COOKIES` | — | (secret) cookies.txt contents for yt-dlp. |
| `GUNICORN_THREADS` | `8` | Worker threads (single worker process). |

> **Single worker only.** Session state and the chunk cache live in process memory, so
> the server must run **one** gunicorn worker (`--workers 1`, already set in the
> entrypoint). Scale with threads, not workers.

## Phase 2 (later): free GPU for fast stems

CPU demucs is slow (~30 s–2 min per transition). For fast stems at zero cost, port the
front-end to the **Gradio SDK** and wrap the demucs call in `@spaces.GPU` to use HF
**ZeroGPU** (free, dynamically-allocated GPU). Not required for the demo above.
