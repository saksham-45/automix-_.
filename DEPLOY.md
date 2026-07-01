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

## Option A — Hugging Face Spaces, free CPU (recommended)

Runs the whole app — **including demucs on CPU** — on the free tier (2 vCPU / 16 GB).
demucs is slow on CPU (~10–30 s per transition) and off by default, so the demo stays
snappy; turning on **HQ transitions** still works, just slowly.

> ⚠️ **Do not push the bundled audio to a public Space.** `mixes/` (~1.3 GB) and
> `data/old_mixes/` (~445 MB) are tracked `.wav`s of copyrighted / copyright-derived
> audio and are **not needed at runtime** (the sample is the synthesized loops, generated
> at build). The recipe below sends a clean single-commit tree without them or the heavy
> git history.

1. **Account + token.** Create a free account, then **Settings → Access Tokens → New
   token** (role: *write*). Copy it.
2. **Create the Space:** huggingface.co/new-space → name `automix`, **SDK: Docker**,
   **Hardware: CPU basic (free)**. It reads the `README.md` front-matter
   (`sdk: docker`, `app_port: 7860`) and builds the `Dockerfile`.
3. **Push a clean tree** (from this worktree, after committing your changes):
   ```bash
   # one orphan commit = only current files; no heavy history, no copyrighted audio
   git checkout --orphan space-deploy
   git add -A
   git rm -r --cached --quiet mixes data/old_mixes data/demo   # drop bundled audio (files stay on disk)
   git commit -m "automix: HF Space deploy build"
   git remote add space https://<user>:<HF_WRITE_TOKEN>@huggingface.co/spaces/<user>/automix
   git push -f space space-deploy:main
   git checkout fix/debug-instrumentation-and-bugs   # back to your working branch
   ```
   (`data/demo` is dropped too — the demo loops are regenerated at build by the
   Dockerfile's `python -m src.demo_samples` step, so the Space recreates them itself.)
4. First build takes a few minutes (CPU torch + demucs). When it's up, open the Space —
   **Sample** works immediately, no network or keys needed.

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

## Running on a GPU (web)

A GPU only accelerates **demucs** (the "HQ transitions" toggle). The rest of the mix
is CPU DSP. demucs auto-selects CUDA when a GPU is visible — no code change, just a
CUDA torch build (`--build-arg TORCH_VARIANT=cu121`). Pick one of three paths:

### Path 1 — persistent GPU box (least rewrite, paid)
Keep this exact Flask app; run it on GPU hardware.

- **Hugging Face Spaces (Docker) + GPU hardware.** Deploy as in Option A, then
  **Settings → Hardware → pick a GPU** (e.g. *Nvidia T4 small*, ~US$0.40/hr; billed
  while running — set a **sleep timer** so an idle Space stops). Rebuild the Space with
  build arg `TORCH_VARIANT=cu121` (Settings → Variables: add `TORCH_VARIANT=cu121`).
- **Fly.io GPU / RunPod / Lambda / a GPU VPS.** `docker build --build-arg TORCH_VARIANT=cu121 -t automix . && docker run --gpus all -p 7860:7860 automix`.

There is **no free persistent GPU** — a server holding a GPU 24/7 always costs money.
Mitigate with idle-sleep so you only pay while it's actually mixing.

### Path 2 — free GPU via HF ZeroGPU (needs a Gradio-SDK Space)
ZeroGPU (free, dynamically-allocated A100) is only available to **Gradio/Streamlit SDK**
Spaces, **not** Docker Spaces. To use it while keeping this custom streaming UI:

1. Create a **Gradio SDK** Space.
2. In `app.py`, build a Gradio app, then **mount this Flask/streaming API onto it**
   (`gradio.mount_gradio_app` / a FastAPI sub-app) so the chunk endpoints + cassette UI
   still serve.
3. Move the demucs call into a function decorated with `@spaces.GPU` (import `spaces`);
   ZeroGPU attaches a GPU only for that call. demucs-per-transition is short, so this
   fits the ZeroGPU model well and burns little quota.

Free, but GPU access is **quota-limited** and queued under load. This is a real port
(a day's work), not a flag flip.

### Path 3 — keep app on free CPU, offload only demucs to serverless GPU
Best cost/scale: the Flask app stays on a free CPU box; only stem separation is sent to
a pay-per-second GPU that scales to zero.

- **Modal** (~US$30/mo free credits) or **Replicate / RunPod serverless**.
- Wrap `StemSeparator.separate_stems` (`src/stem_separator.py`) so it sends the short
  transition segment to a remote GPU function running htdemucs and returns the stems.
- At low volume this is effectively free and there's no idle GPU cost.

### Which should I pick?
- **Just want it live, cheaply, and HQ stems are optional** → free CPU Space (Option A).
  CPU demucs works, just slow; or leave HQ off (EQ transitions sound great).
- **Want fast stems and will pay a little** → **Path 1** (HF GPU hardware + `cu121`, with
  idle-sleep). Least effort.
- **Want fast stems for free** → **Path 2** (ZeroGPU; accept the Gradio port + quota).
- **Want scale-to-zero, pay-per-use** → **Path 3** (Modal/Replicate offload).
