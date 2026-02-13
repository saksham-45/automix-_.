#!/usr/bin/env python3
"""Fetch a YouTube playlist and generate one mix WAV per consecutive song pair.
Output: playlist_mix_01.wav (song1→song2), playlist_mix_02.wav (song2→song3), ...
Play files in order to hear the full mix from first to last song.
"""
import sys
import subprocess
import shutil
from pathlib import Path

# Unbuffer stdout so piped logs (e.g. tee playlist_mix_log.txt) update line-by-line
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMP_DIR = PROJECT_ROOT / "temp_audio"
TEMP_DIR.mkdir(exist_ok=True)


def download_youtube_audio(
    url: str,
    output_path: Path,
    max_duration: int = 60,
    from_end: bool = False,
) -> Path:
    """Download audio from YouTube URL, limit to max_duration seconds."""
    print(f"Downloading: {url}")
    if from_end:
        print(f"  → {output_path.name} (last {max_duration}s)")
    else:
        print(f"  → {output_path.name} (first {max_duration}s)")

    temp_output = output_path.parent / f"temp_{output_path.name}"
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--no-playlist",
        "--extractor-args",
        "youtube:player_client=android",
        "-o",
        str(temp_output).replace(".wav", ".%(ext)s"),
        url,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error downloading: {e}")
        if e.stderr:
            print(e.stderr[:300])
        raise

    temp_pattern = temp_output.stem
    actual_file = None
    for ext in [".wav", ".opus", ".m4a", ".mp3"]:
        candidate = output_path.parent / f"{temp_pattern}{ext}"
        if candidate.exists():
            actual_file = candidate
            break
    if not actual_file:
        candidates = list(output_path.parent.glob(f"{temp_pattern}.*"))
        actual_file = candidates[0] if candidates else None

    if not actual_file or not actual_file.exists():
        raise FileNotFoundError(f"Downloaded file not found: {output_path}")

    if max_duration > 0:
        if from_end:
            probe_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(actual_file),
            ]
            probe_result = subprocess.run(
                probe_cmd, capture_output=True, text=True, check=True
            )
            total_duration = float(probe_result.stdout.strip())
            start_time = max(0, total_duration - max_duration)
            print(
                f"  Extracting last {max_duration}s (from {start_time:.1f}s to {total_duration:.1f}s)..."
            )
            trim_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(actual_file),
                "-ss",
                str(start_time),
                "-t",
                str(max_duration),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                str(output_path),
            ]
        else:
            print(f"  Trimming to first {max_duration} seconds...")
            trim_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(actual_file),
                "-t",
                str(max_duration),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                str(output_path),
            ]
        subprocess.run(trim_cmd, capture_output=True, check=True)
        actual_file.unlink()
    else:
        if actual_file.suffix != ".wav":
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(actual_file),
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "44100",
                    str(output_path),
                ],
                capture_output=True,
                check=True,
            )
            actual_file.unlink()
        elif actual_file != output_path:
            shutil.move(str(actual_file), str(output_path))

    if output_path.exists():
        print(f"  ✓ Downloaded: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        return output_path
    raise FileNotFoundError(f"Output file not created: {output_path}")


def fetch_playlist_video_ids(playlist_url: str) -> list[dict]:
    """Return list of {id, title} for each entry in the playlist."""
    try:
        import yt_dlp
        ydl_opts = {
            "quiet": True,
            "extract_flat": True,
            "skip_download": True,
            "socket_timeout": 10,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
        if not info or "entries" not in info:
            raise ValueError("Not a valid playlist or no entries")
        entries = []
        for entry in info["entries"]:
            if entry is None:
                continue
            vid = entry.get("id")
            if not vid:
                continue
            entries.append({"id": vid, "title": entry.get("title", vid)})
        return entries
    except ImportError:
        pass  # Fall back to CLI

    # Fallback: use yt-dlp CLI when Python package not installed
    result = subprocess.run(
        [
            "yt-dlp",
            "--flat-playlist",
            "--print", "%(id)s\t%(title)s",
            "--no-warnings",
            playlist_url,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"✗ Failed to fetch playlist: {result.stderr}")
        raise RuntimeError("Failed to fetch playlist video IDs")
    entries = []
    for line in (result.stdout or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        vid = parts[0].strip()
        title = parts[1].strip() if len(parts) > 1 else vid
        if vid and not vid.startswith("#"):
            entries.append({"id": vid, "title": title})
    if not entries:
        raise ValueError("Not a valid playlist or no entries")
    return entries


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate one mix WAV per consecutive pair in a YouTube playlist"
    )
    parser.add_argument(
        "playlist_url",
        help="YouTube/YouTube Music playlist URL",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Seconds to use from each song (default 60)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT,
        help="Directory to write playlist_mix_01.wav, ... (default: project root)",
    )
    parser.add_argument(
        "--prefix",
        default="playlist_mix",
        help="Output file prefix (default: playlist_mix)",
    )
    parser.add_argument(
        "--from-mix",
        type=int,
        default=1,
        metavar="N",
        help="Only create mixes from this number (1-based, default 1)",
    )
    parser.add_argument(
        "--to-mix",
        type=int,
        default=None,
        metavar="N",
        help="Only create mixes up to this number (1-based, default: all)",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching playlist...")
    entries = fetch_playlist_video_ids(args.playlist_url)
    n = len(entries)
    if n < 2:
        print("✗ Playlist has fewer than 2 songs. Need at least 2 to create a mix.")
        sys.exit(1)
    max_mix = n - 1
    if args.to_mix is None:
        args.to_mix = max_mix
    # i is 0-based pair index (pair i = song i → song i+1 = mix number i+1)
    start_i = max(0, args.from_mix - 1)
    end_i = min(n - 2, args.to_mix - 1)  # last valid pair index
    if start_i > end_i:
        print(f"✗ No mixes in range [--from-mix {args.from_mix}, --to-mix {args.to_mix}].")
        sys.exit(1)
    print(f"✓ Found {n} songs. Creating mixes {args.from_mix}–{args.to_mix} (of 1–{max_mix}).\n")

    song_a_path = TEMP_DIR / "song_a.wav"
    song_b_path = TEMP_DIR / "song_b.wav"

    sys.path.insert(0, str(PROJECT_ROOT))
    import soundfile as sf
    from src.smart_mixer import SmartMixer

    mixer = SmartMixer()
    for i in range(start_i, end_i + 1):
        url_a = f"https://www.youtube.com/watch?v={entries[i]['id']}"
        url_b = f"https://www.youtube.com/watch?v={entries[i + 1]['id']}"
        title_a = entries[i]["title"][:40]
        title_b = entries[i + 1]["title"][:40]
        print("=" * 60)
        print(f"Mix {i + 1}/{n - 1}: {title_a} → {title_b}")
        print("=" * 60)

        for p in (song_a_path, song_b_path):
            if p.exists():
                p.unlink()

        try:
            download_youtube_audio(
                url_a, song_a_path, args.duration, from_end=True
            )
            download_youtube_audio(
                url_b, song_b_path, args.duration, from_end=False
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"✗ Skip mix {i + 1}: {e}")
            continue

        print("\nCreating mix...")
        try:
            mixed = mixer.create_superhuman_mix(
                str(song_a_path),
                str(song_b_path),
                transition_duration=16.0,
                creativity_level=0.6,
                optimize_quality=True,
            )
        except Exception as e:
            print(f"✗ Mix failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        out_name = f"{args.prefix}_{i + 1:02d}.wav"
        out_path = args.output_dir / out_name
        sf.write(out_path, mixed, mixer.sr)
        print(f"✅ Saved: {out_path}")
        print(f"   Duration: {len(mixed) / mixer.sr:.1f}s\n")

        if song_a_path.exists():
            song_a_path.unlink()
        if song_b_path.exists():
            song_b_path.unlink()

    print("=" * 60)
    print("Done. Play playlist_mix_01.wav, 02, 03, ... in order to hear the full mix.")


if __name__ == "__main__":
    main()
