#!/usr/bin/env python3
"""Mix the entire UTOPIA album by Travis Scott — generates transitions between every consecutive track."""
import subprocess
import sys
import shutil
import time
from pathlib import Path

OUTPUT_DIR = Path("utopia_mix")
OUTPUT_DIR.mkdir(exist_ok=True)

# UTOPIA tracklist in album order
TRACKS = [
    "HYAENA Travis Scott",
    "THANK GOD Travis Scott",
    "MODERN JAM Travis Scott",
    "MY EYES Travis Scott",
    "GOD'S COUNTRY Travis Scott",
    "SIRENS Travis Scott",
    "MELTDOWN Travis Scott",
    "FE!N Travis Scott",
    "DELRESTO (ECHOES) Travis Scott",
    "I KNOW ? Travis Scott",
    "TOPIA TWINS Travis Scott",
    "CIRCUS MAXIMUS Travis Scott",
    "PARASAIL Travis Scott",
    "SKITZO Travis Scott",
    "LOST FOREVER Travis Scott",
    "LOOOVE Travis Scott",
    "K-POP Travis Scott",
    "TELEKINESIS Travis Scott",
    "TIL FURTHER NOTICE Travis Scott",
]

def short_name(track: str) -> str:
    """Convert track query to a filename-safe short name."""
    return track.split(" Travis")[0].lower().replace(" ", "_").replace(".", "").replace("'", "").replace("!", "").replace("?", "").replace("(", "").replace(")", "")

def main():
    total = len(TRACKS) - 1
    print("=" * 60)
    print(f"⛰️  UTOPIA FULL ALBUM MIX — {total} transitions")
    print("=" * 60)
    
    results = []
    start_all = time.time()
    
    for i in range(total):
        song_a = TRACKS[i]
        song_b = TRACKS[i + 1]
        idx = f"{i+1:02d}"
        name_a = short_name(song_a)
        name_b = short_name(song_b)
        out_name = f"{idx}_{name_a}_to_{name_b}.wav"
        out_path = OUTPUT_DIR / out_name
        
        # Skip if already done (resume capability)
        if out_path.exists() and out_path.stat().st_size > 100_000:
            print(f"\n⏭  [{idx}/{total}] {name_a} → {name_b} — already exists, skipping")
            results.append((out_name, "skipped"))
            continue
        
        print(f"\n{'='*60}")
        print(f"🎵 [{idx}/{total}] {song_a} → {song_b}")
        print(f"{'='*60}")
        
        t0 = time.time()
        try:
            cmd = [
                sys.executable, "create_mix_from_youtube.py",
                song_a, song_b,
                "--transition", "84",
            ]
            result = subprocess.run(cmd, capture_output=False, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"  ✗ Mix failed with exit code {result.returncode}")
                results.append((out_name, "failed"))
                continue
            
            # Find the generated mix file (ai_dj_mix_*.wav in cwd)
            cwd = Path(".")
            mix_files = sorted(cwd.glob("ai_dj_mix_*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)
            
            if mix_files:
                latest = mix_files[0]
                shutil.move(str(latest), str(out_path))
                elapsed = time.time() - t0
                size_mb = out_path.stat().st_size / 1024 / 1024
                print(f"  ✓ Saved: {out_path} ({size_mb:.1f} MB, {elapsed:.0f}s)")
                results.append((out_name, "success"))
            else:
                print(f"  ✗ No output file found")
                results.append((out_name, "no_output"))
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timed out after 10 minutes")
            results.append((out_name, "timeout"))
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append((out_name, "error"))
    
    total_time = time.time() - start_all
    
    print("\n" + "=" * 60)
    print("⛰️  UTOPIA MIX — RESULTS")
    print("=" * 60)
    success = sum(1 for _, s in results if s in ("success", "skipped"))
    print(f"  ✅ {success}/{total} transitions generated")
    print(f"  ⏱  Total time: {total_time/60:.1f} minutes")
    print(f"  📂 Output: {OUTPUT_DIR.resolve()}")
    print()
    for name, status in results:
        icon = "✅" if status in ("success", "skipped") else "❌"
        print(f"  {icon} {name} — {status}")
    print()

if __name__ == "__main__":
    main()
