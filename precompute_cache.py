#!/usr/bin/env python3
"""
precompute_cache.py — Walk all video directories once and save the
list of available frame numbers per video to a JSON cache file.

Run this ONCE before training:
    python precompute_cache.py

Subsequent dataset loads will read the cache instantly instead of
running glob on every video directory.
"""

import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

EK_FRAMES  = Path("/scr/aunag/ek100_frames")
CACHE_PATH = Path("/scr/aunag/frame_cache.json")
N_THREADS  = 48


def scan_video(vid_dir: Path):
    """Return (pid/vid, sorted frame number list) for one video directory."""
    key = f"{vid_dir.parent.name}/{vid_dir.name}"
    frames = sorted(int(p.stem.split("_")[1]) for p in vid_dir.glob("frame_*.jpg"))
    return key, frames


def main():
    # Find all video directories (P*/P*_*/  two levels deep)
    vid_dirs = sorted(EK_FRAMES.glob("P*/P*_*"))
    print(f"Found {len(vid_dirs)} video directories. Scanning frames...")

    cache = {}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        futures = {pool.submit(scan_video, d): d for d in vid_dirs}
        for i, future in enumerate(as_completed(futures), 1):
            key, frames = future.result()
            cache[key] = frames
            if i % 50 == 0 or i == len(vid_dirs):
                elapsed = time.time() - t0
                print(f"  {i}/{len(vid_dirs)}  ({elapsed:.0f}s elapsed)", end="\r")

    print(f"\nScanned {len(cache)} videos in {time.time()-t0:.1f}s")

    # Save
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

    size_mb = CACHE_PATH.stat().st_size / 1e6
    total_frames = sum(len(v) for v in cache.values())
    print(f"Saved to {CACHE_PATH}  ({size_mb:.1f} MB, {total_frames:,} total frames)")


if __name__ == "__main__":
    main()