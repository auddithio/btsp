#!/usr/bin/env python3
"""
diagnose_dataset.py — Audits effective dataset size and reports exactly
what is being dropped and why, before training starts.

Usage:
    python diagnose_dataset.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

EK_FRAMES   = Path("/scr/aunag/ek100_frames")
ANNOTATIONS = Path("/scr/aunag/annotations")
FPS         = 60.0
ANTICIPATION_GAP_SEC = 1.0
OBSERVED_SEC         = 2.0


def audit_split(csv_name: str):
    csv_path = ANNOTATIONS / csv_name
    if not csv_path.exists():
        print(f"  [SKIP] {csv_name} not found\n")
        return

    print(f"\n{'='*60}")
    print(f"  {csv_name}")
    print(f"{'='*60}")

    # Counters
    total_rows       = 0
    drop_no_dir      = 0   # video directory missing entirely
    drop_empty_dir   = 0   # directory exists but no frames
    drop_bad_window  = 0   # anticipation window out of frame range
    drop_gap_frames  = 0   # frames in window have gaps (partial extraction)
    kept             = 0

    # Per-video stats
    frame_cache      = {}
    video_coverage   = defaultdict(lambda: {"total": 0, "kept": 0})
    gap_videos       = set()

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            total_rows += 1
            pid = row["participant_id"]
            vid = row["video_id"]
            frame_dir = EK_FRAMES / pid / vid

            # 1. Directory missing
            if not frame_dir.exists():
                drop_no_dir += 1
                video_coverage[vid]["total"] += 1
                continue

            # 2. Cache available frames
            if vid not in frame_cache:
                existing = sorted(frame_dir.glob("frame_*.jpg"))
                frame_cache[vid] = [int(p.stem.split("_")[1]) for p in existing]

            available = frame_cache[vid]

            # 3. Empty directory
            if not available:
                drop_empty_dir += 1
                video_coverage[vid]["total"] += 1
                continue

            first_f, last_f = available[0], available[-1]
            expected_n = last_f - first_f + 1
            actual_n   = len(available)
            gap_pct    = 100 * (1 - actual_n / expected_n) if expected_n > 0 else 0

            if gap_pct > 1.0:   # >1% frames missing = note this video
                gap_videos.add((vid, gap_pct, actual_n, expected_n))

            # 4. Anticipation window
            action_start = int(row["start_frame"])
            obs_end   = max(first_f, min(action_start - int(ANTICIPATION_GAP_SEC * FPS), last_f))
            obs_start = max(first_f, obs_end - int(OBSERVED_SEC * FPS))

            video_coverage[vid]["total"] += 1

            if obs_end <= obs_start:
                drop_bad_window += 1
                continue

            # 5. Check frames in window actually exist
            window_frames = [f for f in available if obs_start <= f <= obs_end]
            needed = int(OBSERVED_SEC * FPS)
            if len(window_frames) < needed * 0.5:   # less than 50% of expected frames
                drop_gap_frames += 1
                gap_videos.add((vid, gap_pct, actual_n, expected_n))
                continue

            kept += 1
            video_coverage[vid]["kept"] += 1

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n  Total annotation rows:       {total_rows:>7,}")
    print(f"  Kept (usable samples):       {kept:>7,}  ({100*kept/max(1,total_rows):.1f}%)")
    print(f"\n  Dropped — no directory:      {drop_no_dir:>7,}  ({100*drop_no_dir/max(1,total_rows):.1f}%)")
    print(f"  Dropped — empty directory:   {drop_empty_dir:>7,}  ({100*drop_empty_dir/max(1,total_rows):.1f}%)")
    print(f"  Dropped — bad window:        {drop_bad_window:>7,}  ({100*drop_bad_window/max(1,total_rows):.1f}%)")
    print(f"  Dropped — frame gaps:        {drop_gap_frames:>7,}  ({100*drop_gap_frames/max(1,total_rows):.1f}%)")

    # ── Videos with significant frame gaps ──────────────────────────────────
    if gap_videos:
        print(f"\n  Videos with >1% frame gaps ({len(gap_videos)} total):")
        for vid, gap_pct, actual, expected in sorted(gap_videos, key=lambda x: -x[1])[:20]:
            print(f"    {vid:12s}  {actual:6,}/{expected:6,} frames  ({gap_pct:.1f}% missing)")
        if len(gap_videos) > 20:
            print(f"    ... and {len(gap_videos)-20} more")

    # ── Videos with worst coverage ──────────────────────────────────────────
    worst = [(vid, d["kept"], d["total"])
             for vid, d in video_coverage.items()
             if d["total"] > 0 and d["kept"] / d["total"] < 0.5 and d["total"] >= 5]
    if worst:
        print(f"\n  Videos with <50% annotation coverage ({len(worst)} total):")
        for vid, k, t in sorted(worst, key=lambda x: x[1]/x[2])[:15]:
            print(f"    {vid:12s}  {k}/{t} annotations kept  ({100*k/t:.0f}%)")


if __name__ == "__main__":
    for split in ["EPIC_100_train.csv", "EPIC_100_validation.csv"]:
        audit_split(split)
    print("\nDone.")