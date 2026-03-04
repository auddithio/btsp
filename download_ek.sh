#!/bin/bash
# copy_missing_frames.sh
# Copies missing EK-2018 training videos into /scr/aunag/ek100_frames
# using parallel rsync for maximum I/O throughput.
#
# Usage: bash copy_missing_frames.sh

SRC=/vision/group/EPIC-KITCHENS/EPIC_KITCHENS_2018.Bingbin/frames_rgb_flow/rgb/train
DST=/scr/aunag/ek100_frames
MISSING_LIST=/tmp/missing_videos.txt
N_JOBS=48  # Leave headroom; rsync is I/O bound, not CPU bound

# ── 1. Find missing videos ───────────────────────────────────────────────────
echo "Checking for missing videos..."
python3 - <<'EOF'
import csv
from pathlib import Path

frames_root = Path("/scr/aunag/ek100_frames")
csv_path    = "/scr/aunag/annotations/EPIC_100_train.csv"
src_root    = Path("/vision/group/EPIC-KITCHENS/EPIC_KITCHENS_2018.Bingbin/frames_rgb_flow/rgb/train")

missing = set()
with open(csv_path) as f:
    for row in csv.DictReader(f):
        pid, vid = row["participant_id"], row["video_id"]
        dst_dir = frames_root / pid / vid
        src_dir = src_root    / pid / vid
        if not dst_dir.exists() and src_dir.exists():
            missing.add((pid, vid))

with open("/tmp/missing_videos.txt", "w") as f:
    for pid, vid in sorted(missing):
        f.write(f"{pid} {vid}\n")

print(f"Found {len(missing)} videos to copy.")
EOF

n=$(wc -l < $MISSING_LIST)
if [ "$n" -eq 0 ]; then
    echo "Nothing to copy — all videos present."
    exit 0
fi
echo "Copying $n videos using $N_JOBS parallel jobs..."

# ── 2. Copy in parallel ──────────────────────────────────────────────────────
# rsync -a: preserves timestamps/permissions
# --inplace: writes directly (faster on NVMe)
# progress tracked via a counter file
cat $MISSING_LIST | parallel -j $N_JOBS --colsep ' ' '
    pid={1}
    vid={2}
    src="'"$SRC"'/$pid/$vid/"
    dst="'"$DST"'/$pid/$vid/"
    mkdir -p $dst
    rsync -a --inplace $src $dst && echo "OK  $vid" || echo "ERR $vid"
' | tee /tmp/copy_log.txt

# ── 3. Summary ───────────────────────────────────────────────────────────────
ok=$(grep  "^OK"  /tmp/copy_log.txt | wc -l)
err=$(grep "^ERR" /tmp/copy_log.txt | wc -l)
echo ""
echo "Done. Copied: $ok  Failed: $err"
if [ "$err" -gt 0 ]; then
    echo "Failed videos:"
    grep "^ERR" /tmp/copy_log.txt
fi

# ── 4. Verify coverage ───────────────────────────────────────────────────────
echo ""
echo "Verifying annotation coverage..."
python3 - <<'EOF'
import csv
from pathlib import Path

frames_root = Path("/scr/aunag/ek100_frames")
csv_path    = "/scr/aunag/annotations/EPIC_100_train.csv"

missing = set()
with open(csv_path) as f:
    for row in csv.DictReader(f):
        pid, vid = row["participant_id"], row["video_id"]
        if not (frames_root / pid / vid).exists():
            missing.add(vid)

if missing:
    print(f"Still missing {len(missing)} videos (likely not in EK-2018 source):")
    for v in sorted(missing): print(f"  {v}")
else:
    print("Full coverage — all annotation videos present on disk.")
EOF