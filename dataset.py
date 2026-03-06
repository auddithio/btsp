"""
dataset.py — EPIC-KITCHENS-100 action anticipation dataset.

Uses pre-extracted RGB frames (from .tar archives) for fast O(1) random
access during training, avoiding decord seek overhead on MP4s.

Directory structure (after extraction):
  {ek_frames_root}/
    P01/
      P01_101/
        frame_0000000001.jpg
        frame_0000000002.jpg
        ...
    P02/
      ...

Annotations (from epic-kitchens-100-annotations repo):
  {annotations_root}/EPIC_100_train.csv
  {annotations_root}/EPIC_100_validation.csv

Anticipation protocol:
  Observed clip ends τ_a=1.0s before action start; model predicts verb+noun.
"""

import csv
from pathlib import Path
from typing import List, Optional

import torch
import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

from config import DataConfig


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(img_size: int, train: bool) -> T.Compose:
    transforms = [
        T.Resize((img_size, img_size), antialias=True),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if train:
        transforms.insert(1, T.RandomHorizontalFlip())
        transforms.insert(2, T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1))
    return T.Compose(transforms)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EpicKitchensAnticipationDataset(Dataset):
    """
    EPIC-KITCHENS-100 next-action anticipation (K=1).

    Each item is a clip of frames ending τ_a seconds before an annotated
    action starts, with verb_class and noun_class as prediction targets.
    Clips are sorted by (video_id, start_frame) for BTSP streaming continuity.
    """

    ANTICIPATION_GAP_SEC: float = 1.0   # τ_a: standard EK-100 setting
    OBSERVED_SEC: float = 2.0           # duration of observed window
    FPS: float = 60.0                   # EK-100 is 60fps

    def __init__(
        self,
        cfg: DataConfig,
        split: str = "train",
        transform: Optional[T.Compose] = None,
    ):
        self.cfg = cfg
        self.split = split
        self.transform = transform or build_transforms(cfg.img_size, train=(split == "train"))
        self._frame_cache = self._load_frame_cache()
        self.clips = self._load_annotations()

    def _load_frame_cache(self) -> dict:
        """
        Load pre-computed frame cache from disk on rank 0 only, then broadcast
        to all other ranks — avoids N simultaneous JSON reads spiking CPU RAM.
        Run precompute_cache.py once before training to generate the cache file.
        """
        import json
        import os

        cache_path = Path(self.cfg.frame_cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Frame cache not found at {cache_path}. "
                f"Run 'python precompute_cache.py' first."
            )

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Only rank 0 reads from disk; others wait at the barrier
        if local_rank == 0:
            with open(cache_path) as f:
                raw = json.load(f)
            cache = {k.split("/")[1]: v for k, v in raw.items()}
        else:
            cache = None

        # Broadcast from rank 0 to all ranks if running distributed
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                container = [cache]
                dist.broadcast_object_list(container, src=0)
                cache = container[0]
        except Exception:
            pass  # single-GPU — no broadcast needed

        return cache

    def _load_annotations(self) -> List[dict]:
        """
        Parse EPIC_100_{train|validation}.csv.
        Key columns: video_id, start_frame, stop_frame, verb_class, noun_class.
        """
        csv_name = "EPIC_100_train.csv" if self.split == "train" else "EPIC_100_validation.csv"
        csv_path = Path(self.cfg.annotation_json) / csv_name

        clips = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row["video_id"]
                pid = row["participant_id"]

                frame_dir = Path(self.cfg.ego4d_root) / pid / vid
                if not frame_dir.exists():
                    continue

                available = self._frame_cache.get(vid, [])
                if not available:
                    continue

                first_f = available[0]
                last_f  = available[-1]

                action_start_frame = int(row["start_frame"])
                obs_end_frame   = max(first_f, min(
                    action_start_frame - int(self.ANTICIPATION_GAP_SEC * self.FPS),
                    last_f,
                ))
                obs_start_frame = max(first_f, obs_end_frame - int(self.OBSERVED_SEC * self.FPS))

                if obs_end_frame <= obs_start_frame:
                    continue

                clips.append({
                    "video_id":        vid,
                    "frame_dir":       str(frame_dir),
                    "obs_start_frame": obs_start_frame,
                    "obs_end_frame":   obs_end_frame,
                    "verb_label":      int(row["verb_class"]),
                    "noun_label":      int(row["noun_class"]),
                })

        clips.sort(key=lambda x: (x["video_id"], x["obs_start_frame"]))

        prev_vid = None
        for c in clips:
            c["is_new_video"] = c["video_id"] != prev_vid
            prev_vid = c["video_id"]

        return clips

    def _load_frames(self, frame_dir: str, start_frame: int, end_frame: int) -> torch.Tensor:
        """
        Load clip_len frames sampled only from frames that exist on disk.
        Uses the pre-built frame cache to avoid any missing-file errors.
        Returns (T, C, H, W).
        """
        vid = Path(frame_dir).name
        available = self._frame_cache.get(vid, [])
        # Filter to frames within the requested window
        window = [f for f in available if start_frame <= f <= end_frame]

        # Fall back to nearest available frames if window is empty
        if not window:
            window = available

        # Subsample clip_len frames evenly across window
        if len(window) <= self.cfg.clip_len:
            indices = window + [window[-1]] * (self.cfg.clip_len - len(window))
        else:
            step = len(window) / self.cfg.clip_len
            indices = [window[int(i * step)] for i in range(self.cfg.clip_len)]

        frame_dir_path = Path(frame_dir)
        frames = []
        last_good = None
        for i in indices:
            try:
                img = read_image(str(frame_dir_path / f"frame_{i:010d}.jpg"))
                last_good = img
                frames.append(img)
            except RuntimeError:
                # Truncated or corrupt JPEG — reuse the last good frame
                if last_good is not None:
                    frames.append(last_good)
                else:
                    # No good frame yet — defer; will be filled by next good frame
                    frames.append(None)

        # Replace any leading Nones with the first good frame
        first_good = next((f for f in frames if f is not None), None)
        if first_good is None:
            # Entire clip is corrupt — return blank at expected raw size
            blank = torch.zeros(3, 256, 456, dtype=torch.uint8)
            frames = [blank] * self.cfg.clip_len
        else:
            frames = [f if f is not None else first_good for f in frames]

        return self.transform(torch.stack(frames))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        frames = self._load_frames(
            clip["frame_dir"], clip["obs_start_frame"], clip["obs_end_frame"]
        )
        return {
            "pixel_values": frames,
            "verb_label":   torch.tensor(clip["verb_label"], dtype=torch.long),
            "noun_label":   torch.tensor(clip["noun_label"], dtype=torch.long),
            "video_uid":    clip["video_id"],
            "is_new_video": clip["is_new_video"],
        }


# ---------------------------------------------------------------------------
# Collate + DataLoader
# ---------------------------------------------------------------------------

def collate_fn(batch: List[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "verb_labels":  torch.stack([b["verb_label"]   for b in batch]),
        "noun_labels":  torch.stack([b["noun_label"]   for b in batch]),
        "is_new_video": [b["is_new_video"] for b in batch],
        "video_uids":   [b["video_uid"]    for b in batch],
    }


def build_dataloader(cfg: DataConfig, split: str, shuffle: bool = True) -> DataLoader:
    dataset = EpicKitchensAnticipationDataset(cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle and (split == "train"),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
        prefetch_factor=2,
        drop_last=True,   # ensures all ranks get identical batch counts in DDP
    )