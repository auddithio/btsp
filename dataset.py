"""
dataset.py — Ego4D continuous video stream dataset.

Unlike clip-based datasets, we treat each Ego4D recording as a long,
unsegmented stream and yield fixed-length chunks sequentially.
This is the regime where BTSP is algorithmically relevant:
the model must accumulate memory across chunks within the same video.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.io import read_image

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
# Clip sampler — yields (clip_tensor, label, is_new_video) tuples
# ---------------------------------------------------------------------------

class Ego4DContinuousDataset(Dataset):
    """
    Treats each Ego4D video as a contiguous stream.
    Each __getitem__ call returns one chunk of `clip_len` frames, sampled
    with stride `frame_stride`, along with its action label and a flag
    indicating whether this chunk starts a new video (so the caller can
    reset memory state).

    Args:
        cfg:        DataConfig
        split:      "train" | "val" | "test"
        transform:  optional override
    """

    def __init__(
        self,
        cfg: DataConfig,
        split: str = "train",
        transform: Optional[T.Compose] = None,
    ):
        self.cfg = cfg
        self.split = split
        self.transform = transform or build_transforms(cfg.img_size, train=(split == "train"))

        self.clips = self._load_annotations()
        print(f"\nLoaded {len(self.clips)} clips from {len(set(c['video_uid'] for c in self.clips))} videos.\n") 

    def _round_down_to_stride(self, frame_num: int) -> int:
        """Round down frame_num to nearest multiple of frame_stride."""
        return (frame_num // self.cfg.frame_stride) * self.cfg.frame_stride  

    def _load_annotations(self) -> List[dict]:
        """
        Parse Ego4D annotations using the videos -> annotated_intervals schema.
        Returns a flat list of clip descriptors sorted by (video_uid, start_frame).
        """
        with open(self.cfg.annotation_json) as f:
            data = json.load(f)

        clips = []

        print(f"Parsing annotations from {self.cfg.annotation_json}...")
        print(f"Found {len(data.get('clips', []))} videos in the annotation file.")
        print(f"Keys for each video entry: {data.get('clips', [{}])[0].keys() if data.get('clips') else 'N/A'}")


        for clip in data.get("clips", []):
            uid = clip["clip_uid"]
            frame_dir = Path(self.cfg.ego4d_root) / uid
            if not frame_dir.exists():
                raise FileNotFoundError(f"Frame directory {frame_dir} does not exist for clip {uid}.")

            clips.append({
                "clip_uid":   uid,
                "frame_dir":   str(frame_dir),
                "start_frame": clip["interval_start_frame"],
                "end_frame":   clip["interval_end_frame"],
                "label":       clip["verb_label"],   # swap for a real label field if available
            })

        # Sort so same-video chunks are contiguous
        clips.sort(key=lambda x: (x["clip_uid"], x["start_frame"]))

        # Mark which clips start a new video
        prev_uid = None
        for c in clips:
            c["is_new_video"] = c["clip_uid"] != prev_uid
            prev_uid = c["clip_uid"]

        return clips

    def _load_frames(self, frame_dir: str, start_frame: int, end_frame: int) -> torch.Tensor:
        """
        Load `clip_len` JPG frames sampled from [start_frame, end_frame].
        Returns tensor of shape (T, C, H, W).
        """
        print(f"Loading frames from {frame_dir} [{start_frame}, {end_frame}]...")
        indices = list(range(start_frame, end_frame, self.cfg.frame_stride))[: self.cfg.clip_len]

        # Pad if the interval is shorter than clip_len
        while len(indices) < self.cfg.clip_len:
            indices.append(indices[-1])

        frames = [read_image(os.path.join(frame_dir, f"frame_{i:06d}.jpg")) for i in indices]
        return self.transform(torch.stack(frames))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        frames = self._load_frames(clip["frame_dir"], 
                                   self._round_down_to_stride(clip["start_frame"]), 
                                   self._round_down_to_stride(clip["end_frame"]))
        return {
            "pixel_values": frames,            # (T, C, H, W)
            "label": torch.tensor(clip["label"], dtype=torch.long),
            "video_uid": clip["video_uid"],
            "is_new_video": clip["is_new_video"],
        }


# ---------------------------------------------------------------------------
# Collate function — handle variable-length within batch
# ---------------------------------------------------------------------------

def collate_fn(batch: List[dict]) -> dict:
    pixel_values = torch.stack([b["pixel_values"] for b in batch])  # (B, T, C, H, W)
    labels = torch.stack([b["label"] for b in batch])
    is_new = [b["is_new_video"] for b in batch]
    uids = [b["video_uid"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "is_new_video": is_new,
        "video_uids": uids,
    }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(cfg: DataConfig, split: str, shuffle: bool = True) -> DataLoader:
    dataset = Ego4DContinuousDataset(cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=1,                   # actual batching handled in trainer
        shuffle=shuffle and (split == "train"),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
        prefetch_factor=2,
    )