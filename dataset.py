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
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import decord
from decord import VideoReader, cpu

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

    def _load_annotations(self) -> List[dict]:
        """
        Parse fho_lta annotations for next-action anticipation (K=1).

        Each entry provides an observed clip and the immediately following
        action (verb_label + noun_label) as the prediction target.

        Schema: data["clips"][i] has:
          - clip_uid, video_uid
          - clip_parent_start_sec / clip_parent_end_sec  (observed window)
          - future_actions[0].verb_label, future_actions[0].noun_label (K=1 target)
        """
        with open(self.cfg.annotation_json) as f:
            data = json.load(f)

        clips = []
        for entry in data.get("clips", []):
            uid       = entry["video_uid"]
            clip_uid  = entry["clip_uid"]
            video_path = Path(self.cfg.ego4d_root) / "clips" / f"{clip_uid}.mp4"
            if not video_path.exists():
                continue

            # K=1: take the first future action as the prediction target
            future = entry.get("future_actions", [])
            if not future:
                continue
            verb_label = future[0].get("verb_label", 0)
            noun_label = future[0].get("noun_label", 0)

            clips.append({
                "video_path":  str(video_path),
                "video_uid":   uid,
                "clip_uid":    clip_uid,
                "start_sec":   entry["clip_parent_start_sec"],
                "end_sec":     entry["clip_parent_end_sec"],
                "verb_label":  verb_label,
                "noun_label":  noun_label,
            })

        # Sort so same-video chunks are contiguous
        clips.sort(key=lambda x: (x["video_uid"], x["start_sec"]))

        # Mark which clips start a new video
        prev_uid = None
        for c in clips:
            c["is_new_video"] = c["video_uid"] != prev_uid
            prev_uid = c["video_uid"]

        return clips

    def _load_frames(self, video_path: str, start_sec: float, end_sec: float) -> torch.Tensor:
        """
        Decode `clip_len` frames uniformly sampled from [start_sec, end_sec].
        Returns tensor of shape (T, C, H, W).
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()

        start_f = int(start_sec * fps)
        end_f   = min(int(end_sec * fps), len(vr) - 1)

        total_frames = end_f - start_f
        stride = max(self.cfg.frame_stride, total_frames // self.cfg.clip_len)
        indices = list(range(start_f, end_f, stride))[: self.cfg.clip_len]

        while len(indices) < self.cfg.clip_len:
            indices.append(indices[-1])

        frames = vr.get_batch(indices).asnumpy()                    # (T, H, W, C) uint8
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)       # (T, C, H, W)
        return self.transform(frames)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        frames = self._load_frames(clip["video_path"], clip["start_sec"], clip["end_sec"])
        return {
            "pixel_values": frames,
            "verb_label":   torch.tensor(clip["verb_label"], dtype=torch.long),
            "noun_label":   torch.tensor(clip["noun_label"], dtype=torch.long),
            "video_uid":    clip["video_uid"],
            "is_new_video": clip["is_new_video"],
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[dict]) -> dict:
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    verb_labels  = torch.stack([b["verb_label"]   for b in batch])
    noun_labels  = torch.stack([b["noun_label"]   for b in batch])
    is_new = [b["is_new_video"] for b in batch]
    uids   = [b["video_uid"]    for b in batch]
    return {
        "pixel_values": pixel_values,
        "verb_labels":  verb_labels,
        "noun_labels":  noun_labels,
        "is_new_video": is_new,
        "video_uids":   uids,
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