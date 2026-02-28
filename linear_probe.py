"""
linear_probe.py — Linear evaluation on top of a frozen BTSP-trained backbone.

Usage:
    python linear_probe.py \
        --checkpoint ./runs/btsp_exp/checkpoint_best.pt \
        --ego4d_root /vision/group/ego4d_full_frames \
        --annotation_json /path/to/labeled_split.json \
        --num_classes 110
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import top_k_accuracy_score
import numpy as np

from config import ExperimentConfig
from model import BTSPVideoTransformer
from dataset import build_dataloader

logger = logging.getLogger(__name__)


def extract_features(
    model: BTSPVideoTransformer,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple:
    """
    Single pass through loader to extract (N, D) embeddings and (N,) labels.
    Model weights are frozen — no grad needed here.
    """
    model.eval()
    all_h, all_labels = [], []
    video_states: Dict[str, dict] = {}

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)
            uids         = batch["video_uids"]
            is_new       = batch["is_new_video"]

            states = []
            for i, uid in enumerate(uids):
                if is_new[i] or uid not in video_states:
                    video_states[uid] = model.init_state(1, device)
                states.append(video_states[uid])

            batch_state = {
                "trace":    torch.cat([s["trace"]    for s in states], dim=0),
                "cooldown": torch.cat([s["cooldown"] for s in states], dim=0),
                "prev_z":   torch.cat([s["prev_z"]   for s in states], dim=0),
            }

            with autocast(enabled=use_amp):
                h, new_state, _ = model(pixel_values, batch_state)

            all_h.append(h.cpu())
            all_labels.append(labels.cpu())

            for i, uid in enumerate(uids):
                video_states[uid] = {k: v[i:i+1].detach() for k, v in new_state.items()}

    return torch.cat(all_h, dim=0), torch.cat(all_labels, dim=0)


def train_probe(
    train_h: torch.Tensor,   # (N, D) pre-extracted features
    train_y: torch.Tensor,   # (N,)
    num_classes: int,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
) -> nn.Linear:
    probe = nn.Linear(train_h.size(1), num_classes).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    dataset = torch.utils.data.TensorDataset(train_h, train_y)
    loader  = DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(1, epochs + 1):
        probe.train()
        total_loss = 0.0
        for h_batch, y_batch in loader:
            h_batch, y_batch = h_batch.to(device), y_batch.to(device)
            loss = F.cross_entropy(probe(h_batch), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"[Probe Epoch {epoch}] loss={total_loss/len(loader):.4f}")

    return probe


@torch.no_grad()
def evaluate_probe(
    probe: nn.Linear,
    val_h: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
) -> Dict:
    probe.eval()
    logits = probe(val_h.to(device)).cpu().numpy()
    labels = val_y.numpy()
    top1 = top_k_accuracy_score(labels, logits, k=1)
    top5 = top_k_accuracy_score(labels, logits, k=5) if logits.shape[-1] >= 5 else float("nan")
    return {"top1": top1, "top5": top5}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      required=True)
    parser.add_argument("--ego4d_root",      required=True)
    parser.add_argument("--annotation_json", required=True)
    parser.add_argument("--num_classes",     type=int, default=110)
    parser.add_argument("--probe_epochs",    type=int, default=20)
    parser.add_argument("--probe_lr",        type=float, default=1e-3)
    parser.add_argument("--output_dir",      type=str, default="./runs/probe")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load frozen backbone ----
    cfg = ExperimentConfig()
    cfg.data.ego4d_root     = args.ego4d_root
    cfg.data.annotation_json = args.annotation_json

    model = BTSPVideoTransformer(cfg.model).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info(f"Loaded backbone from {args.checkpoint} — all weights frozen")

    # ---- Extract features once (cheap — no grad) ----
    train_loader = build_dataloader(cfg.data, "train", shuffle=False)
    val_loader   = build_dataloader(cfg.data, "val",   shuffle=False)

    logger.info("Extracting train features...")
    train_h, train_y = extract_features(model, train_loader, device)
    logger.info("Extracting val features...")
    val_h,   val_y   = extract_features(model, val_loader,   device)

    # ---- Train linear probe ----
    probe = train_probe(train_h, train_y, args.num_classes, device,
                        epochs=args.probe_epochs, lr=args.probe_lr)

    # ---- Evaluate ----
    metrics = evaluate_probe(probe, val_h, val_y, device)
    logger.info(f"Linear probe — top-1: {metrics['top1']:.4f}  top-5: {metrics['top5']:.4f}")

    # ---- Save probe ----
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), out / "probe.pt")
    logger.info(f"Probe saved to {out / 'probe.pt'}")