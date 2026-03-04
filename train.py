"""
train.py — Training loop for BTSP-augmented video transformer.

Key design choices:
  - Streaming state: eligibility trace and cooldown counters persist across
    chunks of the same video; reset when a new video begins.
  - Memory bank: shared across the entire training session (not reset per video)
    to simulate open-world, continuous accumulation.
  - Two loss components:
      L_task = cross-entropy on action recognition
      L_pred = MSE predictive loss (trains the surprise signal; small weight)
  - Grad accumulation to handle large effective batch sizes on limited VRAM.

Distributed Training (DDP):
  Launch with torchrun:
    torchrun --nproc_per_node=8 train.py --distributed

  Design notes:
  - BTSP memory bank is NOT synced across GPUs — each rank maintains its own
    local memory, which is intentional: each GPU sees a different shard of the
    video stream, so its memory bank captures a different slice of experience.
    This mirrors the biological intuition of independent hippocampal traces.
  - Gradients ARE synced across ranks via DDP's all-reduce (slow pathway).
  - DistributedSampler ensures non-overlapping data shards per rank.
  - Metrics are all-reduced before logging on rank 0 only.
"""

import os
import time
import random
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from config import ExperimentConfig
from model import BTSPVideoTransformer
from dataset import build_dataloader, EpicKitchensAnticipationDataset, collate_fn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialise NCCL process group. Called once per rank at startup."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def all_reduce_mean(value: float, device: torch.device) -> float:
    """Average a Python scalar across all ranks."""
    if not dist.is_initialized():
        return value
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


def build_distributed_dataloader(cfg, split: str) -> DataLoader:
    dataset = EpicKitchensAnticipationDataset(cfg, split=split)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=(split == "train"),
        drop_last=True,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
        prefetch_factor=2,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, cfg: ExperimentConfig) -> optim.Optimizer:
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        else:
            other_params.append(p)

    return optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.train.lr * 0.1},
            {"params": other_params,    "lr": cfg.train.lr},
        ],
        weight_decay=cfg.train.weight_decay,
    )


def build_scheduler(optimizer: optim.Optimizer, cfg: ExperimentConfig, steps_per_epoch: int):
    total_steps = cfg.train.epochs * steps_per_epoch
    warmup_steps = cfg.train.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: ExperimentConfig, num_verb_classes: int, num_noun_classes: int, distributed: bool = False):
        self.cfg = cfg
        self.distributed = distributed

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # Rank-aware seed so each GPU gets different data augmentation
        set_seed(cfg.train.seed + local_rank)

        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if is_main_process():
            world = dist.get_world_size() if distributed else 1
            logger.info(f"Distributed: {distributed} | World size: {world} | Device: {self.device}")

        # ---- Model ----
        # DDP syncs gradients across ranks (slow pathway).
        # BTSP memory bank buffers are intentionally NOT synced — each rank
        # accumulates independent memories from its own video shard.
        raw_model = BTSPVideoTransformer(cfg.model, num_verb_classes, num_noun_classes).to(self.device)
        if distributed:
            self.model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=False)
        else:
            self.model = raw_model
        self.raw_model = raw_model  # unwrapped reference for state/memory ops

        # ---- Data ----
        if distributed:
            self.train_loader = build_distributed_dataloader(cfg.data, "train")
            self.val_loader   = build_distributed_dataloader(cfg.data, "val")
        else:
            self.train_loader = build_dataloader(cfg.data, "train", shuffle=True)
            self.val_loader   = build_dataloader(cfg.data, "val",   shuffle=False)

        # ---- Optimisation ----
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(
            self.optimizer, cfg, steps_per_epoch=len(self.train_loader)
        )
        self.scaler    = GradScaler(enabled=cfg.train.use_amp)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # ---- Logging (rank 0 only) ----
        self.output_dir = Path(cfg.train.output_dir)
        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.output_dir / "tb")
        else:
            self.writer = None
        self.global_step = 0

        if cfg.train.resume:
            self._load_checkpoint(cfg.train.resume)

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        if self.distributed:
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        n_plateau  = 0.0
        n_verb_correct = n_noun_correct = n_samples = 0
        t0 = time.time()

        video_states: Dict[str, dict] = {}
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            pixel_values = batch["pixel_values"].to(self.device)
            verb_labels  = batch["verb_labels"].to(self.device)
            noun_labels  = batch["noun_labels"].to(self.device)
            uids         = batch["video_uids"]
            is_new       = batch["is_new_video"]
            B            = pixel_values.size(0)

            # ---- Per-video streaming state ----
            states = []
            for i, uid in enumerate(uids):
                if is_new[i] or uid not in video_states:
                    video_states[uid] = self.raw_model.init_state(1, self.device)
                states.append(video_states[uid])

            batch_state = {
                "trace":    torch.cat([s["trace"]    for s in states], dim=0),
                "cooldown": torch.cat([s["cooldown"] for s in states], dim=0),
                "prev_z":   torch.cat([s["prev_z"]   for s in states], dim=0),
            }

            # ---- Forward + combined loss ----
            with autocast(enabled=self.cfg.train.use_amp):
                verb_logits, noun_logits, new_state, aux = self.model(pixel_values, batch_state)
                task_loss = (
                    self.criterion(verb_logits, verb_labels) +
                    self.criterion(noun_logits, noun_labels)
                )
                loss = (task_loss + 0.1 * aux["pred_loss"]) / self.cfg.train.grad_accum_steps

            self.scaler.scale(loss).backward()

            for i, uid in enumerate(uids):
                video_states[uid] = {k: v[i:i+1].detach() for k, v in new_state.items()}

            # ---- Gradient step ----
            if (step + 1) % self.cfg.train.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # ---- Metrics ----
            total_loss     += loss.item() * self.cfg.train.grad_accum_steps
            n_plateau      += aux["plateau_rate"]
            n_verb_correct += (verb_logits.argmax(-1) == verb_labels).sum().item()
            n_noun_correct += (noun_logits.argmax(-1) == noun_labels).sum().item()
            n_samples      += B
            self.global_step += 1

            if is_main_process() and self.global_step % self.cfg.train.log_every == 0:
                avg_loss      = all_reduce_mean(total_loss     / (step + 1),       self.device)
                avg_plateau   = all_reduce_mean(n_plateau      / (step + 1),       self.device)
                avg_verb_acc  = all_reduce_mean(n_verb_correct / max(1, n_samples), self.device)
                avg_noun_acc  = all_reduce_mean(n_noun_correct / max(1, n_samples), self.device)
                lr = self.optimizer.param_groups[-1]["lr"]
                self.writer.add_scalar("train/loss",      avg_loss,     self.global_step)
                self.writer.add_scalar("train/verb_acc",  avg_verb_acc, self.global_step)
                self.writer.add_scalar("train/noun_acc",  avg_noun_acc, self.global_step)
                self.writer.add_scalar("train/plateau_rate", avg_plateau, self.global_step)
                self.writer.add_scalar("train/lr",        lr,           self.global_step)
                logger.info(
                    f"[Ep {epoch} | Step {step}] loss={avg_loss:.4f}  "
                    f"verb={avg_verb_acc:.3f}  noun={avg_noun_acc:.3f}  "
                    f"plateau={avg_plateau:.3f}  lr={lr:.2e}  elapsed={time.time()-t0:.1f}s"
                )

        return {"loss": all_reduce_mean(total_loss / len(self.train_loader), self.device)}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict:
        self.model.eval()
        total_loss = 0.0
        n_verb_correct = n_noun_correct = n_samples = 0
        video_states: Dict[str, dict] = {}

        for batch in self.val_loader:
            pixel_values = batch["pixel_values"].to(self.device)
            verb_labels  = batch["verb_labels"].to(self.device)
            noun_labels  = batch["noun_labels"].to(self.device)
            uids         = batch["video_uids"]
            is_new       = batch["is_new_video"]
            B            = pixel_values.size(0)

            states = []
            for i, uid in enumerate(uids):
                if is_new[i] or uid not in video_states:
                    video_states[uid] = self.raw_model.init_state(1, self.device)
                states.append(video_states[uid])

            batch_state = {
                "trace":    torch.cat([s["trace"]    for s in states], dim=0),
                "cooldown": torch.cat([s["cooldown"] for s in states], dim=0),
                "prev_z":   torch.cat([s["prev_z"]   for s in states], dim=0),
            }

            with autocast(enabled=self.cfg.train.use_amp):
                verb_logits, noun_logits, new_state, aux = self.model(pixel_values, batch_state)
                total_loss += (
                    self.criterion(verb_logits, verb_labels) +
                    self.criterion(noun_logits, noun_labels)
                ).item()

            n_verb_correct += (verb_logits.argmax(-1) == verb_labels).sum().item()
            n_noun_correct += (noun_logits.argmax(-1) == noun_labels).sum().item()
            n_samples      += B

            for i, uid in enumerate(uids):
                video_states[uid] = {k: v[i:i+1].detach() for k, v in new_state.items()}

        loss      = all_reduce_mean(total_loss     / len(self.val_loader),    self.device)
        verb_acc  = all_reduce_mean(n_verb_correct / max(1, n_samples),       self.device)
        noun_acc  = all_reduce_mean(n_noun_correct / max(1, n_samples),       self.device)

        if is_main_process():
            self.writer.add_scalar("val/loss",     loss,     self.global_step)
            self.writer.add_scalar("val/verb_acc", verb_acc, self.global_step)
            self.writer.add_scalar("val/noun_acc", noun_acc, self.global_step)
            logger.info(f"[Val Epoch {epoch}] loss={loss:.4f}  verb={verb_acc:.4f}  noun={noun_acc:.4f}")
        return {"loss": loss, "verb_acc": verb_acc, "noun_acc": noun_acc}

    # ------------------------------------------------------------------
    # Checkpointing (rank 0 only)
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metrics: dict):
        if not is_main_process():
            return
        ckpt = {
            "epoch":       epoch,
            "global_step": self.global_step,
            "model":       self.raw_model.state_dict(),  # save unwrapped weights
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "scaler":      self.scaler.state_dict(),
            "metrics":     metrics,
            "config":      self.cfg,
        }
        path = self.output_dir / f"checkpoint_epoch{epoch:03d}.pt"
        torch.save(ckpt, path)
        logger.info(f"Saved checkpoint → {path}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.raw_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["global_step"]
        if is_main_process():
            logger.info(f"Resumed from {path} (epoch {ckpt['epoch']})")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        if is_main_process():
            logger.info(f"Starting training for {self.cfg.train.epochs} epochs")
        best_val_acc = 0.0

        for epoch in range(1, self.cfg.train.epochs + 1):
            train_metrics = self._train_epoch(epoch)

            val_metrics = {}
            if epoch % self.cfg.train.eval_every == 0:
                val_metrics = self._val_epoch(epoch)
                if val_metrics["acc"] > best_val_acc:
                    best_val_acc = val_metrics["acc"]

            if epoch % self.cfg.train.save_every == 0:
                self._save_checkpoint(
                    epoch,
                    {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
                )

            # Synchronise all ranks before next epoch
            if self.distributed:
                dist.barrier()

        if is_main_process():
            logger.info(f"Training complete. Best val acc: {best_val_acc:.4f}")
            self.writer.close()

        if self.distributed:
            cleanup_distributed()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_verb_classes", type=int, default=97)
    parser.add_argument("--num_noun_classes", type=int, default=300)
    parser.add_argument("--ego4d_root",      type=str,  default="/scr/aunag/ek100_frames")
    parser.add_argument("--output_dir",      type=str,  default="./runs/ek_linear")
    parser.add_argument("--epochs",          type=int,  default=30)
    parser.add_argument("--batch_size",      type=int,  default=8)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--resume",          type=str,  default=None)
    parser.add_argument("--distributed",     action="store_true",
                        help="Enable DDP. Launch via: torchrun --nproc_per_node=8 train.py --distributed")
    args = parser.parse_args()

    if args.distributed:
        setup_distributed()

    cfg = ExperimentConfig()
    cfg.data.ego4d_root       = args.ego4d_root
    cfg.train.output_dir      = args.output_dir
    cfg.train.epochs          = args.epochs
    cfg.train.batch_size      = args.batch_size
    cfg.model.freeze_backbone = args.freeze_backbone
    cfg.train.resume          = args.resume

    trainer = Trainer(cfg, num_verb_classes=args.num_verb_classes,
                      num_noun_classes=args.num_noun_classes, distributed=args.distributed)
    trainer.train()