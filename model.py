"""
model.py — Full BTSP-augmented video transformer.

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │  VideoMAE backbone  →  z_t  (B, D)                          │
  │        │                                                     │
  │        ├──► Predictive Head  →  z'_{t+1}  (B, D)           │
  │        │         │  surprise ↓                              │
  │        │    PlateauDetector  →  plateau_mask  (B,)          │
  │        │                                                     │
  │  EligibilityTrace: e_t = λ·e_{t-1} + z_t                   │
  │        │                                                     │
  │  BTSPMemoryBank ←─ write(e_t) on plateau                    │
  │        │                                                     │
  │        └──► read(z_t)  →  m_t  (B, D)                      │
  │                                                             │
  │  Fusion(z_t, m_t)  →  h_t  →  Task Head                    │
  └──────────────────────────────────────────────────────────────┘

The fast learning pathway (write) is completely gradient-free.
The slow learning pathway (backbone + predictive head + read + task head)
is trained end-to-end via SGD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel, VideoMAEConfig
from typing import Optional, Tuple, Dict

from config import ModelConfig, BTSPConfig
from btsp import EligibilityTrace, PlateauDetector, BTSPMemoryBank


# ---------------------------------------------------------------------------
# Predictive Head
# ---------------------------------------------------------------------------

class PredictiveHead(nn.Module):
    """
    Small MLP that predicts z_{t+1} from z_t.
    Used exclusively to compute the surprise signal for plateau detection.
    """

    def __init__(self, in_dim: int, hidden: int, n_layers: int):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers.append(nn.Linear(d, in_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Fusion Modules
# ---------------------------------------------------------------------------

class ConcatFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([z, m], dim=-1))


class GateFusion(nn.Module):
    """Learned sigmoid gate controls how much memory contributes."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([z, m], dim=-1))
        return g * m + (1 - g) * z


class AddFusion(nn.Module):
    def forward(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return z + m


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class BTSPVideoTransformer(nn.Module):
    """
    VideoMAE backbone augmented with a BTSP fast-memory pathway.

    Usage pattern (streaming, frame-by-frame or chunk-by-chunk):

        state = model.init_state(batch_size, device)
        for chunk in video_stream:
            logits, state, aux = model(chunk, state)
            loss = criterion(logits, labels) + aux["pred_loss"]
            loss.backward()
    """

    def __init__(self, cfg: ModelConfig, num_verb_classes: int, num_noun_classes: int):
        super().__init__()
        self.cfg = cfg
        btsp_cfg = cfg.btsp

        # ---- Backbone ----
        self.backbone = VideoMAEModel.from_pretrained(cfg.backbone)
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # ---- Predictive Head ----
        self.pred_head = PredictiveHead(
            in_dim=cfg.embed_dim,
            hidden=cfg.pred_hidden,
            n_layers=cfg.pred_layers,
        )

        # ---- BTSP Components ----
        self.trace   = EligibilityTrace(btsp_cfg)
        self.plateau = PlateauDetector(btsp_cfg)
        self.memory  = BTSPMemoryBank(btsp_cfg)

        # ---- Fusion ----
        fusion_map = {"concat": ConcatFusion, "gate": GateFusion, "add": AddFusion}
        fusion_cls = fusion_map[cfg.fusion]
        self.fusion = fusion_cls(cfg.embed_dim) if cfg.fusion != "add" else AddFusion()

        # ---- Separate verb and noun heads ----
        self.verb_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, num_verb_classes),
        )
        self.noun_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, num_noun_classes),
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def init_state(self, batch_size: int, device: torch.device) -> Dict:
        """Initialise per-video running state."""
        return {
            "trace": self.trace.init_trace(batch_size, device),
            "cooldown": torch.zeros(batch_size, dtype=torch.long, device=device),
            "prev_z": torch.zeros(batch_size, self.cfg.embed_dim, device=device),
        }

    def reset_memory(self):
        """Clear the memory bank (call between unrelated videos)."""
        self.memory.reset()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run VideoMAE backbone on a clip and pool to a single (B, D) vector.

        Args:
            pixel_values: (B, T, C, H, W)  — standard VideoMAE format

        Returns:
            z: (B, D) clip-level embedding
        """
        out = self.backbone(pixel_values=pixel_values)
        # out.last_hidden_state: (B, num_patches, D)  — mean-pool over patches
        z = out.last_hidden_state.mean(dim=1)
        return z

    def forward(
        self,
        pixel_values: torch.Tensor,    # (B, T, C, H, W)
        state: Dict,
        labels: Optional[torch.Tensor] = None,  # (B,) action class
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Args:
            pixel_values: current video chunk
            state: running state dict (trace, cooldown, prev_z)
            labels: optional ground-truth for computing task loss

        Returns:
            logits:   (B, num_classes)
            state:    updated running state
            aux:      dict with "pred_loss", "plateau_rate", "n_writes", "surprise"
        """
        # ---- 1. Backbone encoding ----
        z = self.encode(pixel_values)     # (B, D)

        # ---- 2. Predictive head — surprise against *current* z ----
        z_pred = self.pred_head(state["prev_z"])   # prediction made at t-1
        pred_loss = F.mse_loss(z_pred, z.detach()) # auxiliary slow-pathway loss

        # ---- 3. Plateau detection (no grad) ----
        plateau_mask, surprise, new_cooldown = self.plateau(
            z_pred.detach(), z.detach(), state["cooldown"]
        )

        # ---- 4. Eligibility trace update (no grad) ----
        new_trace = self.trace(z.detach(), state["trace"])

        # ---- 5. Memory write on plateau (no grad) ----
        n_writes = self.memory.write(new_trace, plateau_mask)

        # ---- 6. Memory read (gradients flow through query projection) ----
        retrieved, sim_scores = self.memory.read(z)

        # ---- 7. Fusion ----
        h = self.fusion(z, retrieved)    # (B, D)

        # ---- 8. Anticipation heads ----
        verb_logits = self.verb_head(h)  # (B, num_verb_classes)
        noun_logits = self.noun_head(h)  # (B, num_noun_classes)

        # ---- Update state ----
        new_state = {
            "trace":    new_trace,
            "cooldown": new_cooldown,
            "prev_z":   z.detach(),
        }

        aux = {
            "pred_loss":    pred_loss,
            "plateau_rate": plateau_mask.float().mean().item(),
            "n_writes":     n_writes,
            "surprise":     surprise.mean().item(),
        }

        return verb_logits, noun_logits, new_state, aux


# ---------------------------------------------------------------------------
# Baseline — identical architecture WITHOUT BTSP (ablation)
# ---------------------------------------------------------------------------

class BaselineVideoTransformer(nn.Module):
    """
    Gradient-only baseline: VideoMAE + classifier, no BTSP pathway.
    Used for ablation comparisons in the paper.
    """

    def __init__(self, cfg: ModelConfig, num_classes: int):
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained(cfg.backbone)
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor, labels=None):
        out = self.backbone(pixel_values=pixel_values)
        z = out.last_hidden_state.mean(dim=1)
        return self.classifier(z), {}, {"pred_loss": torch.tensor(0.0)}