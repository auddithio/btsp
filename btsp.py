"""
btsp.py — Core Behavioral Time-Scale Synaptic Plasticity (BTSP) components.

Three sub-modules:
  1. EligibilityTrace   – running decaying trace over frame embeddings
  2. PlateauDetector    – fires a sparse binary event when prediction surprise is high
  3. BTSPMemoryBank     – content-addressable memory written on plateau events,
                          read via cosine-similarity lookup (no gradients through writes)

Reference:
  Wu & Maass, "A simple model for BTSP provides content addressable memory
  with binary synapses and one-shot learning." Nat Commun 16, 342 (2025).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import BTSPConfig


# ---------------------------------------------------------------------------
# 1. Eligibility Trace
# ---------------------------------------------------------------------------

class EligibilityTrace(nn.Module):
    """
    Maintains a per-sample exponentially decaying eligibility trace.

        e_t = λ * e_{t-1} + z_t

    The trace integrates frame embeddings over the behavioral timescale,
    providing a smeared history that captures temporal context beyond a
    single frame — analogous to the dendritic trace in BTSP neurons.
    """

    def __init__(self, cfg: BTSPConfig):
        super().__init__()
        self.decay = cfg.trace_decay
        self.dim = cfg.trace_dim

    @torch.no_grad()
    def forward(
        self,
        z: torch.Tensor,          # (B, D) current frame embedding
        trace: torch.Tensor,      # (B, D) previous trace state
    ) -> torch.Tensor:            # (B, D) updated trace
        return self.decay * trace + z

    def init_trace(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.dim, device=device)


# ---------------------------------------------------------------------------
# 2. Plateau Detector
# ---------------------------------------------------------------------------

class PlateauDetector(nn.Module):
    """
    Fires a sparse binary plateau signal when prediction error exceeds a threshold.

    Surprise is measured as the L2 norm of the difference between the
    predicted next embedding z'_{t+1} and the actual embedding z_{t+1}.
    A cooldown prevents dense plateau firing and mimics the refractory
    period of dendritic plateau potentials.

    Returns:
        plateau_mask: (B,) bool tensor, True where a plateau fires
        surprise:     (B,) float tensor of raw surprise magnitudes
    """

    def __init__(self, cfg: BTSPConfig):
        super().__init__()
        self.threshold = cfg.plateau_threshold
        self.cooldown = cfg.plateau_cooldown
        # Per-sample cooldown counter (not a parameter — managed externally)

    @torch.no_grad()
    def forward(
        self,
        z_pred: torch.Tensor,           # (B, D) predicted embedding
        z_actual: torch.Tensor,         # (B, D) actual embedding
        cooldown_counter: torch.Tensor, # (B,) int tensor, frames since last plateau
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (plateau_mask, surprise, updated_cooldown_counter).
        """
        surprise = torch.norm(z_actual - z_pred, dim=-1)        # (B,)
        threshold_exceeded = surprise > self.threshold           # (B,) bool
        off_cooldown = cooldown_counter >= self.cooldown         # (B,) bool
        plateau_mask = threshold_exceeded & off_cooldown         # (B,) bool

        # Update cooldown: reset to 0 where plateau fired, else increment
        new_counter = torch.where(
            plateau_mask,
            torch.zeros_like(cooldown_counter),
            cooldown_counter + 1,
        )
        return plateau_mask, surprise, new_counter


# ---------------------------------------------------------------------------
# 3. BTSP Memory Bank
# ---------------------------------------------------------------------------

class BTSPMemoryBank(nn.Module):
    """
    A fixed-size, content-addressable memory bank.

    Writes (no gradient):
        On a plateau event, the current eligibility trace is written into
        the next available slot (or overwrites the least-recently-used slot
        when the bank is full — simple circular buffer).

    Reads (with gradient):
        Given a query embedding, retrieve the top-k most similar memory
        slots via cosine similarity and return a weighted sum.

    No gradients flow through write operations, enforcing the separation
    between slow (SGD) and fast (event-gated) learning pathways.
    """

    def __init__(self, cfg: BTSPConfig):
        super().__init__()
        self.size = cfg.memory_size
        self.dim = cfg.memory_dim
        self.top_k = cfg.memory_top_k
        self.write_momentum = cfg.memory_write_momentum

        # Memory slots are NOT parameters; they live as buffers.
        self.register_buffer("memory", torch.zeros(cfg.memory_size, cfg.memory_dim))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("filled_slots", torch.tensor(0, dtype=torch.long))

        # Optional learned key projector (can be identity)
        self.key_proj = nn.Linear(cfg.memory_dim, cfg.memory_dim, bias=False)
        nn.init.eye_(self.key_proj.weight)  # start as identity

    @torch.no_grad()
    def write(
        self,
        trace: torch.Tensor,      # (B, D) eligibility traces
        plateau_mask: torch.Tensor,  # (B,) bool
    ) -> int:
        """
        Write traces of samples where plateau_mask is True into memory slots.
        Returns the number of writes performed.
        """
        traces_to_write = trace[plateau_mask]   # (N_plateau, D)
        n_writes = traces_to_write.size(0)
        if n_writes == 0:
            return 0

        # Normalise before writing for numerically stable cosine retrieval
        traces_normed = F.normalize(traces_to_write, dim=-1)

        for vec in traces_normed:
            idx = int(self.write_ptr.item()) % self.size
            if self.write_momentum > 0 and self.filled_slots > idx:
                self.memory[idx] = (
                    self.write_momentum * self.memory[idx]
                    + (1 - self.write_momentum) * vec
                )
            else:
                self.memory[idx] = vec
            self.write_ptr.add_(1)

        self.filled_slots = torch.clamp(
            self.filled_slots + n_writes, max=self.size
        )
        return n_writes

    def read(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k memory slots for each query.

        Args:
            query: (B, D) query embeddings (gradients flow through here)

        Returns:
            retrieved: (B, D) weighted sum of top-k memory slots
            sim_scores: (B, k) cosine similarities (for auxiliary loss if needed)
        """
        n_filled = int(self.filled_slots.item())
        if n_filled == 0:
            # Memory is empty — return zeros
            return torch.zeros_like(query), torch.zeros(query.size(0), self.top_k, device=query.device)

        active_memory = self.memory[:n_filled]           # (M, D)
        query_proj = self.key_proj(query)                # (B, D)
        query_normed = F.normalize(query_proj, dim=-1)   # (B, D)

        # Cosine similarity: (B, M)
        sim = query_normed @ active_memory.T

        k = min(self.top_k, n_filled)
        top_sim, top_idx = sim.topk(k, dim=-1)          # (B, k)

        # Softmax-weighted retrieval — gradients flow via query_proj / sim
        weights = torch.softmax(top_sim, dim=-1)         # (B, k)
        retrieved_slots = active_memory[top_idx]         # (B, k, D)
        retrieved = (weights.unsqueeze(-1) * retrieved_slots).sum(dim=1)  # (B, D)

        return retrieved, top_sim

    def reset(self):
        """Clear all memory slots (useful at the start of a new video)."""
        self.memory.zero_()
        self.write_ptr.zero_()
        self.filled_slots.zero_()