"""
config.py — Centralized hyperparameter and experiment configuration.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BTSPConfig:
    # ------- Eligibility Trace -------
    trace_decay: float = 0.9          # λ: decay factor for eligibility trace
    trace_dim: int = 768              # dimensionality of trace / embedding

    # ------- Plateau Detection -------
    plateau_threshold: float = 0.5    # surprise norm threshold to fire a plateau
    plateau_cooldown: int = 4         # min frames between consecutive plateau events

    # ------- Memory Bank -------
    memory_size: int = 4096           # max number of memory slots
    memory_dim: int = 768             # must match trace_dim
    memory_top_k: int = 8            # top-k slots retrieved per query
    memory_write_momentum: float = 0.0  # 0 = hard write; >0 = EMA update of slot

    # ------- Readout -------
    readout_hidden: int = 512


@dataclass
class ModelConfig:
    # ------- Backbone -------
    backbone: str = "MCG-NJU/videomae-base"   # HuggingFace VideoMAE checkpoint
    freeze_backbone: bool = False
    embed_dim: int = 768

    # ------- Predictive Head -------
    pred_hidden: int = 512
    pred_layers: int = 2

    # ------- BTSP -------
    btsp: BTSPConfig = field(default_factory=BTSPConfig)

    # ------- Fusion -------
    fusion: str = "concat"   # "concat" | "add" | "gate"


@dataclass
class DataConfig:
    ego4d_root: str = "/vision/group/ego4d_full_frames/"
    annotation_json: str = "/vision/group/ego4d/v2/annotations/fho_lta_train.json"
    clip_len: int = 16          # frames per clip fed to backbone
    frame_stride: int = 15       # temporal stride when sampling frames
    img_size: int = 224
    num_workers: int = 8
    pin_memory: bool = True


@dataclass
class TrainConfig:
    output_dir: str = "./runs/btsp_exp"
    seed: int = 42

    # ------- Optimisation -------
    epochs: int = 30
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    clip_grad: float = 1.0

    # ------- Misc -------
    log_every: int = 50          # steps
    eval_every: int = 1          # epochs
    save_every: int = 5          # epochs
    use_amp: bool = True
    resume: Optional[str] = None


@dataclass
class EvalConfig:
    few_shot_k: list = field(default_factory=lambda: [1, 5, 10])
    action_recognition_split: str = "val"


@dataclass
class ExperimentConfig:
    name: str = "btsp_videomae_ego4d"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)