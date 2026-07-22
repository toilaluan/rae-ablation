from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class REGConfig:
    enabled: bool = False
    model_name: str = "vit_base_patch16_dinov3.lvd1689m"
    feature_dim: int = 768
    register_tokens: int = 4
    token_mode: str = "cls"
    pool_size: int = 4
    repa_block: int = 4
    repa_hidden_size: int = 2048
    cls_predictor: str = "standard"
    cls_head_block: int = 4
    cls_head_hidden_size: int = 2048
    cls_head_depth: int = 2
    cls_head_mlp_ratio: float = 4.0
    cls_loss_weight: float = 0.03
    repa_loss_weight: float = 0.5

    @property
    def num_tokens(self) -> int:
        if self.token_mode == "cls":
            return 1
        if self.token_mode == "cls_spatial":
            return 1 + self.pool_size**2
        raise ValueError(f"unknown REG token mode: {self.token_mode}")


@dataclass
class ModelConfig:
    image_size: int = 256
    patch_size: int = 32
    in_channels: int = 3
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    gradient_checkpointing: bool = False
    variant: str = "SiT-B"
    reg: REGConfig = field(default_factory=REGConfig)

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def model_variant(self) -> str:
        return f"{self.variant}/{self.patch_size}"


@dataclass
class CompileConfig:
    enabled: bool = True
    backend: str = "inductor"
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool = False


@dataclass
class DataConfig:
    repository: str = "benjamin-paine/imagenet-1k-256x256"
    revision: str = "1bd0400450249a7fe90c0aece37d0d03e7ea956a"
    split: str = "train"
    workers: int = 16
    prefetch_factor: int = 4
    max_samples: int | None = None


@dataclass
class OptimizerConfig:
    kind: str = "muon"
    lr: float = 2e-4
    min_lr: float = 2e-5
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95


@dataclass
class TrainConfig:
    epochs: int = 80
    global_batch_size: int = 1024
    micro_batch_size: int = 256
    max_steps: int | None = None
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    seed: int = 0
    timestep_loc: float = 0.0
    timestep_scale: float = 1.0
    horizontal_flip_probability: float = 0.5
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class ValidationConfig:
    interval: int = 5
    periodic_samples: int = 5_000
    final_samples: int = 50_000
    plan_path: str = "validation-plan.pt"
    validate_now: bool = False


@dataclass
class CheckpointConfig:
    directory: str = "checkpoints"
    interval: int = 5
    keep_last_n: int = 8
    save_best_fid: bool = True


@dataclass
class WandbConfig:
    enabled: bool = False
    entity: str | None = None
    project: str = "jit-reg"
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"
    checkpoint_artifacts: bool = False


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def validate(self) -> None:
        m = self.model
        if m.image_size < 1 or m.patch_size < 1:
            raise ValueError("model.image_size and model.patch_size must be positive")
        if m.image_size % m.patch_size:
            raise ValueError("model.image_size must be exactly divisible by model.patch_size")
        if m.in_channels != 3:
            raise ValueError("model.in_channels must be 3 for RGB pixel training")
        if min(m.hidden_size, m.depth, m.num_heads, m.num_classes) < 1 or m.mlp_ratio <= 0:
            raise ValueError("model dimensions must be positive")
        if m.hidden_size % m.num_heads:
            raise ValueError("model.hidden_size must be divisible by model.num_heads")
        if (m.hidden_size // m.num_heads) % 4:
            raise ValueError(
                "model attention head dimension must be divisible by 4 for 2D RoPE"
            )
        if not 0 <= m.class_dropout_prob < 1:
            raise ValueError("model.class_dropout_prob must be in [0, 1)")
        if not m.variant:
            raise ValueError("model.variant cannot be empty")
        reg = m.reg
        if min(
            reg.feature_dim,
            reg.register_tokens,
            reg.pool_size,
            reg.repa_block,
            reg.repa_hidden_size,
            reg.cls_head_hidden_size,
            reg.cls_head_depth,
        ) < 1:
            raise ValueError("model.reg dimensions must be positive")
        if reg.token_mode not in {"cls", "cls_spatial"}:
            raise ValueError("model.reg.token_mode must be cls or cls_spatial")
        if reg.pool_size > 16:
            raise ValueError("model.reg.pool_size cannot exceed the native 16x16 feature grid")
        if reg.cls_head_block < 1:
            raise ValueError("model.reg.cls_head_block must be positive")
        if reg.cls_head_mlp_ratio <= 0:
            raise ValueError("model.reg.cls_head_mlp_ratio must be positive")
        if reg.cls_predictor not in {"standard", "wide_intermediate"}:
            raise ValueError(
                "model.reg.cls_predictor must be standard or wide_intermediate"
            )
        if reg.enabled and reg.repa_block > m.depth:
            raise ValueError("model.reg.repa_block must be within model.depth")
        if reg.cls_predictor == "wide_intermediate":
            if not reg.enabled:
                raise ValueError("wide_intermediate CLS predictor requires REG")
            if not 1 <= reg.cls_head_block < m.depth:
                raise ValueError(
                    "model.reg.cls_head_block must be at least 1 and before model.depth"
                )
        if reg.cls_loss_weight < 0 or reg.repa_loss_weight < 0:
            raise ValueError("model.reg loss weights cannot be negative")
        if self.train.optimizer.kind not in {"adamw", "muon"}:
            raise ValueError("optimizer.kind must be adamw or muon")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        if self.data.workers < 0 or self.data.prefetch_factor < 1:
            raise ValueError("data workers must be non-negative and prefetch_factor positive")
        if self.data.max_samples is not None and self.data.max_samples < 1:
            raise ValueError("data.max_samples must be positive")
        if not self.data.repository or not self.data.revision or not self.data.split:
            raise ValueError("data repository, revision, and split cannot be empty")
        for name, value in (
            ("validation.interval", self.validation.interval),
            ("checkpoint.interval", self.checkpoint.interval),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.validation.periodic_samples > self.validation.final_samples:
            raise ValueError("periodic validation must be a subset of final validation")
        if (
            self.validation.periodic_samples % m.num_classes
            or self.validation.final_samples % m.num_classes
        ):
            raise ValueError("validation sample counts must be class balanced")
        if self.train.global_batch_size < self.train.micro_batch_size:
            raise ValueError("global batch size must be at least the micro batch size")
        if self.train.epochs < 1 or self.train.micro_batch_size < 1:
            raise ValueError("training epochs and batch sizes must be positive")
        if self.train.max_steps is not None and self.train.max_steps < 1:
            raise ValueError("train.max_steps must be positive")
        if not 0 < self.train.ema_decay < 1:
            raise ValueError("train.ema_decay must be in (0, 1)")
        if not math.isfinite(self.train.timestep_loc):
            raise ValueError("train.timestep_loc must be finite")
        if not math.isfinite(self.train.timestep_scale) or self.train.timestep_scale <= 0:
            raise ValueError("train.timestep_scale must be finite and positive")
        if not 0 <= self.train.horizontal_flip_probability <= 1:
            raise ValueError("train.horizontal_flip_probability must be in [0, 1]")
        if self.checkpoint.keep_last_n < 0:
            raise ValueError("checkpoint.keep_last_n cannot be negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> ExperimentConfig:
    """Load a structured OmegaConf config and reject unknown keys."""
    from omegaconf import OmegaConf

    schema = OmegaConf.structured(ExperimentConfig)
    cfg = schema
    if path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_struct(cfg, True)
    result = OmegaConf.to_object(cfg)
    if not isinstance(result, ExperimentConfig):
        raise TypeError("configuration did not resolve to ExperimentConfig")
    result.validate()
    return result
