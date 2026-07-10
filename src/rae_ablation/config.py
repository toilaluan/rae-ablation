from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DepthAttentionConfig:
    enabled: bool = True
    stride: int = 4


@dataclass
class DDTIGConfig:
    encoder_hidden_size: int = 1440
    decoder_hidden_size: int = 2048
    encoder_depth: int = 28
    decoder_depth: int = 2
    encoder_num_heads: int = 20
    decoder_num_heads: int = 16
    mlp_ratio: float = 4.0
    base_model_depth: int = 8
    num_time_tokens: int = 4
    num_class_tokens: int = 8


@dataclass
class ModelConfig:
    architecture: str = "depth_attention_dit"
    in_channels: int = 1024
    hidden_size: int = 1024
    depth: int = 16
    num_heads: int = 16
    mlp_size: int = 4096
    patch_size: int = 1
    base_grid_size: int = 16
    early_exit_block: int = 8
    num_classes: int = 1000
    dropout: float = 0.0
    gradient_checkpointing: bool = True
    depth_attention: DepthAttentionConfig = field(default_factory=DepthAttentionConfig)
    ddt_ig: DDTIGConfig = field(default_factory=DDTIGConfig)


@dataclass
class CompileConfig:
    enabled: bool = True
    backend: str = "inductor"
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool = False


@dataclass
class CacheConfig:
    codec: str = "both"
    directory: str = "cache/imagenet-train"
    split: str = "train"
    shard_size: int = 256
    batch_size: int = 64
    workers: int = 8
    prefetch_factor: int = 4
    output_buffers: int = 3
    max_samples: int | None = None
    verify: bool = True


@dataclass
class OptimizerConfig:
    kind: str = "adamw"
    lr: float = 2e-4
    min_lr: float = 2e-5
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95


@dataclass
class TrainConfig:
    epochs: int = 80
    global_batch_size: int = 1024
    micro_batch_size: int = 16
    max_steps: int | None = None
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    seed: int = 0
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
    interval: int = 1
    keep_last_n: int = 3
    save_best_fid: bool = True


@dataclass
class WandbConfig:
    enabled: bool = False
    entity: str | None = None
    project: str = "rae-ablation"
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"
    checkpoint_artifacts: bool = False


@dataclass
class ExperimentConfig:
    codec: str = "rae"
    cache: CacheConfig = field(default_factory=CacheConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        from .constants import LATENT_SPECS

        return LATENT_SPECS[self.codec]

    def validate(self) -> None:
        m = self.model
        if self.codec not in {"rae", "rae_compact"}:
            raise ValueError("codec must be rae or rae_compact")
        if self.cache.codec not in {"rae", "rae_compact", "both"}:
            raise ValueError("cache.codec must be rae, rae_compact, or both")
        if self.train.optimizer.kind not in {"adamw", "muon"}:
            raise ValueError("optimizer.kind must be adamw or muon")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        if m.architecture not in {"depth_attention_dit", "raev2_ddt_ig"}:
            raise ValueError(
                "model.architecture must be depth_attention_dit or raev2_ddt_ig"
            )
        if m.patch_size != 1:
            raise ValueError("V1 requires latent patch size 1")
        if not 0 <= m.dropout < 1:
            raise ValueError("model.dropout must be in [0, 1)")
        if m.architecture == "depth_attention_dit":
            if m.hidden_size % m.num_heads:
                raise ValueError("model.hidden_size must be divisible by model.num_heads")
            if (m.hidden_size // m.num_heads) % 4:
                raise ValueError("attention head size must be divisible by 4 for 2D RoPE")
            if not 1 <= m.early_exit_block <= m.depth:
                raise ValueError("model.early_exit_block must be in [1, model.depth]")
            if m.depth_attention.stride < 1:
                raise ValueError("depth_attention.stride must be positive")
            if m.depth < 1 or m.hidden_size < 1 or m.num_heads < 1 or m.mlp_size < 1:
                raise ValueError("model dimensions must be positive")
        else:
            ddt = m.ddt_ig
            dimensions = (
                ddt.encoder_hidden_size,
                ddt.decoder_hidden_size,
                ddt.encoder_depth,
                ddt.decoder_depth,
                ddt.encoder_num_heads,
                ddt.decoder_num_heads,
                ddt.num_time_tokens,
                ddt.num_class_tokens,
            )
            if any(value < 1 for value in dimensions) or ddt.mlp_ratio <= 0:
                raise ValueError("model.ddt_ig dimensions and token counts must be positive")
            if not 1 <= ddt.base_model_depth <= ddt.encoder_depth:
                raise ValueError(
                    "model.ddt_ig.base_model_depth must be in [1, encoder_depth]"
                )
            for hidden, heads, name in (
                (
                    ddt.encoder_hidden_size,
                    ddt.encoder_num_heads,
                    "model.ddt_ig.encoder_hidden_size",
                ),
                (
                    ddt.decoder_hidden_size,
                    ddt.decoder_num_heads,
                    "model.ddt_ig.decoder_hidden_size",
                ),
            ):
                if hidden % heads:
                    raise ValueError(f"{name} must be divisible by its number of heads")
                if (hidden // heads) % 4:
                    raise ValueError(
                        f"{name} attention head size must be divisible by 4 for 2D RoPE"
                    )
        if (
            self.cache.batch_size < 1
            or self.cache.shard_size < 1
            or self.cache.shard_size % self.cache.batch_size
        ):
            raise ValueError("cache.shard_size must be a positive multiple of cache.batch_size")
        if self.cache.max_samples is not None and self.cache.max_samples < 1:
            raise ValueError("cache.max_samples must be positive")
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
        if not 0 < self.train.ema_decay < 1:
            raise ValueError("train.ema_decay must be in (0, 1)")
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
