from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 11
FLUX2_SPATIAL_COMPRESSION = 16
FLUX2_LATENT_CHANNELS = 128


def _positive(name: str, *values: int | float) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} must be positive")


@dataclass
class ModelConfig:
    image_size: int = 256
    in_channels: int = FLUX2_LATENT_CHANNELS
    patch_size: int = 2
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    gradient_checkpointing: bool = False
    variant: str = "SiT-B/2"

    @property
    def input_size(self) -> int:
        return self.image_size // FLUX2_SPATIAL_COMPRESSION

    @property
    def grid_size(self) -> int:
        return self.input_size // self.patch_size

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self.in_channels, self.input_size, self.input_size

    @property
    def model_variant(self) -> str:
        return f"{self.variant}-flux2"

    def validate(self) -> None:
        _positive(
            "model dimensions",
            self.image_size,
            self.in_channels,
            self.patch_size,
            self.hidden_size,
            self.depth,
            self.num_heads,
            self.mlp_ratio,
            self.num_classes,
        )
        if self.in_channels != FLUX2_LATENT_CHANNELS:
            raise ValueError(f"model.in_channels must be {FLUX2_LATENT_CHANNELS}")
        if self.image_size % FLUX2_SPATIAL_COMPRESSION:
            raise ValueError("model.image_size must be divisible by 16 for FLUX.2")
        if self.input_size % self.patch_size:
            raise ValueError("FLUX.2 latent size must be divisible by model.patch_size")
        if self.hidden_size % self.num_heads:
            raise ValueError("model.hidden_size must be divisible by model.num_heads")
        if (self.hidden_size // self.num_heads) % 4:
            raise ValueError("attention head dimension must be divisible by 4 for 2D RoPE")
        if not 0 <= self.class_dropout_prob < 1:
            raise ValueError("model.class_dropout_prob must be in [0, 1)")
        if not self.variant:
            raise ValueError("model.variant cannot be empty")


@dataclass
class SIGRegConfig:
    clean_forward: str = "truncated"
    weight: float = 0.1
    projections: int = 1024
    knots: int = 17
    max_frequency: float = 3.0
    projection_chunk_size: int = 128


@dataclass
class ObjectiveConfig:
    kind: str = "vanilla"
    mixed_token_probability: float = 0.25
    align_layers: list[int] = field(default_factory=list)
    alignment_weight: float = 0.8
    projector_hidden_size: int = 1536
    ema_decay: float = 0.9999
    sigreg: SIGRegConfig = field(default_factory=SIGRegConfig)

    @property
    def uses_dual_timestep(self) -> bool:
        return self.kind != "vanilla"

    @property
    def uses_alignment(self) -> bool:
        return self.kind in {"ema", "sigreg"}

    def validate(self, depth: int) -> None:
        if self.kind not in {"vanilla", "dual_timestep", "ema", "sigreg"}:
            raise ValueError("objective.kind must be vanilla, dual_timestep, ema, or sigreg")
        if (
            not math.isfinite(self.mixed_token_probability)
            or not 0 <= self.mixed_token_probability <= 1
        ):
            raise ValueError("objective.mixed_token_probability must be in [0, 1]")
        if self.alignment_weight < 0 or not math.isfinite(self.alignment_weight):
            raise ValueError("objective.alignment_weight must be finite and non-negative")
        if self.uses_alignment:
            if not self.align_layers:
                raise ValueError("alignment objectives require objective.align_layers")
            if self.align_layers != sorted(set(self.align_layers)):
                raise ValueError("objective.align_layers must be sorted and unique")
            if any(
                isinstance(layer, bool) or not 1 <= layer <= depth for layer in self.align_layers
            ):
                raise ValueError("objective.align_layers must be within model.depth")
            _positive("objective.projector_hidden_size", self.projector_hidden_size)
        elif self.align_layers:
            raise ValueError("objective.align_layers is only valid for ema or sigreg")
        if not 0 < self.ema_decay < 1:
            raise ValueError("objective.ema_decay must be in (0, 1)")
        sigreg = self.sigreg
        if self.kind != "sigreg" and sigreg != SIGRegConfig():
            raise ValueError("objective.sigreg options are only valid for kind=sigreg")
        if sigreg.clean_forward not in {"truncated", "full"}:
            raise ValueError("objective.sigreg.clean_forward must be truncated or full")
        if not math.isfinite(sigreg.weight) or sigreg.weight < 0:
            raise ValueError("objective.sigreg.weight must be finite and non-negative")
        _positive(
            "SIGReg dimensions",
            sigreg.projections,
            sigreg.knots,
            sigreg.max_frequency,
            sigreg.projection_chunk_size,
        )


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
class VAECacheConfig:
    enabled: bool = False
    directory: str = "data/latents"
    batch_size: int = 128
    log_interval: int = 10


@dataclass
class VAEConfig:
    model_name: str = "black-forest-labs/FLUX.2-klein-4B"
    revision: str = "5e67da950fce4a097bc150c22958a05716994cea"
    subfolder: str = "vae"
    micro_batch_size: int = 16
    cache: VAECacheConfig = field(default_factory=VAECacheConfig)


@dataclass
class OptimizerConfig:
    kind: str = "adamw"
    lr: float = 2e-4
    min_lr: float = 2e-5
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95


@dataclass
class TrainConfig:
    epochs: int = 80
    global_batch_size: int = 1024
    max_steps: int | None = None
    grad_clip: float = 1.0
    seed: int = 0
    timestep_loc: float = 0.0
    timestep_scale: float = 1.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class SamplingConfig:
    num_samples: int = 50_000
    batch_size: int = 128
    sampling_steps: int = 50
    solver: str = "euler"
    guidance_scale: float = 3.0
    timestep_shift: float = 1.0
    cfg_start: float = 0.0
    cfg_end: float = 1.0
    log_interval: int = 10


@dataclass
class EvaluationConfig:
    reference_npz: str = "data/VIRTUAL_imagenet256_labeled.npz"
    batch_size: int = 128
    device: str = "auto"
    log_interval: int = 10


@dataclass
class ValidationConfig:
    interval: int = 5
    periodic_samples: int = 10_000
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
    project: str = "self-flow"
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"
    checkpoint_artifacts: bool = False


@dataclass
class ExperimentConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    data: DataConfig = field(default_factory=DataConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version: {CONFIG_SCHEMA_VERSION} is required; migration is unsupported"
            )
        self.model.validate()
        self.objective.validate(self.model.depth)
        if not self.vae.model_name or not self.vae.revision or not self.vae.subfolder:
            raise ValueError("a pinned FLUX.2 VAE model, revision, and subfolder are required")
        _positive(
            "VAE settings",
            self.vae.micro_batch_size,
            self.vae.cache.batch_size,
            self.vae.cache.log_interval,
        )
        if not self.vae.cache.directory:
            raise ValueError("vae.cache.directory cannot be empty")
        if self.train.optimizer.kind not in {"adamw", "muon"}:
            raise ValueError("optimizer.kind must be adamw or muon")
        if self.train.epochs < 1 or self.train.global_batch_size < 1:
            raise ValueError("training epochs and global batch size must be positive")
        if self.train.max_steps is not None and self.train.max_steps < 1:
            raise ValueError("train.max_steps must be positive")
        if not math.isfinite(self.train.timestep_loc):
            raise ValueError("train.timestep_loc must be finite")
        if not math.isfinite(self.train.timestep_scale) or self.train.timestep_scale <= 0:
            raise ValueError("train.timestep_scale must be finite and positive")
        if self.data.workers < 0 or self.data.prefetch_factor < 1:
            raise ValueError("data workers must be non-negative and prefetch_factor positive")
        if self.data.max_samples is not None and self.data.max_samples < 1:
            raise ValueError("data.max_samples must be positive")
        if not self.data.repository or not self.data.revision or not self.data.split:
            raise ValueError("data repository, revision, and split cannot be empty")
        if self.objective.kind == "sigreg" and self.train.global_batch_size < 2:
            raise ValueError("SIGReg requires a global batch size of at least two")
        if self.checkpoint.keep_last_n < 0:
            raise ValueError("checkpoint.keep_last_n cannot be negative")
        for name, value in (
            ("validation.interval", self.validation.interval),
            ("checkpoint.interval", self.checkpoint.interval),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.validation.periodic_samples > self.validation.final_samples:
            raise ValueError("periodic validation must be a subset of final validation")
        if (
            self.validation.periodic_samples % self.model.num_classes
            or self.validation.final_samples % self.model.num_classes
        ):
            raise ValueError("validation sample counts must be class balanced")
        self._validate_sampling()
        if self.evaluation.batch_size < 1 or self.evaluation.log_interval < 1:
            raise ValueError("evaluation batch size and log interval must be positive")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")

    def _validate_sampling(self) -> None:
        sampling = self.sampling
        _positive(
            "sampling counts",
            sampling.num_samples,
            sampling.batch_size,
            sampling.sampling_steps,
            sampling.log_interval,
        )
        if sampling.solver not in {"euler", "heun"}:
            raise ValueError("sampling.solver must be euler or heun")
        if sampling.num_samples % self.model.num_classes:
            raise ValueError("sampling.num_samples must be class balanced")
        if not math.isfinite(sampling.guidance_scale):
            raise ValueError("sampling.guidance_scale must be finite")
        if not math.isfinite(sampling.timestep_shift) or sampling.timestep_shift <= 0:
            raise ValueError("sampling.timestep_shift must be finite and positive")
        if not 0 <= sampling.cfg_start <= sampling.cfg_end <= 1:
            raise ValueError("sampling CFG interval must satisfy 0 <= start <= end <= 1")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_current_schema(values: Any, source: str | Path) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ValueError(f"{source} must contain a configuration object")
    if values.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{source} requires schema_version: {CONFIG_SCHEMA_VERSION}; old artifacts cannot be migrated"
        )
    return values


def _structured_config(values: dict[str, Any]) -> ExperimentConfig:
    from omegaconf import OmegaConf

    try:
        cfg = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), values)
        OmegaConf.set_struct(cfg, True)
        result = OmegaConf.to_object(cfg)
    except Exception as error:
        raise ValueError(
            f"invalid schema-v{CONFIG_SCHEMA_VERSION} configuration: {error}"
        ) from error
    if not isinstance(result, ExperimentConfig):
        raise TypeError("configuration did not resolve to ExperimentConfig")
    result.validate()
    return result


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> ExperimentConfig:
    from omegaconf import OmegaConf

    values: dict[str, Any] = {"schema_version": CONFIG_SCHEMA_VERSION}
    if path is not None:
        values = _require_current_schema(
            OmegaConf.to_container(OmegaConf.load(path), resolve=True), path
        )
    if overrides:
        override_values = OmegaConf.to_container(OmegaConf.from_dotlist(overrides), resolve=True)
        if not isinstance(override_values, dict):
            raise ValueError("configuration overrides must form an object")
        values = OmegaConf.to_container(OmegaConf.merge(values, override_values), resolve=True)
        assert isinstance(values, dict)
    return _structured_config(values)


def load_checkpoint_config(path: str | Path) -> ExperimentConfig:
    return _structured_config(
        deepcopy(_require_current_schema(json.loads(Path(path).read_text()), path))
    )
