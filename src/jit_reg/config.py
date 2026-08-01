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
class EncoderConfig:
    model_name: str = "vit_base_patch16_dinov3.lvd1689m"
    feature_dim: int = 768
    register_tokens: int = 4


@dataclass
class REPACacheConfig:
    enabled: bool = False
    directory: str = "data/features"
    batch_size: int = 128
    log_interval: int = 10


@dataclass
class REPAConfig:
    enabled: bool = False
    block: int = 4
    projection_hidden_size: int = 2048
    loss_weight: float = 0.5
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    cache: REPACacheConfig = field(default_factory=REPACacheConfig)


@dataclass
class ExplorationConfig:
    k: int = 1
    selector: str = "none"
    candidate_chunk_size: int | None = None


@dataclass
class ModelConfig:
    input_space: str = "pixels"
    image_size: int = 256
    in_channels: int = 3
    patch_size: int = 16
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    gradient_checkpointing: bool = False
    variant: str = "SiT-B"

    @property
    def input_size(self) -> int:
        return (
            self.image_size
            if self.input_space == "pixels"
            else self.image_size // FLUX2_SPATIAL_COMPRESSION
        )

    @property
    def grid_size(self) -> int:
        return self.input_size // self.patch_size

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self.in_channels, self.input_size, self.input_size

    @property
    def model_variant(self) -> str:
        return f"{self.variant}/{self.patch_size}-{self.input_space}"

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
        if self.input_space not in {"pixels", "flux2_latents"}:
            raise ValueError("model.input_space must be pixels or flux2_latents")
        expected_channels = 3 if self.input_space == "pixels" else FLUX2_LATENT_CHANNELS
        if self.in_channels != expected_channels:
            raise ValueError(
                f"model.in_channels must be {expected_channels} for "
                f"model.input_space={self.input_space}"
            )
        if self.input_space == "flux2_latents" and (
            self.image_size % FLUX2_SPATIAL_COMPRESSION
        ):
            raise ValueError("FLUX.2 latent mode requires image_size divisible by 16")
        if self.input_size % self.patch_size:
            raise ValueError("diffusion input size must be divisible by model.patch_size")
        if self.hidden_size % self.num_heads:
            raise ValueError("model.hidden_size must be divisible by model.num_heads")
        if (self.hidden_size // self.num_heads) % 4:
            raise ValueError("attention head dimension must be divisible by four for 2D RoPE")
        if not 0 <= self.class_dropout_prob < 1:
            raise ValueError("model.class_dropout_prob must be in [0, 1)")
        if not self.variant:
            raise ValueError("model.variant cannot be empty")


@dataclass
class CompileConfig:
    enabled: bool = True
    backend: str = "inductor"
    mode: str = "default"
    fullgraph: bool = True
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
    max_steps: int | None = None
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    seed: int = 0
    timestep_loc: float = 0.0
    timestep_scale: float = 1.0
    velocity_denominator_floor: float = 0.05
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class SamplingConfig:
    num_samples: int = 50_000
    batch_size: int = 32
    sampling_steps: int = 50
    solver: str = "euler"
    guidance_scale: float = 1.78
    timestep_shift: float = 1.0
    cfg_start: float = 0.0
    cfg_end: float = 0.9
    log_interval: int = 10


@dataclass
class EvaluationConfig:
    reference_npz: str = "data/VIRTUAL_imagenet256_labeled.npz"
    batch_size: int = 32
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
    project: str = "sit-k-exploration"
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
    repa: REPAConfig = field(default_factory=REPAConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
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
                f"configuration schema version {self.schema_version!r} is unsupported; "
                f"schema_version: {CONFIG_SCHEMA_VERSION} is required (clean break)"
            )
        self.model.validate()
        if self.model.input_space == "flux2_latents":
            if not self.vae.model_name or not self.vae.revision or not self.vae.subfolder:
                raise ValueError("latent mode requires a pinned VAE model, revision, and subfolder")
            _positive(
                "VAE settings",
                self.vae.micro_batch_size,
                self.vae.cache.batch_size,
                self.vae.cache.log_interval,
            )
        elif self.vae.cache.enabled:
            raise ValueError("vae.cache.enabled requires flux2_latents input")

        repa = self.repa
        if repa.enabled:
            _positive(
                "REPA settings",
                repa.block,
                repa.projection_hidden_size,
                repa.encoder.feature_dim,
                repa.encoder.register_tokens,
                repa.cache.batch_size,
                repa.cache.log_interval,
            )
            if repa.block > self.model.depth:
                raise ValueError("repa.block must be within model.depth")
            if repa.loss_weight < 0:
                raise ValueError("repa.loss_weight cannot be negative")
            if not repa.encoder.model_name or not repa.cache.directory:
                raise ValueError("REPA encoder and cache directory cannot be empty")
        elif repa.cache.enabled:
            raise ValueError("repa.cache.enabled requires repa.enabled=true")

        explore = self.exploration
        if not isinstance(explore.k, int) or isinstance(explore.k, bool) or explore.k < 1:
            raise ValueError("exploration.k must be a positive integer")
        if explore.selector not in {"none", "final_velocity", "repa"}:
            raise ValueError("exploration.selector must be none, final_velocity, or repa")
        if explore.k == 1 and explore.selector != "none":
            raise ValueError("K=1 requires exploration.selector=none")
        if explore.k > 1 and explore.selector == "none":
            raise ValueError("K>1 requires an exploration selector")
        if explore.selector == "repa" and not repa.enabled:
            raise ValueError("REPA selection requires repa.enabled=true")
        if explore.candidate_chunk_size is not None:
            chunk = explore.candidate_chunk_size
            if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
                raise ValueError("candidate_chunk_size must be a positive integer")
            if chunk > explore.k or explore.k % chunk:
                raise ValueError("candidate_chunk_size must divide exploration.k")

        if self.train.optimizer.kind not in {"adamw", "muon", "fused_muon"}:
            raise ValueError("optimizer.kind must be adamw, muon, or fused_muon")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        if self.data.workers < 0 or self.data.prefetch_factor < 1:
            raise ValueError("data workers must be non-negative and prefetch_factor positive")
        if self.data.max_samples is not None and self.data.max_samples < 1:
            raise ValueError("data.max_samples must be positive")
        if not self.data.repository or not self.data.revision or not self.data.split:
            raise ValueError("data repository, revision, and split cannot be empty")
        if self.train.epochs < 1 or self.train.global_batch_size < 1:
            raise ValueError("training epochs and global batch size must be positive")
        if self.train.max_steps is not None and self.train.max_steps < 1:
            raise ValueError("train.max_steps must be positive")
        if not 0 < self.train.ema_decay < 1:
            raise ValueError("train.ema_decay must be in (0, 1)")
        if not math.isfinite(self.train.timestep_loc):
            raise ValueError("train.timestep_loc must be finite")
        if not math.isfinite(self.train.timestep_scale) or self.train.timestep_scale <= 0:
            raise ValueError("train.timestep_scale must be finite and positive")
        if self.train.velocity_denominator_floor <= 0:
            raise ValueError("train.velocity_denominator_floor must be positive")
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
        if self.checkpoint.keep_last_n < 0:
            raise ValueError("checkpoint.keep_last_n cannot be negative")
        self._validate_sampling()

    def _validate_sampling(self) -> None:
        s = self.sampling
        _positive("sampling counts", s.num_samples, s.batch_size, s.sampling_steps, s.log_interval)
        if s.solver not in {"euler", "heun"}:
            raise ValueError("sampling.solver must be euler or heun")
        if s.num_samples % self.model.num_classes:
            raise ValueError("sampling.num_samples must be class balanced")
        if not math.isfinite(s.guidance_scale):
            raise ValueError("sampling.guidance_scale must be finite")
        if not math.isfinite(s.timestep_shift) or s.timestep_shift <= 0:
            raise ValueError("sampling.timestep_shift must be finite and positive")
        if not 0 <= s.cfg_start <= s.cfg_end <= 1:
            raise ValueError("sampling CFG interval must satisfy 0 <= start <= end <= 1")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_current_schema(values: Any, source: str | Path) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ValueError(f"{source} must contain a configuration object")
    if values.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{source} uses unsupported configuration schema {values.get('schema_version')!r}; "
            f"schema_version: {CONFIG_SCHEMA_VERSION} is required"
        )
    return values


def _structured_config(values: dict[str, Any]) -> ExperimentConfig:
    from omegaconf import OmegaConf

    try:
        cfg = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), values)
        OmegaConf.set_struct(cfg, True)
        result = OmegaConf.to_object(cfg)
    except Exception as error:
        raise ValueError(f"invalid schema-v{CONFIG_SCHEMA_VERSION} configuration: {error}") from error
    if not isinstance(result, ExperimentConfig):
        raise TypeError("configuration did not resolve to ExperimentConfig")
    result.validate()
    return result


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    from omegaconf import OmegaConf

    values: dict[str, Any] = {"schema_version": CONFIG_SCHEMA_VERSION}
    if path is not None:
        loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        values = _require_current_schema(loaded, path)
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
