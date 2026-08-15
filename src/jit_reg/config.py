from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 13
IMAGE_SIZE = 256
RAE_CHANNELS = 1024
RAE_INPUT_SIZE = 8
RAE_TOKEN_COUNT = RAE_INPUT_SIZE**2
RAE_BASE_REPOSITORY = "toilaluan/raev2-dinov3l-k7"
RAE_BASE_REVISION = "46ce5a7e76df99eb5528020d3c65cef528a355fa"
RAE_COMPRESSOR_REPOSITORY = "toilaluan/raev2-compressor"
RAE_COMPRESSOR_REVISION = "846e2213dcf6218d3b726448ae52d8104c925c0f"


def _positive(name: str, *values: int | float) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} must be positive")


@dataclass
class ExplorationConfig:
    k_early: int = 1
    k_full: int = 1
    early_candidate_chunk_size: int | None = None
    full_candidate_chunk_size: int | None = None

    @property
    def k_total(self) -> int:
        return self.k_early * self.k_full


@dataclass
class DDTConfig:
    input_size: int = RAE_INPUT_SIZE
    patch_size: list[int] = field(default_factory=lambda: [1, 1])
    in_channels: int = RAE_CHANNELS
    hidden_size: list[int] = field(default_factory=lambda: [768, 2048])
    depth: list[int] = field(default_factory=lambda: [12, 2])
    num_heads: list[int] = field(default_factory=lambda: [12, 16])
    mlp_ratio: float = 4.0
    base_model_depth: int = 4
    num_classes: int = 1000
    cfg_dropout_prob: float = 0.1
    num_timestep_tokens: int = 4
    num_class_tokens: int = 8
    gradient_checkpointing: bool = False
    variant: str = "DDT-B/1-RAEv2-8x8"

    @property
    def grid_size(self) -> int:
        return self.input_size // self.patch_size[0]

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self.in_channels, self.input_size, self.input_size

    @property
    def model_variant(self) -> str:
        return self.variant

    def validate(self) -> None:
        if len(self.patch_size) != 2 or len(self.hidden_size) != 2:
            raise ValueError("model.patch_size and model.hidden_size must contain two values")
        if len(self.depth) != 2 or len(self.num_heads) != 2:
            raise ValueError("model.depth and model.num_heads must contain two values")
        _positive(
            "DDT dimensions",
            self.input_size,
            *self.patch_size,
            self.in_channels,
            *self.hidden_size,
            *self.depth,
            *self.num_heads,
            self.mlp_ratio,
            self.base_model_depth,
            self.num_classes,
            self.num_timestep_tokens,
            self.num_class_tokens,
        )
        if self.input_size % self.patch_size[0] or self.input_size % self.patch_size[1]:
            raise ValueError("DDT input size must be divisible by both patch sizes")
        for stage, (width, heads) in enumerate(
            zip(self.hidden_size, self.num_heads, strict=True)
        ):
            if width % heads:
                raise ValueError(f"model.hidden_size[{stage}] must be divisible by num_heads")
            if (width // heads) % 4:
                raise ValueError("attention head dimensions must be divisible by four for 2D RoPE")
        if self.base_model_depth > self.depth[0]:
            raise ValueError("model.base_model_depth must be within encoder depth")
        if not 0 <= self.cfg_dropout_prob < 1:
            raise ValueError("model.cfg_dropout_prob must be in [0, 1)")
        if not self.variant:
            raise ValueError("model.variant cannot be empty")


# ModelConfig is retained as a descriptive alias, not the old generic model schema.
ModelConfig = DDTConfig


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
class TokenCacheConfig:
    directory: str = "data/rae-tokens"
    batch_size: int = 128
    log_interval: int = 10


@dataclass
class RAEConfig:
    base_repository: str = RAE_BASE_REPOSITORY
    base_revision: str = RAE_BASE_REVISION
    compressor_repository: str = RAE_COMPRESSOR_REPOSITORY
    compressor_revision: str = RAE_COMPRESSOR_REVISION
    trust_remote_code: bool = True
    remote_code_schema: int = 1
    micro_batch_size: int = 16
    cache: TokenCacheConfig = field(default_factory=TokenCacheConfig)

    def validate(self) -> None:
        expected = (
            ("base_repository", self.base_repository, RAE_BASE_REPOSITORY),
            ("base_revision", self.base_revision, RAE_BASE_REVISION),
            ("compressor_repository", self.compressor_repository, RAE_COMPRESSOR_REPOSITORY),
            ("compressor_revision", self.compressor_revision, RAE_COMPRESSOR_REVISION),
        )
        for name, actual, pinned in expected:
            if actual != pinned:
                raise ValueError(f"rae.{name} must use pinned source {pinned}")
        if self.trust_remote_code is not True:
            raise ValueError("rae.trust_remote_code must be true")
        _positive(
            "RAE settings",
            self.remote_code_schema,
            self.micro_batch_size,
            self.cache.batch_size,
            self.cache.log_interval,
        )
        if not self.cache.directory:
            raise ValueError("rae.cache.directory cannot be empty")


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
    timestep_shift: float = 4.0
    velocity_denominator_floor: float = 0.05
    base_loss_weight: float = 1.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class PriorConfig:
    """Optional frozen checkpoint used to transform Gaussian flow sources."""

    checkpoint_dir: str | None = None
    weight_source: str = "ema"


@dataclass
class SamplingConfig:
    num_samples: int = 50_000
    batch_size: int = 32
    sampling_steps: int = 50
    solver: str = "euler"
    timestep_shift: float = 4.0
    ig_scale: float = 1.78
    ig_start: float = 0.1
    ig_end: float = 1.0
    cfg_scale: float = 1.0
    cfg_start: float = 0.0
    cfg_end: float = 1.0
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
    project: str = "rae-ddt-k-exploration"
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"
    checkpoint_artifacts: bool = False


@dataclass
class ExperimentConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    data: DataConfig = field(default_factory=DataConfig)
    rae: RAEConfig = field(default_factory=RAEConfig)
    model: DDTConfig = field(default_factory=DDTConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
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
        self.rae.validate()
        if self.model.input_shape != (RAE_CHANNELS, RAE_INPUT_SIZE, RAE_INPUT_SIZE):
            raise ValueError("schema-v13 experiments must consume 1024x8x8 RAE tokens")
        if self.model.patch_size != [1, 1]:
            raise ValueError("schema-v13 experiments require model.patch_size [1, 1]")
        explore = self.exploration
        for name, value in (("k_early", explore.k_early), ("k_full", explore.k_full)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"exploration.{name} must be a positive integer")
        for name, chunk, candidates, active in (
            ("early_candidate_chunk_size", explore.early_candidate_chunk_size, explore.k_total, explore.k_early > 1),
            ("full_candidate_chunk_size", explore.full_candidate_chunk_size, explore.k_full, explore.k_full > 1),
        ):
            if chunk is None:
                continue
            if not active:
                raise ValueError(f"exploration.{name} requires its selection stage")
            if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
                raise ValueError(f"exploration.{name} must be a positive integer")
            if chunk > candidates or candidates % chunk:
                raise ValueError(f"exploration.{name} must divide its stage candidate count")
        if self.prior.weight_source not in {"model", "ema"}:
            raise ValueError("prior.weight_source must be model or ema")
        if self.prior.checkpoint_dir is not None:
            if not self.prior.checkpoint_dir:
                raise ValueError("prior.checkpoint_dir cannot be empty")
            if explore.k_total != 1:
                raise ValueError("checkpoint prior experiments require exploration K=1")
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
        _positive("training counts", self.train.epochs, self.train.global_batch_size)
        if self.train.max_steps is not None and self.train.max_steps < 1:
            raise ValueError("train.max_steps must be positive")
        if not 0 < self.train.ema_decay < 1:
            raise ValueError("train.ema_decay must be in (0, 1)")
        if not math.isfinite(self.train.timestep_loc):
            raise ValueError("train.timestep_loc must be finite")
        _positive(
            "training flow settings",
            self.train.timestep_scale,
            self.train.timestep_shift,
            self.train.velocity_denominator_floor,
        )
        if self.train.base_loss_weight < 0:
            raise ValueError("train.base_loss_weight cannot be negative")
        for name, value in (("validation.interval", self.validation.interval), ("checkpoint.interval", self.checkpoint.interval)):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.validation.periodic_samples > self.validation.final_samples:
            raise ValueError("periodic validation must be a subset of final validation")
        if self.validation.periodic_samples % self.model.num_classes or self.validation.final_samples % self.model.num_classes:
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
        for name, value in (("timestep_shift", s.timestep_shift), ("ig_scale", s.ig_scale), ("cfg_scale", s.cfg_scale)):
            if not math.isfinite(value):
                raise ValueError(f"sampling.{name} must be finite")
        if s.timestep_shift <= 0:
            raise ValueError("sampling.timestep_shift must be positive")
        if not 0 <= s.ig_start < s.ig_end <= 1:
            raise ValueError("sampling IG interval must satisfy 0 <= start < end <= 1")
        if not 0 <= s.cfg_start < s.cfg_end <= 1:
            raise ValueError("sampling CFG interval must satisfy 0 <= start < end <= 1")

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
    return _structured_config(deepcopy(_require_current_schema(json.loads(Path(path).read_text()), path)))
