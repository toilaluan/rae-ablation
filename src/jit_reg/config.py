from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 10

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
class REGLoRAConfig:
    enabled: bool = False
    rank: int = 16
    alpha: float = 16.0
    mlp: bool = False


@dataclass
class REGConfig:
    enabled: bool = False
    pool_size: int = 4
    target_std: float = 0.53
    spatial_rope: bool = True
    loss_weight: float = 0.03
    internal_guidance_block: int = 4
    attention_mode: str = "joint"
    join_after_block: int = 4
    dropout_probability: float = 0.0
    lora: REGLoRAConfig = field(default_factory=REGLoRAConfig)

    @property
    def num_tokens(self) -> int:
        return self.pool_size**2


@dataclass
class REPAConfig:
    enabled: bool = False
    block: int = 4
    projection_hidden_size: int = 2048
    loss_weight: float = 0.5
    token_scope: str = "image"


@dataclass
class FeaturesConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    reg: REGConfig = field(default_factory=REGConfig)
    repa: REPAConfig = field(default_factory=REPAConfig)

    @property
    def enabled(self) -> bool:
        return self.reg.enabled or self.repa.enabled


@dataclass
class DDTDecoderConfig:
    hidden_size: int = 2048
    depth: int = 2
    num_heads: int = 16
    mlp_ratio: float = 4.0


@dataclass
class ModelConfig:
    input_space: str = "pixels"
    image_size: int = 256
    in_channels: int = 3
    patch_size: int = 16
    num_image_timestep_tokens: int = 1
    num_reg_timestep_tokens: int = 0
    num_image_class_tokens: int = 1
    num_reg_class_tokens: int = 0
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    fused_mlp: bool = False
    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    gradient_checkpointing: bool = False
    variant: str = "DDT-B"
    decoder: DDTDecoderConfig = field(default_factory=DDTDecoderConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)

    @property
    def grid_size(self) -> int:
        return self.input_size // self.patch_size

    @property
    def model_variant(self) -> str:
        return f"{self.variant}/{self.patch_size}-{self.input_space}"

    @property
    def input_size(self) -> int:
        if self.input_space == "pixels":
            return self.image_size
        return self.image_size // FLUX2_SPATIAL_COMPRESSION

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.in_channels, self.input_size, self.input_size)

    def validate(self) -> None:
        _positive(
            "model dimensions",
            self.image_size,
            self.in_channels,
            self.patch_size,
            self.hidden_size,
            self.depth,
            self.num_heads,
            self.num_classes,
            self.mlp_ratio,
        )
        if self.input_space not in {"pixels", "flux2_latents"}:
            raise ValueError(
                "model.input_space must be pixels or flux2_latents"
            )
        expected_channels = (
            3 if self.input_space == "pixels" else FLUX2_LATENT_CHANNELS
        )
        if self.in_channels != expected_channels:
            raise ValueError(
                f"model.in_channels must be {expected_channels} for "
                f"model.input_space={self.input_space}"
            )
        if (
            self.input_space == "flux2_latents"
            and self.image_size % FLUX2_SPATIAL_COMPRESSION
        ):
            raise ValueError(
                "FLUX.2 latent mode requires model.image_size to be "
                "divisible by 16"
            )
        if self.input_size % self.patch_size:
            raise ValueError(
                "the diffusion input size must be exactly divisible by "
                "model.patch_size"
            )
        token_counts = {
            "model.num_image_timestep_tokens": self.num_image_timestep_tokens,
            "model.num_reg_timestep_tokens": self.num_reg_timestep_tokens,
            "model.num_image_class_tokens": self.num_image_class_tokens,
            "model.num_reg_class_tokens": self.num_reg_class_tokens,
        }
        for name, count in token_counts.items():
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
            ):
                raise ValueError(
                    f"{name} must be a non-negative integer"
                )
        reg = self.features.reg
        if not reg.enabled and (
            self.num_reg_timestep_tokens or self.num_reg_class_tokens
        ):
            raise ValueError(
                "REG timestep/class token counts must be zero when REG is disabled"
            )
        if self.hidden_size % self.num_heads:
            raise ValueError(
                "model.hidden_size must be divisible by model.num_heads"
            )
        if (self.hidden_size // self.num_heads) % 4:
            raise ValueError(
                "model attention head dimension must be divisible by 4 for 2D RoPE"
            )
        decoder = self.decoder
        _positive(
            "model.decoder dimensions",
            decoder.hidden_size,
            decoder.depth,
            decoder.num_heads,
            decoder.mlp_ratio,
        )
        if decoder.hidden_size % decoder.num_heads:
            raise ValueError(
                "model.decoder.hidden_size must be divisible by "
                "model.decoder.num_heads"
            )
        if (decoder.hidden_size // decoder.num_heads) % 4:
            raise ValueError(
                "model decoder attention head dimension must be divisible by 4 "
                "for 2D RoPE"
            )
        if not 0 <= self.class_dropout_prob < 1:
            raise ValueError(
                "model.class_dropout_prob must be in [0, 1)"
            )
        if not self.variant:
            raise ValueError("model.variant cannot be empty")

        features = self.features
        if features.enabled:
            _positive(
                "model.features.encoder dimensions",
                features.encoder.feature_dim,
                features.encoder.register_tokens,
            )
            if not features.encoder.model_name:
                raise ValueError(
                    "model.features.encoder.model_name cannot be empty"
                )
        if reg.enabled:
            if reg.attention_mode not in {"joint", "asymmetric"}:
                raise ValueError(
                    "model.features.reg.attention_mode must be joint or asymmetric"
                )
            if (
                not isinstance(reg.join_after_block, int)
                or isinstance(reg.join_after_block, bool)
                or not 0 <= reg.join_after_block <= self.depth
            ):
                raise ValueError(
                    "model.features.reg.join_after_block must be an integer "
                    "within [0, model.depth]"
                )
            if (
                not math.isfinite(reg.dropout_probability)
                or not 0 <= reg.dropout_probability <= 1
            ):
                raise ValueError(
                    "model.features.reg.dropout_probability must be in [0, 1]"
                )
            if (
                reg.dropout_probability
                and reg.attention_mode != "asymmetric"
            ):
                raise ValueError(
                    "model.features.reg.dropout_probability requires "
                    "attention_mode=asymmetric"
                )
            if reg.pool_size < 1 or reg.pool_size > self.grid_size:
                raise ValueError(
                    "model.features.reg.pool_size must not exceed the model grid"
                )
            if not math.isfinite(reg.target_std) or reg.target_std <= 0:
                raise ValueError(
                    "model.features.reg.target_std must be finite and positive"
                )
            if reg.loss_weight < 0:
                raise ValueError(
                    "model.features.reg.loss_weight cannot be negative"
                )
            if (
                not isinstance(reg.internal_guidance_block, int)
                or isinstance(reg.internal_guidance_block, bool)
                or not 1 <= reg.internal_guidance_block <= self.depth
            ):
                raise ValueError(
                    "model.features.reg.internal_guidance_block must be within "
                    "model.depth"
                )
            if reg.lora.enabled:
                if (
                    not isinstance(reg.lora.rank, int)
                    or isinstance(reg.lora.rank, bool)
                    or reg.lora.rank < 1
                ):
                    raise ValueError(
                        "model.features.reg.lora.rank must be a positive integer"
                    )
                if (
                    not math.isfinite(reg.lora.alpha)
                    or reg.lora.alpha <= 0
                ):
                    raise ValueError(
                        "model.features.reg.lora.alpha must be finite and positive"
                    )
        if features.repa.enabled:
            repa = features.repa
            if not 1 <= repa.block <= self.depth:
                raise ValueError(
                    "model.features.repa.block must be within model.depth"
                )
            if repa.projection_hidden_size < 1:
                raise ValueError(
                    "model.features.repa.projection_hidden_size must be positive"
                )
            if repa.loss_weight < 0:
                raise ValueError(
                    "model.features.repa.loss_weight cannot be negative"
                )
            if repa.token_scope not in {"image", "image_and_reg"}:
                raise ValueError(
                    "model.features.repa.token_scope must be image or image_and_reg"
                )
            if repa.token_scope == "image_and_reg" and not reg.enabled:
                raise ValueError(
                    "REPA image_and_reg token scope requires REG"
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
class FeatureCacheConfig:
    enabled: bool = False
    directory: str = "data/features"
    batch_size: int = 128
    log_interval: int = 10


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
    reg_shift_loc: float = 0.0
    reg_shift_scale: float = 1.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class SamplingConfig:
    num_samples: int = 50_000
    batch_size: int = 32
    sampling_steps: int = 50
    solver: str = "euler"
    guidance_scale: float = 1.78
    reg_guidance_scale: float = 1.78
    timestep_shift: float = 1.0
    reg_timestep_shifts: list[float] = field(default_factory=lambda: [1.0])
    img_denoising_interval: list[float] = field(
        default_factory=lambda: [0.0, 1.0]
    )
    reg_denoising_interval: list[float] = field(
        default_factory=lambda: [0.0, 1.0]
    )
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
    project: str = "jit-reg"
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"
    checkpoint_artifacts: bool = False


@dataclass
class ExperimentConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    data: DataConfig = field(default_factory=DataConfig)
    feature_cache: FeatureCacheConfig = field(default_factory=FeatureCacheConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
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
                f"schema_version: {CONFIG_SCHEMA_VERSION} is required (clean break; migration "
                "is not supported)"
            )
        m = self.model
        m.validate()
        if m.input_space == "flux2_latents":
            vae = self.vae
            if not vae.model_name or not vae.revision or not vae.subfolder:
                raise ValueError(
                    "latent mode requires a pinned VAE model, revision, and subfolder"
                )
            _positive("vae.micro_batch_size", vae.micro_batch_size)
            if not vae.cache.directory:
                raise ValueError("vae.cache.directory cannot be empty")
            _positive(
                "VAE-cache batch size and log interval",
                vae.cache.batch_size,
                vae.cache.log_interval,
            )
        elif self.vae.cache.enabled:
            raise ValueError(
                "vae.cache.enabled requires model.input_space=flux2_latents"
            )
        features = m.features
        if self.train.optimizer.kind not in {"adamw", "muon", "fused_muon"}:
            raise ValueError(
                "optimizer.kind must be adamw, muon, or fused_muon"
            )
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
        if self.train.epochs < 1 or self.train.global_batch_size < 1:
            raise ValueError(
                "training epochs and global batch size must be positive"
            )
        if self.train.max_steps is not None and self.train.max_steps < 1:
            raise ValueError("train.max_steps must be positive")
        if not 0 < self.train.ema_decay < 1:
            raise ValueError("train.ema_decay must be in (0, 1)")
        if not math.isfinite(self.train.timestep_loc):
            raise ValueError("train.timestep_loc must be finite")
        if not math.isfinite(self.train.timestep_scale) or self.train.timestep_scale <= 0:
            raise ValueError("train.timestep_scale must be finite and positive")
        if features.reg.enabled:
            if not math.isfinite(self.train.reg_shift_loc):
                raise ValueError("train.reg_shift_loc must be finite")
            if (
                not math.isfinite(self.train.reg_shift_scale)
                or self.train.reg_shift_scale <= 0
            ):
                raise ValueError(
                    "train.reg_shift_scale must be finite and positive"
                )
        if features.enabled:
            if not self.feature_cache.directory:
                raise ValueError("feature_cache.directory cannot be empty")
            _positive(
                "feature-cache batch size and log interval",
                self.feature_cache.batch_size,
                self.feature_cache.log_interval,
            )
        if self.checkpoint.keep_last_n < 0:
            raise ValueError("checkpoint.keep_last_n cannot be negative")
        self._validate_sampling()
        if self.evaluation.batch_size < 1 or self.evaluation.log_interval < 1:
            raise ValueError("evaluation batch size and log interval must be positive")
        if not self.evaluation.reference_npz or not self.evaluation.device:
            raise ValueError("evaluation reference_npz and device cannot be empty")

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
        if not math.isfinite(sampling.reg_guidance_scale):
            raise ValueError("sampling.reg_guidance_scale must be finite")
        if not math.isfinite(sampling.timestep_shift) or sampling.timestep_shift <= 0:
            raise ValueError("sampling.timestep_shift must be finite and positive")
        shifts = sampling.reg_timestep_shifts
        if (
            not shifts
            or any(not math.isfinite(value) or value <= 0 for value in shifts)
            or len(set(shifts)) != len(shifts)
        ):
            raise ValueError(
                "sampling.reg_timestep_shifts must be non-empty, unique, finite, "
                "and positive"
            )
        for name, interval in (
            ("sampling.img_denoising_interval", sampling.img_denoising_interval),
            ("sampling.reg_denoising_interval", sampling.reg_denoising_interval),
        ):
            if len(interval) != 2:
                raise ValueError(f"{name} must contain exactly [start, end]")
            start, end = interval
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or not 0 <= start < end <= 1
            ):
                raise ValueError(f"{name} must satisfy 0 <= start < end <= 1")
            start_step = math.floor(start * sampling.sampling_steps + 0.5)
            end_step = math.floor(end * sampling.sampling_steps + 0.5)
            if start_step >= end_step:
                raise ValueError(
                    f"{name} collapses to zero steps after rounding"
                )
        if not 0 <= sampling.cfg_start <= sampling.cfg_end <= 1:
            raise ValueError("sampling CFG interval must satisfy 0 <= start <= end <= 1")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_current_schema(values: Any, source: str | Path) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ValueError(f"{source} must contain a configuration object")
    version = values.get("schema_version")
    if version != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{source} uses unsupported configuration schema {version!r}; "
            f"schema_version: {CONFIG_SCHEMA_VERSION} is required (clean break; old configs "
            "and checkpoints cannot be migrated)"
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
    """Load the current structured YAML schema and reject old or unknown fields."""
    from omegaconf import OmegaConf

    values: dict[str, Any] = {"schema_version": CONFIG_SCHEMA_VERSION}
    if path is not None:
        loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        values = _require_current_schema(loaded, path)
    if overrides:
        override_values = OmegaConf.to_container(
            OmegaConf.from_dotlist(overrides), resolve=True
        )
        if not isinstance(override_values, dict):
            raise ValueError("configuration overrides must form an object")
        values = OmegaConf.to_container(
            OmegaConf.merge(values, override_values), resolve=True
        )
        assert isinstance(values, dict)
    return _structured_config(values)


def load_checkpoint_config(path: str | Path) -> ExperimentConfig:
    """Load a resolved current-schema checkpoint config."""
    return _structured_config(
        deepcopy(
            _require_current_schema(
                json.loads(Path(path).read_text()),
                path,
            )
        )
    )
