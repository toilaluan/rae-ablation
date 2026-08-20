from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from .config import IMAGE_SIZE, PixelCodecConfig

PIXEL_CODEC_ARTIFACT_SCHEMA_VERSION = 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv1(F.silu(self.norm1(x)))
        return x + self.conv2(F.silu(self.norm2(residual)))


class PixelCodec(nn.Module):
    """Small deterministic RGB codec with a 1024x8x8 aligned bottleneck."""

    def __init__(self, config: PixelCodecConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        widths = config.widths
        self.encoder_stem = nn.Conv2d(3, widths[0], 3, padding=1)
        encoder: list[nn.Module] = []
        for index in range(len(widths) - 1):
            encoder.extend(ResidualBlock(widths[index]) for _ in range(config.residual_blocks_per_scale))
            encoder.append(nn.Conv2d(widths[index], widths[index + 1], 3, stride=2, padding=1))
        encoder.extend(ResidualBlock(widths[-1]) for _ in range(config.residual_blocks_per_scale))
        self.encoder_stages = nn.Sequential(*encoder)
        self.encoder_projection = nn.Conv2d(widths[-1], config.latent_channels, 1)
        self.bottleneck_norm = nn.BatchNorm2d(
            config.latent_channels, affine=False, track_running_stats=True
        )

        self.decoder_projection = nn.Conv2d(config.latent_channels, widths[-1], 1)
        decoder: list[nn.Module] = [
            *(ResidualBlock(widths[-1]) for _ in range(config.residual_blocks_per_scale))
        ]
        for index in range(len(widths) - 1, 0, -1):
            decoder.extend(
                (
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(widths[index], widths[index - 1], 3, padding=1),
                )
            )
            decoder.extend(
                ResidualBlock(widths[index - 1])
                for _ in range(config.residual_blocks_per_scale)
            )
        self.decoder_stages = nn.Sequential(*decoder)
        self.decoder_norm = nn.GroupNorm(32, widths[0])
        self.decoder_output = nn.Conv2d(widths[0], 3, 3, padding=1)
        self._initialize_weights()

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return self.config.latent_channels, self.config.latent_size, self.config.latent_size

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, pixels: Tensor) -> Tensor:
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"pixel codec input must have shape [B, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]")
        latent = self.encoder_stages(self.encoder_stem(pixels.float().mul(2).sub(1)))
        latent = self.bottleneck_norm(self.encoder_projection(latent))
        if tuple(latent.shape[1:]) != self.latent_shape:
            raise ValueError(f"pixel codec encoded {tuple(latent.shape[1:])}, expected {self.latent_shape}")
        return latent

    def decode(self, latent: Tensor) -> Tensor:
        if latent.ndim != 4 or tuple(latent.shape[1:]) != self.latent_shape:
            raise ValueError(f"pixel codec latent must have shape [B, {self.latent_shape}]")
        decoded = self.decoder_stages(self.decoder_projection(latent))
        decoded = self.decoder_output(F.silu(self.decoder_norm(decoded))).tanh().add(1).div(2)
        expected = (latent.shape[0], 3, IMAGE_SIZE, IMAGE_SIZE)
        if tuple(decoded.shape) != expected:
            raise ValueError(f"pixel codec decoded {tuple(decoded.shape)}, expected {expected}")
        return decoded

    def forward(self, pixels: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(pixels)
        return self.decode(latent), latent


class VFLosses(NamedTuple):
    total: Tensor
    marginal_cosine: Tensor
    marginal_distance_matrix: Tensor


def vision_foundation_losses(
    pixel_tokens: Tensor,
    rae_tokens: Tensor,
    *,
    cosine_margin: float = 0.5,
    distance_margin: float = 0.25,
) -> VFLosses:
    if pixel_tokens.shape != rae_tokens.shape or pixel_tokens.ndim != 4:
        raise ValueError("VF alignment requires equal [B,C,H,W] token tensors")
    if not 0 <= cosine_margin < 1 or not 0 <= distance_margin < 1:
        raise ValueError("VF margins must be in [0, 1)")
    pixel = F.normalize(pixel_tokens.float().flatten(2).transpose(1, 2), dim=-1)
    rae = F.normalize(rae_tokens.float().flatten(2).transpose(1, 2), dim=-1)
    pointwise = F.relu(1 - cosine_margin - (pixel * rae).sum(dim=-1)).mean()
    pixel_matrix = pixel @ pixel.transpose(1, 2)
    rae_matrix = rae @ rae.transpose(1, 2)
    matrix = F.relu((pixel_matrix - rae_matrix).abs() - distance_margin).mean()
    return VFLosses(pointwise + matrix, pointwise, matrix)


def adaptive_alignment_weight(
    reconstruction_loss: Tensor,
    alignment_loss: Tensor,
    shared_representation: Tensor,
    *,
    gradient_ratio: float = 0.1,
    eps: float = 1e-6,
    maximum: float = 1e4,
) -> Tensor:
    if gradient_ratio <= 0 or eps <= 0 or maximum <= 0:
        raise ValueError("adaptive alignment settings must be positive")
    reconstruction_gradient = torch.autograd.grad(
        reconstruction_loss, shared_representation, retain_graph=True
    )[0]
    alignment_gradient = torch.autograd.grad(
        alignment_loss, shared_representation, retain_graph=True
    )[0]
    ratio = reconstruction_gradient.float().norm() / (
        alignment_gradient.float().norm() + eps
    )
    return (gradient_ratio * ratio).clamp(max=maximum).detach()


@dataclass(frozen=True)
class PixelTokenStatistics:
    mean: Tensor
    std: Tensor
    rae_correlation: Tensor

    def validate(self, channels: int) -> None:
        expected = (channels,)
        if self.mean.shape != expected or self.std.shape != expected or self.rae_correlation.shape != expected:
            raise ValueError("pixel token statistics have incompatible shapes")
        if (
            not torch.all(torch.isfinite(self.mean))
            or not torch.all(torch.isfinite(self.std))
            or torch.any(self.std <= 0)
            or not torch.all(torch.isfinite(self.rae_correlation))
            or torch.any(self.rae_correlation.abs() > 1 + 1e-4)
        ):
            raise ValueError("pixel token statistics are invalid")

    def normalize(self, tokens: Tensor) -> Tensor:
        self.validate(tokens.shape[1])
        return (tokens.float() - self.mean.to(tokens)[None, :, None, None]) / self.std.to(tokens)[None, :, None, None]

    def denormalize(self, tokens: Tensor) -> Tensor:
        self.validate(tokens.shape[1])
        return tokens.float() * self.std.to(tokens)[None, :, None, None] + self.mean.to(tokens)[None, :, None, None]

    def expected_mixture_variance(self, ratio: Tensor) -> Tensor:
        r = ratio.float()[..., None]
        correlation = self.rae_correlation.to(ratio)[None]
        return r.square() + (1 - r).square() + 2 * r * (1 - r) * correlation


@dataclass(frozen=True)
class PixelCodecArtifact:
    directory: Path
    model: PixelCodec
    statistics: PixelTokenStatistics
    weights_sha256: str
    metadata: dict[str, Any]


def _weights_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}


def save_pixel_codec_artifact(
    directory: str | Path,
    model: PixelCodec,
    statistics: PixelTokenStatistics,
    *,
    metadata: dict[str, Any],
) -> Path:
    target = Path(directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"pixel codec artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.partial-{os.getpid()}"
    temporary.mkdir()
    try:
        weights = temporary / "model.safetensors"
        save_file(_model_state(model), str(weights))
        save_file(
            {
                "mean": statistics.mean.float().cpu().contiguous(),
                "std": statistics.std.float().cpu().contiguous(),
                "rae_correlation": statistics.rae_correlation.float().cpu().contiguous(),
            },
            str(temporary / "statistics.safetensors"),
        )
        digest = _weights_digest(weights)
        values = {
            **metadata,
            "schema_version": PIXEL_CODEC_ARTIFACT_SCHEMA_VERSION,
            "pixel_codec": asdict(model.config),
            "weights_sha256": digest,
        }
        (temporary / "metadata.json").write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def load_pixel_codec_artifact(
    directory: str | Path,
    device: torch.device | str,
    *,
    expected_config: PixelCodecConfig | None = None,
) -> PixelCodecArtifact:
    target = Path(directory).resolve()
    metadata_path = target / "metadata.json"
    weights = target / "model.safetensors"
    stats_path = target / "statistics.safetensors"
    if not metadata_path.is_file() or not weights.is_file() or not stats_path.is_file():
        raise FileNotFoundError(f"pixel codec artifact is incomplete: {target}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != PIXEL_CODEC_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("pixel codec artifact schema is incompatible")
    stored = dict(metadata.get("pixel_codec", {}))
    stored["checkpoint"] = str(directory)
    config = PixelCodecConfig(**stored)
    config.validate()
    if expected_config is not None:
        left = asdict(config)
        right = asdict(expected_config)
        left.pop("checkpoint", None)
        right.pop("checkpoint", None)
        if left != right:
            raise ValueError("pixel codec artifact configuration does not match experiment")
    digest = _weights_digest(weights)
    if digest != metadata.get("weights_sha256"):
        raise ValueError("pixel codec weight digest does not match metadata")
    target_device = torch.device(device)
    model = PixelCodec(config).to(target_device)
    model.load_state_dict(load_file(str(weights), device=str(target_device)))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    values = load_file(str(stats_path), device="cpu")
    statistics = PixelTokenStatistics(
        values["mean"], values["std"], values["rae_correlation"]
    )
    statistics.validate(config.latent_channels)
    return PixelCodecArtifact(target, model, statistics, digest, metadata)
