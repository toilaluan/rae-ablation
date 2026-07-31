from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor, nn

from .config import (
    FLUX2_LATENT_CHANNELS,
    FLUX2_SPATIAL_COMPRESSION,
    VAEConfig,
)

RAW_LATENT_CHANNELS = 32
LATENT_PATCH_SIZE = 2
VAE_SPATIAL_COMPRESSION = 8
LATENT_SCHEMA = "flux2-mode-patch2-bn-v1"


def patchify_latents(latents: Tensor) -> Tensor:
    if latents.ndim != 4 or latents.shape[1] != RAW_LATENT_CHANNELS:
        raise ValueError(f"raw FLUX.2 latents must have shape [B, {RAW_LATENT_CHANNELS}, H, W]")
    batch, channels, height, width = latents.shape
    if height % LATENT_PATCH_SIZE or width % LATENT_PATCH_SIZE:
        raise ValueError("raw FLUX.2 latent dimensions must be divisible by two")
    return (
        latents.view(
            batch,
            channels,
            height // LATENT_PATCH_SIZE,
            LATENT_PATCH_SIZE,
            width // LATENT_PATCH_SIZE,
            LATENT_PATCH_SIZE,
        )
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(
            batch,
            channels * LATENT_PATCH_SIZE**2,
            height // LATENT_PATCH_SIZE,
            width // LATENT_PATCH_SIZE,
        )
    )


def unpatchify_latents(latents: Tensor) -> Tensor:
    if latents.ndim != 4 or latents.shape[1] != FLUX2_LATENT_CHANNELS:
        raise ValueError(
            f"normalized FLUX.2 latents must have shape [B, {FLUX2_LATENT_CHANNELS}, H, W]"
        )
    batch, channels, height, width = latents.shape
    return (
        latents.reshape(
            batch,
            channels // LATENT_PATCH_SIZE**2,
            LATENT_PATCH_SIZE,
            LATENT_PATCH_SIZE,
            height,
            width,
        )
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(
            batch,
            channels // LATENT_PATCH_SIZE**2,
            height * LATENT_PATCH_SIZE,
            width * LATENT_PATCH_SIZE,
        )
    )


class Flux2Codec(nn.Module):
    """Frozen Diffusers FLUX.2 VAE with its official latent normalization."""

    def __init__(
        self,
        vae: nn.Module,
        *,
        image_size: int,
        micro_batch_size: int,
    ) -> None:
        super().__init__()
        if image_size < 1 or image_size % FLUX2_SPATIAL_COMPRESSION:
            raise ValueError("FLUX.2 image size must be positive and divisible by 16")
        if micro_batch_size < 1:
            raise ValueError("VAE micro batch size must be positive")
        self.vae = vae
        self.image_size = image_size
        self.micro_batch_size = micro_batch_size
        self._validate_vae()
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def _validate_vae(self) -> None:
        config = getattr(self.vae, "config", None)
        if config is None:
            raise ValueError("FLUX.2 VAE has no configuration")
        if int(getattr(config, "latent_channels", -1)) != RAW_LATENT_CHANNELS:
            raise ValueError("FLUX.2 VAE must have 32 raw latent channels")
        blocks = tuple(getattr(config, "block_out_channels", ()))
        if len(blocks) != 4:
            raise ValueError("FLUX.2 VAE must use four encoder block widths")
        patch_size = tuple(getattr(config, "patch_size", ()))
        if patch_size != (LATENT_PATCH_SIZE, LATENT_PATCH_SIZE):
            raise ValueError("FLUX.2 VAE must use 2x2 latent patchification")
        bn = getattr(self.vae, "bn", None)
        if not isinstance(bn, nn.BatchNorm2d) or bn.num_features != FLUX2_LATENT_CHANNELS:
            raise ValueError("FLUX.2 VAE must expose 128-channel normalization statistics")

    def train(self, mode: bool = True) -> Flux2Codec:
        super().train(False)
        self.vae.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        size = self.image_size // FLUX2_SPATIAL_COMPRESSION
        return FLUX2_LATENT_CHANNELS, size, size

    def _chunks(self, values: Tensor) -> Iterator[Tensor]:
        return iter(values.split(self.micro_batch_size))

    def _statistics(self, like: Tensor) -> tuple[Tensor, Tensor]:
        bn = self.vae.bn
        mean = bn.running_mean.view(1, -1, 1, 1).to(like)
        std = torch.sqrt(
            bn.running_var.view(1, -1, 1, 1).to(like) + float(self.vae.config.batch_norm_eps)
        )
        return mean, std

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        expected = (images.shape[0], 3, self.image_size, self.image_size)
        if images.ndim != 4 or tuple(images.shape) != expected:
            raise ValueError(
                f"VAE images must have shape [B, 3, {self.image_size}, {self.image_size}]"
            )
        encoded = []
        for chunk in self._chunks(images):
            inputs = chunk.to(device=self.device, dtype=self.dtype).mul(2).sub(1)
            raw = self.vae.encode(inputs).latent_dist.mode()
            normalized = patchify_latents(raw)
            mean, std = self._statistics(normalized)
            encoded.append(((normalized - mean) / std).float())
        result = torch.cat(encoded)
        if tuple(result.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"FLUX.2 VAE returned {tuple(result.shape[1:])}, expected {self.latent_shape}"
            )
        return result

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        expected = (latents.shape[0], *self.latent_shape)
        if latents.ndim != 4 or tuple(latents.shape) != expected:
            raise ValueError(
                f"VAE latents must have shape [B, {self.latent_shape[0]}, "
                f"{self.latent_shape[1]}, {self.latent_shape[2]}]"
            )
        decoded = []
        for chunk in self._chunks(latents):
            normalized = chunk.to(device=self.device, dtype=self.dtype)
            mean, std = self._statistics(normalized)
            raw = unpatchify_latents(normalized * std + mean)
            image = self.vae.decode(raw, return_dict=False)[0]
            decoded.append(image.float().add(1).div(2).clamp(0, 1))
        result = torch.cat(decoded)
        expected_pixels = (latents.shape[0], 3, self.image_size, self.image_size)
        if tuple(result.shape) != expected_pixels:
            raise ValueError(
                f"FLUX.2 VAE decoded {tuple(result.shape)}, expected {expected_pixels}"
            )
        return result

    def identity(self) -> dict[str, Any]:
        return {
            "encoding_schema": LATENT_SCHEMA,
            "raw_latent_channels": RAW_LATENT_CHANNELS,
            "latent_patch_size": LATENT_PATCH_SIZE,
            "spatial_compression": FLUX2_SPATIAL_COMPRESSION,
            "latent_shape": list(self.latent_shape),
        }


def load_flux2_codec(
    config: VAEConfig,
    *,
    image_size: int,
    device: torch.device | str,
) -> Flux2Codec:
    from diffusers import AutoencoderKLFlux2

    device = torch.device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKLFlux2.from_pretrained(
        config.model_name,
        revision=config.revision,
        subfolder=config.subfolder,
        torch_dtype=dtype,
    )
    return Flux2Codec(
        vae.to(device),
        image_size=image_size,
        micro_batch_size=config.micro_batch_size,
    )
