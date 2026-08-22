from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .config import IMAGE_LATENT_SHAPE, IMAGE_SIZE, CompileConfig, ImageLatentConfig
from .mage_vae import MageVAE


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mage_vae_provenance(
    config: ImageLatentConfig, weight_sha256: str
) -> dict[str, Any]:
    config.validate()
    if not weight_sha256:
        raise ValueError("Mage-VAE weight identity cannot be empty")
    return {
        "repository": config.repository,
        "revision": config.revision,
        "weight_file": config.weight_file,
        "weight_sha256": weight_sha256,
        "posterior": "mean",
        "deterministic": True,
        "image_shape": [3, IMAGE_SIZE, IMAGE_SIZE],
        "image_latent_shape": list(IMAGE_LATENT_SHAPE),
        "encode_preprocessing": "native_float_[-1,1]; dataset_uint8_to_[-1,1]",
        "decode_postprocessing": "[-1,1]_to_clamped_[0,1]",
        "patch_packing": False,
        "normalization": None,
    }


class MageVAECodec(nn.Module):
    """Inference-only deterministic native Mage-VAE image codec."""

    def __init__(
        self,
        model: nn.Module,
        config: ImageLatentConfig | None = None,
        *,
        weight_sha256: str = "in-memory",
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.config = config or ImageLatentConfig()
        self.config.validate()
        self.weight_sha256 = weight_sha256
        if hasattr(self.model, "sample_posterior"):
            self.model.sample_posterior = False
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._encode_execution: Callable[[Tensor], Tensor] = self._encode_impl
        self._decode_execution: Callable[[Tensor], Tensor] = self._decode_impl

    @property
    def provenance(self) -> dict[str, Any]:
        return mage_vae_provenance(self.config, self.weight_sha256)

    def train(self, mode: bool = True) -> MageVAECodec:
        del mode
        super().train(False)
        self.model.eval()
        return self

    def compile(self, config: CompileConfig) -> None:
        if config.enabled:
            options = {
                "backend": config.backend,
                "mode": config.mode,
                "fullgraph": config.fullgraph,
                "dynamic": config.dynamic,
            }
            self._encode_execution = torch.compile(self._encode_impl, **options)
            self._decode_execution = torch.compile(self._decode_impl, **options)

    def _model_dtype(self) -> torch.dtype:
        parameter = next(self.model.parameters(), None)
        return parameter.dtype if parameter is not None else torch.float32

    def _encode_impl(self, images: Tensor) -> Tensor:
        image_latents = self.model.encode(images.to(self._model_dtype()))
        if tuple(image_latents.shape[1:]) != IMAGE_LATENT_SHAPE:
            raise ValueError(
                f"native Mage-VAE returned {tuple(image_latents.shape)}, expected [B,128,16,16]"
            )
        return image_latents

    def _decode_impl(self, image_latents: Tensor) -> Tensor:
        decoded = self.model.decode(image_latents.to(self._model_dtype()))
        if tuple(decoded.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"native Mage-VAE decoder returned {tuple(decoded.shape)}, expected [B,3,256,256]"
            )
        return decoded.float().add(1).mul(0.5).clamp(0, 1)

    @torch.inference_mode()
    def encode(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Mage-VAE images must have shape [B,3,{IMAGE_SIZE},{IMAGE_SIZE}]")
        if not images.is_floating_point() or torch.any((images < -1) | (images > 1)):
            raise ValueError("Mage-VAE encode input must be floating point in [-1,1]")
        return self._encode_execution(images)

    @torch.inference_mode()
    def encode_pixels(self, pixels: Tensor) -> Tensor:
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"dataset images must have shape [B,3,{IMAGE_SIZE},{IMAGE_SIZE}]")
        values = pixels.float()
        if pixels.dtype == torch.uint8:
            values = values / 255
        if torch.any((values < 0) | (values > 1)):
            raise ValueError("dataset images must be uint8 or floating point in [0,1]")
        return self._encode_execution(values.mul(2).sub(1))

    @torch.inference_mode()
    def decode(self, image_latents: Tensor) -> Tensor:
        if image_latents.ndim != 4 or tuple(image_latents.shape[1:]) != IMAGE_LATENT_SHAPE:
            raise ValueError("Mage-VAE image latents must have shape [B,128,16,16]")
        return self._decode_execution(image_latents)

    forward = encode


def load_mage_vae_codec(
    config: ImageLatentConfig,
    device: torch.device | str,
) -> MageVAECodec:
    from huggingface_hub import hf_hub_download

    config.validate()
    path = hf_hub_download(
        repo_id=config.repository,
        filename=config.weight_file,
        revision=config.revision,
    )
    target = torch.device(device)
    dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
    model = MageVAE(path, sample_posterior=False).to(target, dtype=dtype)
    codec = MageVAECodec(model, config, weight_sha256=file_sha256(path)).to(target)
    codec.compile(config.compile)
    return codec
