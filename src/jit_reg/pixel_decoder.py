from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn

from .config import DINO_CHANNELS, DINO_GRID_SIZE, IMAGE_SIZE, PixelDecoderConfig
from .model.ddt import DDTEncoderBlock, RMSNorm


class TransformerPixelDecoder(nn.Module):
    """Decode a 16x16 grid of DINO patch tokens directly into RGB patches."""

    def __init__(self, config: PixelDecoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or PixelDecoderConfig()
        self.config.validate()
        self.input_projection = nn.Linear(DINO_CHANNELS, self.config.width)
        self.blocks = nn.ModuleList(
            DDTEncoderBlock(
                self.config.width,
                self.config.num_heads,
                self.config.mlp_ratio,
                DINO_GRID_SIZE,
                0,
            )
            for _ in range(self.config.depth)
        )
        self.norm = RMSNorm(self.config.width)
        self.output_projection = nn.Linear(
            self.config.width, 3 * self.config.output_patch_size**2
        )
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.zeros_(self.output_projection.bias)

    def _unpatchify(self, patches: Tensor) -> Tensor:
        batch = patches.shape[0]
        side = DINO_GRID_SIZE
        patch = self.config.output_patch_size
        expected = (batch, side * side, 3 * patch**2)
        if tuple(patches.shape) != expected:
            raise ValueError(f"decoder patch output must have shape {expected}")
        return (
            patches.reshape(batch, side, side, patch, patch, 3)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, 3, side * patch, side * patch)
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tuple(tokens.shape[1:]) != (DINO_CHANNELS, DINO_GRID_SIZE, DINO_GRID_SIZE):
            raise ValueError(
                f"pixel decoder tokens must have shape [B,{DINO_CHANNELS},{DINO_GRID_SIZE},{DINO_GRID_SIZE}]"
            )
        states = tokens.flatten(2).transpose(1, 2)
        states = self.input_projection(states)
        for block in self.blocks:
            states = block(states)
        patches = self.output_projection(self.norm(states))
        return self._unpatchify(patches).sigmoid()


def _spectral_conv(
    in_channels: int,
    out_channels: int,
    *,
    stride: int,
    normalize: bool = True,
) -> list[nn.Module]:
    conv = nn.utils.spectral_norm(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)
    )
    layers: list[nn.Module] = [conv]
    if normalize:
        layers.append(nn.GroupNorm(1, out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class PatchDiscriminator(nn.Module):
    """Single-scale spectral-normalized PatchGAN discriminator."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        channels = [base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels]
        layers = _spectral_conv(3, channels[0], stride=2, normalize=False)
        for left, right in pairwise(channels):
            layers.extend(_spectral_conv(left, right, stride=2))
        layers.extend(_spectral_conv(channels[-1], channels[-1], stride=1))
        layers.append(
            nn.utils.spectral_norm(nn.Conv2d(channels[-1], 1, kernel_size=4, padding=1))
        )
        self.net = nn.Sequential(*layers)

    def forward(self, pixels: Tensor) -> Tensor:
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"discriminator images must have shape [B,3,{IMAGE_SIZE},{IMAGE_SIZE}]")
        return self.net(pixels)


def discriminator_hinge_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    return 0.5 * (
        F.relu(1 - real_logits.float()).mean() + F.relu(1 + fake_logits.float()).mean()
    )


def generator_hinge_loss(fake_logits: Tensor) -> Tensor:
    return -fake_logits.float().mean()


@dataclass(frozen=True)
class ReconstructionLosses:
    total: Tensor
    l1: Tensor
    lpips: Tensor
    adversarial: Tensor


class ReconstructionObjective(nn.Module):
    def __init__(self, config: PixelDecoderConfig) -> None:
        super().__init__()
        self.config = config
        if config.loss.lpips_weight > 0:
            import lpips

            self.perceptual: nn.Module | None = lpips.LPIPS(net="alex").eval()
            for parameter in self.perceptual.parameters():
                parameter.requires_grad_(False)
        else:
            self.perceptual = None

    def train(self, mode: bool = True) -> ReconstructionObjective:
        super().train(mode)
        if self.perceptual is not None:
            self.perceptual.eval()
        return self

    def forward(
        self,
        reconstructed: Tensor,
        target: Tensor,
        *,
        fake_logits: Tensor | None = None,
    ) -> ReconstructionLosses:
        l1 = F.l1_loss(reconstructed.float(), target.float())
        if self.perceptual is None:
            perceptual = l1.new_zeros(())
        else:
            perceptual = self.perceptual(
                2 * reconstructed.float() - 1,
                2 * target.float() - 1,
            ).mean()
        adversarial = (
            generator_hinge_loss(fake_logits) if fake_logits is not None else l1.new_zeros(())
        )
        weights = self.config.loss
        total = (
            weights.l1_weight * l1
            + weights.lpips_weight * perceptual
            + weights.gan_weight * adversarial
        )
        return ReconstructionLosses(total, l1, perceptual, adversarial)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def decoder_weights_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    if path.is_dir():
        for name in ("ema.safetensors", "decoder.safetensors", "model.safetensors"):
            candidate = path / name
            if candidate.is_file():
                return candidate
        if (path / "last").is_dir() or (path / "last").is_symlink():
            return decoder_weights_path(path / "last")
    if not path.is_file():
        raise FileNotFoundError(f"decoder checkpoint is missing: {path}")
    return path


def load_pixel_decoder(
    config: PixelDecoderConfig,
    device: torch.device | str,
) -> tuple[TransformerPixelDecoder, str]:
    if config.checkpoint is None:
        raise ValueError("decoder.checkpoint is required for diffusion sampling")
    weights = decoder_weights_path(config.checkpoint)
    target = torch.device(device)
    dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
    model = TransformerPixelDecoder(config).to(target, dtype=dtype)
    model.load_state_dict(load_file(str(weights), device=str(torch.device(device))))
    model.eval()
    return model, _sha256(weights)
