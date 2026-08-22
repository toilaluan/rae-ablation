from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import SEMANTIC_IMAGE_SIZE, CompileConfig, SemanticTokenConfig
from .semantic_statistics import SemanticTokenNormalizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def semantic_encoder_fingerprint(config: SemanticTokenConfig) -> str:
    values = asdict(config)
    values.pop("statistics_path", None)
    values.pop("cache", None)
    values.pop("compile", None)
    payload = {
        **values,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "view": "256px-dataset-view-resized-to-128px-bicubic",
        "tokens": "unpooled-final-layernorm-patch-tokens",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class DINOSemanticEncoder(nn.Module):
    """Frozen DINOv3-B/16 final unpooled patch-token encoder."""

    def __init__(
        self,
        model: nn.Module,
        config: SemanticTokenConfig,
        *,
        normalizer: SemanticTokenNormalizer | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.model = model.eval()
        self.config = config
        self.normalizer = normalizer
        self.register_buffer(
            "pixel_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._execution: Callable[[Tensor], Tensor] = self._forward_impl

    def train(self, mode: bool = True) -> DINOSemanticEncoder:
        del mode
        super().train(False)
        self.model.eval()
        return self

    def compile(self, config: CompileConfig) -> None:
        if config.enabled:
            self._execution = torch.compile(
                self._forward_impl,
                backend=config.backend,
                mode=config.mode,
                fullgraph=config.fullgraph,
                dynamic=config.dynamic,
            )

    def _prepare(self, pixels: Tensor) -> Tensor:
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError("DINO images must have shape [B,3,H,W]")
        values = pixels.float()
        if pixels.dtype == torch.uint8:
            values = values / 255
        if tuple(values.shape[-2:]) != (SEMANTIC_IMAGE_SIZE, SEMANTIC_IMAGE_SIZE):
            values = F.interpolate(
                values,
                size=(SEMANTIC_IMAGE_SIZE, SEMANTIC_IMAGE_SIZE),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return (values - self.pixel_mean) / self.pixel_std

    def _patch_tokens(self, states: Tensor) -> Tensor:
        prefix = 1 + self.config.num_register_tokens
        patches = states[:, prefix:]
        side = self.config.grid_size
        expected = (side * side, self.config.channels)
        if tuple(patches.shape[1:]) != expected:
            raise ValueError(
                f"DINO patch output has shape {tuple(patches.shape)}, expected [B,{expected[0]},{expected[1]}]"
            )
        return patches.transpose(1, 2).reshape(
            patches.shape[0], self.config.channels, side, side
        )

    def _forward_impl(self, pixels: Tensor) -> Tensor:
        values = self._prepare(pixels)
        weight = self.model.embeddings.patch_embeddings.weight
        values = values.to(dtype=weight.dtype)
        states = self.model.embeddings(values, bool_masked_pos=None)
        positions = self.model.rope_embeddings(values)
        for block in self.model.layer:
            states = block(states, position_embeddings=positions)
        semantic_tokens = self._patch_tokens(self.model.norm(states))
        if self.normalizer is not None:
            semantic_tokens = self.normalizer.normalize(semantic_tokens)
        return semantic_tokens

    @torch.no_grad()
    def forward(self, pixels: Tensor) -> Tensor:
        return self._execution(pixels)

    encode = forward


def load_dino_semantic_encoder(
    config: SemanticTokenConfig,
    device: torch.device | str,
    *,
    normalizer: SemanticTokenNormalizer | None = None,
) -> DINOSemanticEncoder:
    from transformers import DINOv3ViTModel

    target = torch.device(device)
    dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
    model = DINOv3ViTModel.from_pretrained(
        config.repository,
        revision=config.revision,
        torch_dtype=dtype,
    ).to(target, dtype=dtype)
    actual = model.config
    expected = {
        "hidden_size": config.channels,
        "patch_size": config.patch_size,
        "num_register_tokens": config.num_register_tokens,
    }
    for name, value in expected.items():
        if getattr(actual, name, None) != value:
            raise ValueError(f"pinned DINO config mismatch for {name}")
    encoder = DINOSemanticEncoder(model, config, normalizer=normalizer).to(target)
    encoder.compile(config.compile)
    return encoder
