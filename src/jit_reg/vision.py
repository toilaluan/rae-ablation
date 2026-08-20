from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict

import torch
from torch import Tensor, nn

from .config import IMAGE_SIZE, CompileConfig, VisionConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def vision_fingerprint(config: VisionConfig) -> str:
    payload = {
        **asdict(config),
        "compile": None,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "tokens": "patch-only-after-shared-final-layernorm",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class DINOEndpointEncoder(nn.Module):
    """Frozen DINOv3-B/16 that captures normalized patch endpoints in one traversal."""

    def __init__(
        self,
        model: nn.Module,
        config: VisionConfig,
        *,
        include_semantic: bool = True,
    ) -> None:
        super().__init__()
        config.validate()
        self.model = model.eval()
        self.config = config
        self.include_semantic = include_semantic
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
        self._execution: Callable[[Tensor], tuple[Tensor, Tensor]] = self._forward_impl

    def train(self, mode: bool = True) -> DINOEndpointEncoder:
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

    def _patch_tokens(self, states: Tensor) -> Tensor:
        prefix = 1 + self.config.num_register_tokens
        patches = states[:, prefix:]
        side = self.config.grid_size
        if tuple(patches.shape[1:]) != (side * side, self.config.hidden_size):
            raise ValueError(
                "DINO patch output has the wrong shape: "
                f"{tuple(patches.shape)}, expected [B,{side * side},{self.config.hidden_size}]"
            )
        return patches.transpose(1, 2).reshape(
            patches.shape[0], self.config.hidden_size, side, side
        )

    def _prepare(self, pixels: Tensor) -> Tensor:
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"DINO images must have shape [B,3,{IMAGE_SIZE},{IMAGE_SIZE}]")
        values = pixels.float()
        if pixels.dtype == torch.uint8:
            values = values / 255
        return (values - self.pixel_mean) / self.pixel_std

    def _forward_impl(self, pixels: Tensor) -> tuple[Tensor, Tensor]:
        values = self._prepare(pixels)
        weight = self.model.embeddings.patch_embeddings.weight
        values = values.to(dtype=weight.dtype)
        states = self.model.embeddings(values, bool_masked_pos=None)
        positions = self.model.rope_embeddings(values)
        target: Tensor | None = None
        stop = self.config.semantic_layer if self.include_semantic else self.config.target_layer
        for index, block in enumerate(self.model.layer[:stop], start=1):
            states = block(states, position_embeddings=positions)
            if index == self.config.target_layer:
                target = self.model.norm(states)
        if target is None:
            raise RuntimeError("DINO target layer was not reached")
        semantic = self.model.norm(states) if self.include_semantic else target
        return self._patch_tokens(target), self._patch_tokens(semantic)

    @torch.no_grad()
    def forward(self, pixels: Tensor) -> tuple[Tensor, Tensor]:
        return self._execution(pixels)

    def encode_target(self, pixels: Tensor) -> Tensor:
        target, _semantic = self(pixels)
        return target

    def encode_pair(self, pixels: Tensor) -> tuple[Tensor, Tensor]:
        if not self.include_semantic:
            raise RuntimeError("this DINO encoder was configured for target-only extraction")
        return self(pixels)


def load_dino_encoder(
    config: VisionConfig,
    device: torch.device | str,
    *,
    include_semantic: bool = True,
) -> DINOEndpointEncoder:
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
        "hidden_size": config.hidden_size,
        "patch_size": config.patch_size,
        "num_hidden_layers": config.semantic_layer,
        "num_register_tokens": config.num_register_tokens,
    }
    for name, value in expected.items():
        if getattr(actual, name, None) != value:
            raise ValueError(f"pinned DINO config mismatch for {name}")
    encoder = DINOEndpointEncoder(model, config, include_semantic=include_semantic).to(target)
    encoder.compile(config.compile)
    return encoder
