from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import IMAGE_SIZE, TOKEN_COUNT, TOKEN_DIM, CompileConfig, EncoderConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def encoder_fingerprint(config: EncoderConfig, layers: Sequence[int]) -> str:
    payload = {
        **asdict(config),
        "layers": list(layers),
        "input_normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "view": "128px-bicubic-no-augmentation",
        "selection": "one-indexed-intermediate-layers-after-block",
        "output": "final-norm-patch-tokens-with-cls-and-registers-removed",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class DINOMultiLayerEncoder(nn.Module):
    """Frozen DINOv3 encoder that obtains several layers in a single pass."""

    def __init__(
        self, model: nn.Module, config: EncoderConfig, layers: Sequence[int]
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.config = config
        self.layers = tuple(layers)
        if not self.layers or len(self.layers) != len(set(self.layers)):
            raise ValueError("DINO layers must be a non-empty unique sequence")
        self.register_buffer(
            "pixel_mean",
            torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1),
            persistent=False,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._execution: Callable[[Tensor], dict[int, Tensor]] = self._forward_impl

    def train(self, mode: bool = True) -> DINOMultiLayerEncoder:
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
        source_uint8 = pixels.dtype == torch.uint8
        values = pixels.float()
        if source_uint8:
            values = values.div(255)
        if values.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            values = F.interpolate(
                values,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return (values - self.pixel_mean) / self.pixel_std

    def _blocks(self) -> nn.ModuleList:
        blocks = getattr(self.model, "layer", None)
        if blocks is None and hasattr(self.model, "encoder"):
            blocks = getattr(self.model.encoder, "layer", None)
        if blocks is None:
            raise TypeError("DINO model must expose layer or encoder.layer")
        return blocks

    def _patch_tokens(self, states: Tensor) -> Tensor:
        prefix = 1 + self.config.num_register_tokens
        patches = states[:, prefix:]
        if patches.shape[1:] != (TOKEN_COUNT, TOKEN_DIM):
            raise ValueError(
                "DINO patch output must be [B,64,768] after CLS/register removal; "
                f"received {tuple(patches.shape)}"
            )
        return patches

    def _forward_impl(self, pixels: Tensor) -> dict[int, Tensor]:
        values = self._prepare(pixels)
        embeddings = self.model.embeddings
        patch_embeddings = getattr(embeddings, "patch_embeddings", None)
        weight = getattr(patch_embeddings, "weight", None)
        if weight is None and hasattr(patch_embeddings, "projection"):
            weight = patch_embeddings.projection.weight
        if weight is not None:
            values = values.to(weight.dtype)
        states = embeddings(values, bool_masked_pos=None)
        if not isinstance(states, Tensor):
            states = states[0]
        positions = self.model.rope_embeddings(values)
        selected: dict[int, Tensor] = {}
        wanted = set(self.layers)
        for one_indexed, block in enumerate(self._blocks(), start=1):
            states = block(states, position_embeddings=positions)
            if not isinstance(states, Tensor):
                states = states[0]
            if one_indexed in wanted:
                normalized = self.model.norm(states) if self.config.apply_final_norm else states
                selected[one_indexed] = self._patch_tokens(normalized)
        if set(selected) != wanted:
            missing = sorted(wanted - set(selected))
            raise ValueError(f"DINO model does not contain requested one-indexed layers {missing}")
        return {layer: selected[layer] for layer in self.layers}

    @torch.no_grad()
    def forward(self, pixels: Tensor) -> dict[int, Tensor]:
        return self._execution(pixels)

    encode = forward


def load_dino_encoder(
    config: EncoderConfig,
    layers: Sequence[int],
    device: torch.device | str,
    *,
    compile_config: CompileConfig | None = None,
) -> DINOMultiLayerEncoder:
    from transformers import DINOv3ViTModel

    target = torch.device(device)
    dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
    model = DINOv3ViTModel.from_pretrained(
        config.repository,
        revision=config.revision,
        torch_dtype=dtype,
    ).to(target)
    actual = model.config
    expected: Mapping[str, int] = {
        "hidden_size": config.hidden_size,
        "patch_size": config.patch_size,
        "num_register_tokens": config.num_register_tokens,
    }
    for name, value in expected.items():
        if getattr(actual, name, None) != value:
            raise ValueError(f"pinned DINO config mismatch for {name}")
    encoder = DINOMultiLayerEncoder(model, config, layers).to(target)
    if compile_config is not None:
        encoder.compile(compile_config)
    return encoder


DINOSemanticEncoder = DINOMultiLayerEncoder
