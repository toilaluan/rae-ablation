from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import REGConfig


class REGTargets(NamedTuple):
    cls: Tensor
    spatial: Tensor
    sequence: Tensor


def resize_spatial_features(spatial: Tensor, target_grid: int) -> Tensor:
    """Resize row-major square spatial tokens in float32 and L2-normalize each token."""
    if spatial.ndim != 3 or target_grid < 1:
        raise ValueError("spatial must have shape [B, tokens, channels] and a positive grid")
    source_grid = math.isqrt(spatial.shape[1])
    if source_grid * source_grid != spatial.shape[1]:
        raise ValueError("spatial token count must form a square grid")
    features = spatial.float()
    if source_grid != target_grid:
        features = features.transpose(1, 2).reshape(
            spatial.shape[0], spatial.shape[2], source_grid, source_grid
        )
        features = F.interpolate(
            features,
            size=(target_grid, target_grid),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        features = features.flatten(2).transpose(1, 2)
    return F.normalize(features, dim=-1)


def prepare_reg_targets(cls: Tensor, spatial: Tensor, target_grid: int) -> REGTargets:
    if cls.ndim != 2 or spatial.ndim != 3:
        raise ValueError("CLS and spatial features must have shapes [B, D] and [B, N, D]")
    if cls.shape[0] != spatial.shape[0] or cls.shape[1] != spatial.shape[2]:
        raise ValueError("CLS and spatial feature dimensions do not match")
    resized = resize_spatial_features(spatial, target_grid)
    return REGTargets(cls=cls, spatial=resized, sequence=torch.cat((cls[:, None], resized), 1))


def negative_cosine_repa_loss(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("REPA prediction and target shapes must match")
    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return -(prediction * target).sum(dim=-1).mean()


class DINOv3FeatureExtractor(nn.Module):
    """Frozen timm DINOv3 feature target with resolved native preprocessing."""

    native_image_size = 256

    def __init__(self, config: REGConfig) -> None:
        super().__init__()
        import timm

        self.config = config
        self.encoder = timm.create_model(config.model_name, pretrained=True)
        resolved = timm.data.resolve_model_data_config(self.encoder)
        mean = resolved.get("mean", (0.485, 0.456, 0.406))
        std = resolved.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> DINOv3FeatureExtractor:
        super().train(False)
        self.encoder.eval()
        return self

    def _sequence(self, output: object) -> tuple[Tensor, Tensor] | Tensor:
        if isinstance(output, Tensor):
            return output
        if isinstance(output, dict):
            cls = output.get("x_norm_clstoken")
            patches = output.get("x_norm_patchtokens")
            if isinstance(cls, Tensor) and isinstance(patches, Tensor):
                return cls, patches
            for key in ("x_norm", "tokens", "last_hidden_state", "x"):
                value = output.get(key)
                if isinstance(value, Tensor):
                    return value
        raise TypeError("DINOv3 forward_features returned an unsupported value")

    @torch.no_grad()
    def extract(self, images: Tensor) -> tuple[Tensor, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("DINOv3 images must have shape [B, 3, H, W]")
        pixels = images.float()
        if pixels.shape[-2:] != (self.native_image_size, self.native_image_size):
            pixels = F.interpolate(
                pixels,
                size=(self.native_image_size, self.native_image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        normalized = (pixels - self.mean) / self.std
        output = self._sequence(self.encoder.forward_features(normalized))
        if isinstance(output, tuple):
            cls, spatial = output
        else:
            expected_spatial = (self.native_image_size // 16) ** 2
            expected_tokens = 1 + self.config.register_tokens + expected_spatial
            if output.ndim != 3 or output.shape[1] != expected_tokens:
                raise ValueError(
                    f"DINOv3 must return {expected_tokens} tokens, received {tuple(output.shape)}"
                )
            cls = output[:, 0]
            spatial = output[:, 1 + self.config.register_tokens :]
        expected_spatial = (self.native_image_size // 16) ** 2
        if (
            cls.shape != (images.shape[0], self.config.feature_dim)
            or spatial.shape != (images.shape[0], expected_spatial, self.config.feature_dim)
        ):
            raise ValueError("DINOv3 CLS or spatial feature shape is incompatible with REG")
        return cls.float(), spatial.float()

    @torch.no_grad()
    def targets(self, images: Tensor, target_grid: int) -> REGTargets:
        cls, spatial = self.extract(images)
        return prepare_reg_targets(cls, spatial, target_grid)
