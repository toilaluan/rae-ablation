from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import FeaturesConfig


class FeatureTargets(NamedTuple):
    reg: Tensor | None
    repa: Tensor | None


def scale_reg_targets(
    targets: FeatureTargets, target_std: float
) -> FeatureTargets:
    """Apply the training-only REG scale to canonical online or cached targets."""
    if not math.isfinite(target_std) or target_std <= 0:
        raise ValueError("REG target standard deviation must be finite and positive")
    return FeatureTargets(
        targets.reg * target_std if targets.reg is not None else None,
        targets.repa,
    )


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


def pool_spatial_features(spatial: Tensor, pool_size: int) -> Tensor:
    """Adaptively pool square spatial tokens and apply parameter-free LayerNorm."""
    if spatial.ndim != 3 or pool_size < 1:
        raise ValueError("spatial must have shape [B, tokens, channels] and a positive pool size")
    source_grid = math.isqrt(spatial.shape[1])
    if source_grid * source_grid != spatial.shape[1]:
        raise ValueError("spatial token count must form a square grid")
    if pool_size > source_grid:
        raise ValueError("pool size cannot exceed the source feature grid")
    features = spatial.float().transpose(1, 2).reshape(
        spatial.shape[0], spatial.shape[2], source_grid, source_grid
    )
    pooled = F.adaptive_avg_pool2d(features, (pool_size, pool_size))
    pooled = pooled.flatten(2).transpose(1, 2)
    return F.layer_norm(pooled, (pooled.shape[-1],))


def prepare_feature_targets(
    spatial: Tensor,
    target_grid: int,
    *,
    include_reg: bool = True,
    pool_size: int = 4,
    include_repa: bool = True,
) -> FeatureTargets:
    if spatial.ndim != 3:
        raise ValueError("spatial features must have shape [B, N, D]")
    reg_tokens = pool_spatial_features(spatial, pool_size) if include_reg else None
    repa = resize_spatial_features(spatial, target_grid) if include_repa else None
    return FeatureTargets(reg=reg_tokens, repa=repa)


def repa_target_for_scope(
    targets: FeatureTargets,
    token_scope: str,
) -> Tensor | None:
    if targets.repa is None:
        return None
    if token_scope == "image":
        return targets.repa
    if token_scope == "image_and_reg":
        if targets.reg is None:
            raise ValueError("REPA image_and_reg targets require clean REG tokens")
        return torch.cat((targets.reg, targets.repa), dim=1)
    raise ValueError("REPA token scope must be image or image_and_reg")


def negative_cosine_repa_loss(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("REPA prediction and target shapes must match")
    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return -(prediction * target).sum(dim=-1).mean()


class DINOv3FeatureExtractor(nn.Module):
    """Frozen timm DINOv3 feature target with resolved native preprocessing."""

    native_image_size = 256

    def __init__(self, config: FeaturesConfig) -> None:
        super().__init__()
        import timm

        self.config = config
        self.encoder = timm.create_model(config.encoder.model_name, pretrained=True)
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

    def _sequence(self, output: object) -> Tensor:
        if isinstance(output, Tensor):
            return output
        if isinstance(output, dict):
            patches = output.get("x_norm_patchtokens")
            if isinstance(patches, Tensor):
                return patches
            for key in ("x_norm", "tokens", "last_hidden_state", "x"):
                value = output.get(key)
                if isinstance(value, Tensor):
                    return value
        raise TypeError("DINOv3 forward_features returned an unsupported value")

    @torch.no_grad()
    def extract(self, images: Tensor) -> Tensor:
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
        expected_spatial = (self.native_image_size // 16) ** 2
        if output.ndim != 3:
            raise ValueError(
                f"DINOv3 must return a token sequence, received {tuple(output.shape)}"
            )
        if output.shape[1] == expected_spatial:
            spatial = output
        else:
            expected_tokens = 1 + self.config.encoder.register_tokens + expected_spatial
            if output.ndim != 3 or output.shape[1] != expected_tokens:
                raise ValueError(
                    f"DINOv3 must return {expected_tokens} tokens, received {tuple(output.shape)}"
                )
            spatial = output[:, 1 + self.config.encoder.register_tokens :]
        if spatial.shape != (
            images.shape[0],
            expected_spatial,
            self.config.encoder.feature_dim,
        ):
            raise ValueError("DINOv3 spatial feature shape is incompatible with REG")
        return spatial.float()

    @torch.no_grad()
    def targets(self, images: Tensor, target_grid: int) -> FeatureTargets:
        spatial = self.extract(images)
        return prepare_feature_targets(
            spatial,
            target_grid,
            include_reg=self.config.reg.enabled,
            pool_size=self.config.reg.pool_size,
            include_repa=self.config.repa.enabled,
        )
