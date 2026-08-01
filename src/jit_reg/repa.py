from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import REPAConfig


def resize_spatial_features(spatial: Tensor, target_grid: int) -> Tensor:
    if spatial.ndim != 3 or target_grid < 1:
        raise ValueError("spatial must have shape [B, tokens, channels]")
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
        ).flatten(2).transpose(1, 2)
    return F.normalize(features, dim=-1)


class DINOv3FeatureExtractor(nn.Module):
    native_image_size = 256

    def __init__(self, config: REPAConfig) -> None:
        super().__init__()
        import timm

        self.config = config
        self.encoder = timm.create_model(config.encoder.model_name, pretrained=True)
        resolved = timm.data.resolve_model_data_config(self.encoder)
        self.register_buffer(
            "mean",
            torch.tensor(resolved.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(resolved.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1),
            persistent=False,
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> DINOv3FeatureExtractor:
        super().train(False)
        self.encoder.eval()
        return self

    @staticmethod
    def _sequence(output: object) -> Tensor:
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
        output = self._sequence(
            self.encoder.forward_features((pixels - self.mean) / self.std)
        )
        expected_spatial = (self.native_image_size // 16) ** 2
        if output.ndim != 3:
            raise ValueError("DINOv3 must return a token sequence")
        if output.shape[1] == expected_spatial:
            spatial = output
        else:
            expected = 1 + self.config.encoder.register_tokens + expected_spatial
            if output.shape[1] != expected:
                raise ValueError(f"DINOv3 must return {expected} tokens")
            spatial = output[:, 1 + self.config.encoder.register_tokens :]
        return spatial.float()

    @torch.no_grad()
    def targets(self, images: Tensor, target_grid: int) -> Tensor:
        return resize_spatial_features(self.extract(images), target_grid)
