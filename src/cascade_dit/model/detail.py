from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import PATCH_SIZE, TOKEN_COUNT, TOKEN_DIM


class ResidualTokenMLP(nn.Module):
    def __init__(self, width: int = TOKEN_DIM, expansion: float = 2.0) -> None:
        super().__init__()
        hidden = int(width * expansion)
        self.norm = nn.RMSNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, width),
        )

    def forward(self, values: Tensor) -> Tensor:
        return values + self.mlp(self.norm(values))


class TokenDetailHead(nn.Module):
    """Reusable pointwise detail residual in canonical token coordinates."""

    def __init__(
        self,
        token_dim: int = TOKEN_DIM,
        context_dim: int = TOKEN_DIM,
        *,
        depth: int = 2,
        expansion: float = 2.0,
    ) -> None:
        super().__init__()
        if depth != 2:
            raise ValueError("the token detail head uses exactly two residual MLP blocks")
        self.noisy_projection = nn.Linear(token_dim, token_dim)
        self.context_projection = nn.Linear(context_dim, token_dim)
        self.blocks = nn.ModuleList(
            ResidualTokenMLP(token_dim, expansion) for _ in range(depth)
        )
        self.output_projection = nn.Linear(token_dim, token_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, noisy_tokens: Tensor, context: Tensor) -> Tensor:
        values = self.noisy_projection(noisy_tokens) + self.context_projection(context)
        for block in self.blocks:
            values = block(values)
        return self.output_projection(values)


class _DownBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(1, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 4, stride=2, padding=1),
            nn.GroupNorm(1, output_channels),
            nn.SiLU(),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.block(values)


class _UpBlock(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, skip_channels: int = 0
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, 4, stride=2, padding=1),
            nn.GroupNorm(1, output_channels),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(output_channels + skip_channels, output_channels, 3, padding=1),
            nn.GroupNorm(1, output_channels),
            nn.SiLU(),
        )

    def forward(self, values: Tensor, skip: Tensor | None = None) -> Tensor:
        values = self.up(values)
        if skip is not None:
            if values.shape[-2:] != skip.shape[-2:]:
                raise ValueError("pixel-detail skip resolution mismatch")
            values = torch.cat((values, skip), dim=1)
        return self.fuse(values)


class PixelPatchDetailer(nn.Module):
    """DiP-style U-Net applied to each 3x16x16 patch independently."""

    resolutions = (16, 8, 4, 2, 1, 2, 4, 8, 16)
    channel_progression = (3, 64, 128, 256, 512)

    def __init__(
        self,
        context_dim: int = TOKEN_DIM,
        channels: tuple[int, int, int, int] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        if tuple(channels) != (64, 128, 256, 512):
            raise ValueError("pixel detail channels must be 64,128,256,512")
        c1, c2, c3, c4 = channels
        self.down_blocks = nn.ModuleList(
            (
                _DownBlock(3, c1),
                _DownBlock(c1, c2),
                _DownBlock(c2, c3),
                _DownBlock(c3, c4),
            )
        )
        self.context_projection = (
            nn.Identity() if context_dim == TOKEN_DIM else nn.Linear(context_dim, TOKEN_DIM)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c4 + TOKEN_DIM, c4, 1),
            nn.GroupNorm(1, c4),
            nn.SiLU(),
        )
        self.up_blocks = nn.ModuleList(
            (
                _UpBlock(c4, c3, c3),
                _UpBlock(c3, c2, c2),
                _UpBlock(c2, c1, c1),
                _UpBlock(c1, c1),
            )
        )
        self.output = nn.Conv2d(c1, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy_patches: Tensor, context: Tensor) -> Tensor:
        if noisy_patches.ndim != 3 or noisy_patches.shape[1:] != (TOKEN_COUNT, TOKEN_DIM):
            raise ValueError("pixel detail input must have shape [B,64,768]")
        if context.shape[:2] != noisy_patches.shape[:2]:
            raise ValueError("pixel detail context must align with patches")
        batch = noisy_patches.shape[0]
        values = noisy_patches.reshape(batch * TOKEN_COUNT, 3, PATCH_SIZE, PATCH_SIZE)
        skips: list[Tensor] = []
        for block in self.down_blocks:
            values = block(values)
            skips.append(values)
        embedded_context = self.context_projection(context).reshape(
            batch * TOKEN_COUNT, TOKEN_DIM, 1, 1
        )
        values = self.bottleneck(torch.cat((values, embedded_context), dim=1))
        values = self.up_blocks[0](values, skips[2])
        values = self.up_blocks[1](values, skips[1])
        values = self.up_blocks[2](values, skips[0])
        values = self.up_blocks[3](values)
        residual = self.output(values)
        return residual.reshape(batch, TOKEN_COUNT, TOKEN_DIM)
