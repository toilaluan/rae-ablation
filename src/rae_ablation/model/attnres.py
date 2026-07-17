from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from fla.ops.attnres import fused_attnres, naive_attnres
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class AttnResWeightRecord:
    site: str
    source_labels: tuple[str, ...]
    spatial: Tensor
    time: Tensor
    classes: Tensor


@dataclass
class AttnResCapture:
    spatial_tokens: int
    time_tokens: int
    class_tokens: int
    num_input_residuals: int
    records: list[AttnResWeightRecord] = field(default_factory=list)

    @staticmethod
    def _mean_tokens(weights: Tensor, begin: int, end: int) -> Tensor:
        if begin == end:
            return torch.full(
                (weights.shape[0],),
                torch.nan,
                device=weights.device,
                dtype=torch.float32,
            )
        return weights[:, :, begin:end].float().mean(dim=(1, 2))

    def record(self, site: str, weights: Tensor) -> None:
        expected_tokens = self.spatial_tokens + self.time_tokens + self.class_tokens
        if weights.ndim != 3 or weights.shape[-1] != expected_tokens:
            raise ValueError(
                "captured AttnRes weights must have shape [sources, batch, encoder_tokens]"
            )
        spatial_end = self.spatial_tokens
        time_end = spatial_end + self.time_tokens
        labels = tuple(
            f"input_projection_{index}"
            if index < self.num_input_residuals
            else f"depth_source_{index - self.num_input_residuals}"
            for index in range(weights.shape[0])
        )
        self.records.append(
            AttnResWeightRecord(
                site=site,
                source_labels=labels,
                spatial=self._mean_tokens(weights, 0, spatial_end),
                time=self._mean_tokens(weights, spatial_end, time_end),
                classes=self._mean_tokens(weights, time_end, expected_tokens),
            )
        )


def _attnres(
    *,
    query: Tensor,
    residuals: Sequence[Tensor],
    rms_weight: Tensor,
    output_rms_weight: Tensor | None,
    rms_eps: float,
    return_weights: bool,
) -> Tensor | tuple[Tensor, Tensor]:
    if residuals[0].is_cuda:
        return fused_attnres(
            query=query,
            residuals=residuals,
            rms_weight=rms_weight,
            output_rms_weight=output_rms_weight,
            rms_eps=rms_eps,
            return_weights=return_weights,
            checkpoint_level=1,
        )
    return naive_attnres(
        query=query,
        residuals=residuals,
        rms_weight=rms_weight,
        output_rms_weight=output_rms_weight,
        rms_eps=rms_eps,
        return_weights=return_weights,
    )


class AttnResRouter(nn.Module):
    def __init__(self, hidden_size: int, site: str, eps: float = 1e-6) -> None:
        super().__init__()
        self.query_projection = nn.Linear(hidden_size, 1, bias=False)
        self.key_norm = nn.RMSNorm(hidden_size, eps=eps)
        self.site = site
        self.capture: AttnResCapture | None = None

    def forward(
        self,
        residuals: tuple[Tensor, ...],
        output_rms_weight: Tensor | None = None,
    ) -> Tensor:
        if self.capture is None:
            output = _attnres(
                query=self.query_projection.weight,
                residuals=residuals,
                rms_weight=self.key_norm.weight,
                output_rms_weight=output_rms_weight,
                rms_eps=self.key_norm.eps,
                return_weights=False,
            )
            if not isinstance(output, Tensor):
                raise TypeError("AttnRes returned weights when they were not requested")
            return output
        output, weights = _attnres(
            query=self.query_projection.weight,
            residuals=residuals,
            rms_weight=self.key_norm.weight,
            output_rms_weight=output_rms_weight,
            rms_eps=self.key_norm.eps,
            return_weights=True,
        )
        self.capture.record(self.site, weights)
        return output


__all__ = [
    "AttnResCapture",
    "AttnResRouter",
    "AttnResWeightRecord",
]
