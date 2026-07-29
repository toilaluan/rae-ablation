from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn


class TimestepTokens(NamedTuple):
    base: Tensor
    tokens: Tensor


class ClassTokens(NamedTuple):
    image_tokens: Tensor
    reg_tokens: Tensor


def classifier_free_labels(
    labels: Tensor,
    num_classes: int,
    probability: float,
    generator: torch.Generator,
) -> Tensor:
    """Replace labels with the shared null class using an explicit RNG stream."""
    if labels.ndim != 1:
        raise ValueError("labels must have shape [B]")
    if not 0 <= probability < 1:
        raise ValueError("label dropout probability must be in [0, 1)")
    if probability == 0:
        return labels
    dropped = (
        torch.rand(
            labels.shape,
            device=labels.device,
            generator=generator,
        )
        < probability
    )
    return torch.where(
        dropped,
        torch.full_like(labels, num_classes),
        labels,
    )


class TimestepTokenEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_tokens: int = 1,
        embedding_size: int = 256,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if (
            hidden_size < 1
            or num_tokens < 0
            or embedding_size < 1
            or scale <= 0
        ):
            raise ValueError(
                "timestep embedding dimensions must be valid"
            )
        self.frequencies = nn.Parameter(
            torch.normal(0, scale, (embedding_size,)),
            requires_grad=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.offsets = nn.Parameter(
            torch.normal(
                0,
                hidden_size**-0.5,
                (num_tokens, hidden_size),
            )
        )

    def forward(self, timestep: Tensor) -> TimestepTokens:
        if timestep.ndim != 1:
            raise ValueError("timestep must have shape [B]")
        angles = (
            timestep[:, None].float()
            * self.frequencies[None, :]
            * 2
            * torch.pi
        )
        embedding = torch.cat(
            (torch.sin(angles), torch.cos(angles)),
            dim=-1,
        )
        base = self.mlp(
            embedding.to(self.mlp[0].weight.dtype)
        ).unsqueeze(1)
        return TimestepTokens(
            base=base,
            tokens=base + self.offsets,
        )


class ClassTokenEmbedder(nn.Module):
    """Shared label table with stream-private learned token offsets."""

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        *,
        num_image_tokens: int,
        num_reg_tokens: int,
    ) -> None:
        super().__init__()
        if (
            hidden_size < 1
            or num_classes < 1
            or num_image_tokens < 0
            or num_reg_tokens < 0
        ):
            raise ValueError(
                "class embedding dimensions must be valid"
            )
        self.embedding_table = nn.Embedding(
            num_classes + 1,
            hidden_size,
        )
        self.image_offsets = nn.Parameter(
            torch.normal(
                0,
                hidden_size**-0.5,
                (num_image_tokens, hidden_size),
            )
        )
        self.reg_offsets = nn.Parameter(
            torch.normal(
                0,
                hidden_size**-0.5,
                (num_reg_tokens, hidden_size),
            )
        )

    def base_embedding(self, labels: Tensor) -> Tensor:
        if labels.ndim != 1:
            raise ValueError("labels must have shape [B]")
        return self.embedding_table(labels)

    def forward(self, labels: Tensor) -> ClassTokens:
        base = self.base_embedding(labels).unsqueeze(1)
        return ClassTokens(
            image_tokens=base + self.image_offsets,
            reg_tokens=base + self.reg_offsets,
        )


@dataclass(frozen=True)
class DDTTokenLayout:
    """Immutable authority for the canonical DDT token sequence.

    The full order is ``[REG data, image data, REPA stream, REG timestep,
    REG class, image timestep, image class]``. ``repa_stream_tokens`` is
    declared last to preserve the positional constructor used by schema-v9
    callers.
    """

    reg_data_tokens: int
    image_data_tokens: int
    reg_timestep_tokens: int = 0
    reg_class_tokens: int = 0
    image_timestep_tokens: int = 0
    image_class_tokens: int = 0
    repa_stream_tokens: int = 0

    def __post_init__(self) -> None:
        for name, count in (
            ("reg_data_tokens", self.reg_data_tokens),
            ("image_data_tokens", self.image_data_tokens),
            ("reg_timestep_tokens", self.reg_timestep_tokens),
            ("reg_class_tokens", self.reg_class_tokens),
            ("image_timestep_tokens", self.image_timestep_tokens),
            ("image_class_tokens", self.image_class_tokens),
            ("repa_stream_tokens", self.repa_stream_tokens),
        ):
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
            ):
                raise ValueError(
                    f"{name} must be a non-negative integer"
                )

    @staticmethod
    def _slice(start: int, count: int) -> slice:
        return slice(start, start + count)

    @property
    def reg_data(self) -> slice:
        return self._slice(0, self.reg_data_tokens)

    @property
    def image_data(self) -> slice:
        return self._slice(
            self.reg_data.stop,
            self.image_data_tokens,
        )

    @property
    def repa_stream(self) -> slice:
        return self._slice(
            self.image_data.stop,
            self.repa_stream_tokens,
        )

    @property
    def reg_timestep(self) -> slice:
        return self._slice(
            self.repa_stream.stop,
            self.reg_timestep_tokens,
        )

    @property
    def reg_class(self) -> slice:
        return self._slice(
            self.reg_timestep.stop,
            self.reg_class_tokens,
        )

    @property
    def image_timestep(self) -> slice:
        return self._slice(
            self.reg_class.stop,
            self.image_timestep_tokens,
        )

    @property
    def image_class(self) -> slice:
        return self._slice(
            self.image_timestep.stop,
            self.image_class_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.image_class.stop

    @property
    def data_tokens(self) -> int:
        return (
            self.reg_data_tokens
            + self.image_data_tokens
            + self.repa_stream_tokens
        )

    @property
    def condition_tokens(self) -> int:
        return self.total_tokens - self.data_tokens

    @property
    def reg_stream_tokens(self) -> int:
        return (
            self.reg_data_tokens
            + self.reg_timestep_tokens
            + self.reg_class_tokens
        )

    @property
    def image_stream_tokens(self) -> int:
        return (
            self.image_data_tokens
            + self.repa_stream_tokens
            + self.image_timestep_tokens
            + self.image_class_tokens
        )

    @property
    def reg_stream_slices(self) -> tuple[slice, ...]:
        return (self.reg_data, self.reg_timestep, self.reg_class)

    @property
    def image_stream_slices(self) -> tuple[slice, ...]:
        return (
            self.image_data,
            self.repa_stream,
            self.image_timestep,
            self.image_class,
        )

    @property
    def data_slices(self) -> tuple[slice, ...]:
        return (self.reg_data, self.image_data)

    @property
    def reg_stream_layout(self) -> DDTTokenLayout:
        return DDTTokenLayout(
            reg_data_tokens=self.reg_data_tokens,
            image_data_tokens=0,
            reg_timestep_tokens=self.reg_timestep_tokens,
            reg_class_tokens=self.reg_class_tokens,
        )

    def _gather(
        self,
        x: Tensor,
        slices: tuple[slice, ...],
    ) -> Tensor:
        if x.shape[-2] != self.total_tokens:
            raise ValueError(
                "tensor token count does not match DDTTokenLayout"
            )
        parts = [
            x[..., token_slice, :]
            for token_slice in slices
            if token_slice.stop > token_slice.start
        ]
        if parts:
            return torch.cat(parts, dim=-2)
        return x[..., :0, :]

    def gather_reg_stream(self, x: Tensor) -> Tensor:
        return self._gather(x, self.reg_stream_slices)

    def gather_image_stream(self, x: Tensor) -> Tensor:
        return self._gather(x, self.image_stream_slices)

    def gather_image_conditions(self, x: Tensor) -> Tensor:
        return self._gather(
            x,
            (self.image_timestep, self.image_class),
        )

    def gather_image_data(self, x: Tensor) -> Tensor:
        return self._gather(
            x,
            (self.image_data, self.repa_stream),
        )

    def gather_data(self, x: Tensor) -> Tensor:
        """Gather prediction-bearing REG and image data, excluding REPA memory."""
        return self._gather(x, self.data_slices)

    def pack(
        self,
        *,
        reg_data: Tensor | None = None,
        image_data: Tensor | None = None,
        reg_timestep: Tensor | None = None,
        reg_class: Tensor | None = None,
        image_timestep: Tensor | None = None,
        image_class: Tensor | None = None,
        repa_stream: Tensor | None = None,
    ) -> Tensor:
        values = (
            ("reg_data", self.reg_data_tokens, reg_data),
            ("image_data", self.image_data_tokens, image_data),
            ("repa_stream", self.repa_stream_tokens, repa_stream),
            (
                "reg_timestep",
                self.reg_timestep_tokens,
                reg_timestep,
            ),
            ("reg_class", self.reg_class_tokens, reg_class),
            (
                "image_timestep",
                self.image_timestep_tokens,
                image_timestep,
            ),
            ("image_class", self.image_class_tokens, image_class),
        )
        reference = next(
            (value for _name, _count, value in values if value is not None),
            None,
        )
        if reference is None:
            raise ValueError(
                "at least one tensor is required to pack a token layout"
            )
        parts: list[Tensor] = []
        for name, count, value in values:
            if value is None:
                if count:
                    raise ValueError(
                        f"{name} is required by this token layout"
                    )
                value = reference[..., :0, :]
            if value.shape[-2] != count:
                raise ValueError(
                    f"{name} has {value.shape[-2]} tokens; expected {count}"
                )
            if (
                value.shape[:-2] != reference.shape[:-2]
                or value.shape[-1] != reference.shape[-1]
            ):
                raise ValueError(
                    "all packed token tensors must share batch and hidden shapes"
                )
            parts.append(value)
        return torch.cat(parts, dim=-2)

    def reconstruct_streams(
        self,
        reg_stream: Tensor,
        image_stream: Tensor,
    ) -> Tensor:
        """Reconstruct canonical order from packed REG and image streams."""
        if reg_stream.shape[-2] != self.reg_stream_tokens:
            raise ValueError("REG stream token count does not match layout")
        if image_stream.shape[-2] != self.image_stream_tokens:
            raise ValueError("image stream token count does not match layout")
        if (
            reg_stream.shape[:-2] != image_stream.shape[:-2]
            or reg_stream.shape[-1] != image_stream.shape[-1]
        ):
            raise ValueError(
                "REG and image streams must share batch and hidden shapes"
            )
        reg_data_end = self.reg_data_tokens
        reg_time_end = reg_data_end + self.reg_timestep_tokens
        image_data_end = self.image_data_tokens
        repa_stream_end = image_data_end + self.repa_stream_tokens
        image_time_end = (
            repa_stream_end + self.image_timestep_tokens
        )
        return self.pack(
            reg_data=reg_stream[..., :reg_data_end, :],
            image_data=image_stream[..., :image_data_end, :],
            repa_stream=image_stream[
                ..., image_data_end:repa_stream_end, :
            ],
            reg_timestep=reg_stream[
                ..., reg_data_end:reg_time_end, :
            ],
            reg_class=reg_stream[..., reg_time_end:, :],
            image_timestep=image_stream[
                ..., repa_stream_end:image_time_end, :
            ],
            image_class=image_stream[..., image_time_end:, :],
        )
