from __future__ import annotations

from typing import Literal, overload

from torch import Tensor, nn
import torch

from ..constants import COMPACT_REPO, COMPACT_REVISION, RAE_REPO, RAE_REVISION
from .base import CodecProvenance, LatentCodec, LatentSpec


class RAECodec(LatentCodec):
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @property
    def latent_spec(self) -> LatentSpec:
        return LatentSpec(1024, 16, 16)

    @property
    def provenance(self) -> CodecProvenance:
        return CodecProvenance("rae", RAE_REPO, RAE_REVISION)

    def encode(self, images: Tensor) -> Tensor:
        return self.model.encode(images)

    def decode(self, latents: Tensor) -> Tensor:
        return self.model.decode(latents)


class CompactRAECodec(LatentCodec):
    def __init__(self, model: nn.Module, base: RAECodec) -> None:
        self.model = model
        self.base = base

    @property
    def latent_spec(self) -> LatentSpec:
        return LatentSpec(1024, 8, 8)

    @property
    def provenance(self) -> CodecProvenance:
        return CodecProvenance(
            "rae_compact", COMPACT_REPO, COMPACT_REVISION, RAE_REPO, RAE_REVISION
        )

    def compress(self, rae_features: Tensor) -> Tensor:
        result = self.model.compress(rae_features)
        return result[0] if isinstance(result, tuple) else result

    def encode(self, images: Tensor) -> Tensor:
        return self.compress(self.base.encode(images))

    def decode(self, latents: Tensor) -> Tensor:
        return self.model.decode(latents)


@overload
def load_codecs(kind: Literal["rae"], device: str = "cuda") -> RAECodec: ...


@overload
def load_codecs(kind: Literal["rae_compact"], device: str = "cuda") -> CompactRAECodec: ...


@overload
def load_codecs(
    kind: Literal["both"], device: str = "cuda"
) -> tuple[RAECodec, CompactRAECodec]: ...


def load_codecs(
    kind: str, device: str = "cuda"
) -> RAECodec | CompactRAECodec | tuple[RAECodec, CompactRAECodec]:
    """Load pinned codec weights. Combined mode shares the same RAE instance."""
    from transformers import AutoModel

    rae_model = (
        AutoModel.from_pretrained(
            RAE_REPO,
            revision=RAE_REVISION,
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )
    rae_model = torch.compile(rae_model, mode="default", fullgraph=True)
    rae = RAECodec(rae_model)
    if kind == "rae":
        return rae
    compact_model = (
        AutoModel.from_pretrained(
            COMPACT_REPO,
            revision=COMPACT_REVISION,
            trust_remote_code=True,
            rae=rae_model,
        )
        .eval()
        .to(device)
    )
    compact_model = torch.compile(compact_model, mode="default", fullgraph=True)
    compact = CompactRAECodec(compact_model, rae)
    if kind == "rae_compact":
        return compact
    if kind == "both":
        return rae, compact
    raise ValueError(f"unknown codec: {kind}")
