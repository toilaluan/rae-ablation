from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from torch import Tensor


@dataclass(frozen=True)
class LatentSpec:
    channels: int
    height: int
    width: int
    dtype: str = "float16"

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.channels, self.height, self.width


@dataclass(frozen=True)
class CodecProvenance:
    name: str
    repository: str
    revision: str
    base_repository: str | None = None
    base_revision: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


class LatentCodec(ABC):
    @property
    @abstractmethod
    def latent_spec(self) -> LatentSpec: ...

    @property
    @abstractmethod
    def provenance(self) -> CodecProvenance: ...

    @abstractmethod
    def encode(self, images: Tensor) -> Tensor: ...

    @abstractmethod
    def decode(self, latents: Tensor) -> Tensor: ...
