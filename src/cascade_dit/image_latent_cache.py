from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .data import preprocessing_metadata
from .representation_cache import (
    AsyncRepresentationCacheWriter,
    RepresentationCache,
    RepresentationCacheWriter,
    cache_fingerprint,
    cache_path,
    preflight_cache_capacity,
    required_cache_bytes,
)


def image_latent_cache_spec(
    config: ExperimentConfig,
    dataset_size: int,
    mage_weight_identity: str | dict[str, Any],
) -> dict[str, Any]:
    if dataset_size < 1 or not mage_weight_identity:
        raise ValueError("image latent cache requires a dataset and Mage weight identity")
    values = config.image_latents
    identity = (
        mage_weight_identity
        if isinstance(mage_weight_identity, dict)
        else {
            "repository": values.repository,
            "revision": values.revision,
            "weight_file": values.weight_file,
            "weight_sha256": mage_weight_identity,
        }
    )
    return {
        "schema_version": 1,
        "kind": "image_latents",
        "dataset": {
            **preprocessing_metadata(config.data, values.image_size),
            "dataset_size": dataset_size,
        },
        "encoder": identity,
        "tensor": {
            "description": "native-mage-vae-posterior-mean",
            "shape": list(values.shape),
            "dtype": "bfloat16",
            "posterior": "mean",
            "preprocessing": "dataset_uint8_or_[0,1]_to_native_[-1,1]",
            "patch_packing": False,
            "normalization": None,
        },
    }


def image_latent_cache_fingerprint(spec: dict[str, Any]) -> str:
    return cache_fingerprint(spec)


def image_latent_cache_path(config: ExperimentConfig, spec: dict[str, Any]) -> Path:
    return cache_path(config.image_latents.cache.directory, spec)


def required_image_latent_cache_bytes(spec: dict[str, Any]) -> int:
    return required_cache_bytes(spec)


def preflight_image_latent_cache_capacity(
    directory: str | Path,
    spec: dict[str, Any],
    path: str | Path | None = None,
) -> int:
    return preflight_cache_capacity(directory, spec, path)


ImageLatentCacheWriter = RepresentationCacheWriter
AsyncImageLatentCacheWriter = AsyncRepresentationCacheWriter


class ImageLatentCache(RepresentationCache):
    def __init__(
        self,
        config: ExperimentConfig,
        dataset_size: int,
        mage_weight_identity: str | dict[str, Any],
    ) -> None:
        spec = image_latent_cache_spec(config, dataset_size, mage_weight_identity)
        super().__init__(image_latent_cache_path(config, spec), spec)
