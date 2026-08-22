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
from .vision import semantic_encoder_fingerprint


def semantic_token_cache_spec(
    config: ExperimentConfig,
    dataset_size: int,
    semantic_statistics_sha256: str,
) -> dict[str, Any]:
    if dataset_size < 1 or not semantic_statistics_sha256:
        raise ValueError(
            "semantic token cache requires a dataset and semantic statistics"
        )
    values = config.semantic_tokens
    return {
        "schema_version": 1,
        "kind": "semantic_tokens",
        "dataset": {
            **preprocessing_metadata(config.data, config.image_latents.image_size),
            "dataset_size": dataset_size,
            "semantic_view": "resize-to-128px-bicubic",
        },
        "encoder": {
            "repository": values.repository,
            "revision": values.revision,
            "fingerprint": semantic_encoder_fingerprint(values),
        },
        "semantic_statistics_sha256": semantic_statistics_sha256,
        "tensor": {
            "description": "normalized-dinov3-final-unpooled-patch-tokens",
            "shape": list(values.shape),
            "dtype": "bfloat16",
        },
    }


def semantic_token_cache_fingerprint(spec: dict[str, Any]) -> str:
    return cache_fingerprint(spec)


def semantic_token_cache_path(config: ExperimentConfig, spec: dict[str, Any]) -> Path:
    return cache_path(config.semantic_tokens.cache.directory, spec)


def required_semantic_token_cache_bytes(spec: dict[str, Any]) -> int:
    return required_cache_bytes(spec)


def preflight_semantic_token_cache_capacity(
    directory: str | Path,
    spec: dict[str, Any],
    path: str | Path | None = None,
) -> int:
    return preflight_cache_capacity(directory, spec, path)


SemanticTokenCacheWriter = RepresentationCacheWriter
AsyncSemanticTokenCacheWriter = AsyncRepresentationCacheWriter


class SemanticTokenCache(RepresentationCache):
    def __init__(
        self,
        config: ExperimentConfig,
        dataset_size: int,
        semantic_statistics_sha256: str,
    ) -> None:
        spec = semantic_token_cache_spec(
            config, dataset_size, semantic_statistics_sha256
        )
        super().__init__(semantic_token_cache_path(config, spec), spec)
