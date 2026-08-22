from __future__ import annotations

from cascade_dit.config import ExperimentConfig
from cascade_dit.semantic_token_cache import (
    semantic_token_cache_fingerprint,
    semantic_token_cache_spec,
)


def test_semantic_cache_fingerprint_ignores_ddt_model() -> None:
    config = ExperimentConfig()
    first = semantic_token_cache_spec(config, 2, "statistics")
    config.model.architecture = "decoupled_cascade"
    config.model.hidden_size = [32, 48]
    config.model.depth = [2, 1]
    config.model.num_heads = [4, 3]
    second = semantic_token_cache_spec(config, 2, "statistics")
    assert semantic_token_cache_fingerprint(first) == semantic_token_cache_fingerprint(
        second
    )
