from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jit_reg.config import ExperimentConfig
from jit_reg.reg import REGTargets
from jit_reg.reg_cache import (
    COMPLETION_FILE,
    MANIFEST_FILE,
    REPA_FILE,
    TOKEN_FILE,
    REGTokenCache,
    REGTokenCacheWriter,
    cache_fingerprint,
    cache_path,
    cache_spec,
)


def _config(
    tmp_path: Path,
    token_mode: str = "cls",
    *,
    repa_enabled: bool = False,
) -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.image_size = 32
    config.model.patch_size = 16
    config.model.reg.enabled = True
    config.model.reg.repa_enabled = repa_enabled
    config.model.reg.feature_dim = 8
    config.model.reg.token_mode = token_mode
    config.model.reg.pool_size = 2
    config.reg_cache.enabled = True
    config.reg_cache.directory = str(tmp_path)
    config.train.horizontal_flip_probability = 0
    config.validate()
    return config


def test_reg_cache_is_resumable_indexed_and_bfloat16(tmp_path: Path) -> None:
    config = _config(tmp_path, repa_enabled=True)
    spec = cache_spec(config, 4)
    directory = cache_path(config, 4)
    assert not REGTokenCacheWriter.initialize(directory, spec)
    writer = REGTokenCacheWriter(directory, spec)
    first_indices = torch.tensor([0, 2])
    first = torch.tensor(
        [
            [[1.25, 2, 3, 4, 5, 6, 7, 8]],
            [[9, 10, 11, 12, 13, 14, 15, 16]],
        ],
        dtype=torch.float32,
    )
    first_repa = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    writer.write(first_indices, REGTargets(first, first_repa))
    assert writer.pending_indices().tolist() == [1, 3]

    resumed = REGTokenCacheWriter(directory, spec)
    second_indices = torch.tensor([1, 3])
    second = torch.full((2, 1, 8), 17.0)
    second_repa = torch.full((2, 4, 8), 19.0)
    resumed.write(second_indices, REGTargets(second, second_repa))
    resumed.finalize()

    cache = REGTokenCache(config, 4)
    values = cache.lookup(torch.tensor([3, 0, 2]))
    assert values.reg_tokens.dtype == torch.bfloat16
    torch.testing.assert_close(values.reg_tokens[0], second[1].bfloat16())
    torch.testing.assert_close(values.reg_tokens[1], first[0].bfloat16())
    torch.testing.assert_close(values.reg_tokens[2], first[1].bfloat16())
    assert values.repa is not None and values.repa.dtype == torch.bfloat16
    torch.testing.assert_close(values.repa[0], second_repa[1].bfloat16())
    torch.testing.assert_close(values.repa[1], first_repa[0].bfloat16())
    torch.testing.assert_close(values.repa[2], first_repa[1].bfloat16())
    assert (directory / TOKEN_FILE).stat().st_size == 4 * 1 * 8 * 2
    assert (directory / REPA_FILE).stat().st_size == 4 * 4 * 8 * 2
    assert (directory / COMPLETION_FILE).stat().st_size == 4
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    assert manifest["complete"]
    assert manifest["shape"] == [4, 1, 8]
    assert manifest["repa_shape"] == [4, 4, 8]
    assert manifest["repa_file"] == REPA_FILE


def test_reg_cache_without_repa_returns_none(tmp_path: Path) -> None:
    config = _config(tmp_path)
    spec = cache_spec(config, 2)
    directory = cache_path(config, 2)
    REGTokenCacheWriter.initialize(directory, spec)
    writer = REGTokenCacheWriter(directory, spec)
    writer.write(torch.tensor([0, 1]), REGTargets(torch.zeros(2, 1, 8), None))
    writer.finalize()

    targets = REGTokenCache(config, 2).lookup(torch.tensor([1, 0]))
    assert targets.reg_tokens.shape == (2, 1, 8)
    assert targets.repa is None
    assert not (directory / REPA_FILE).exists()
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    assert manifest["repa_shape"] is None
    assert manifest["repa_file"] is None


def test_reg_cache_fingerprint_tracks_effective_target_configuration(tmp_path: Path) -> None:
    cls = _config(tmp_path, "cls")
    cls_spec = cache_spec(cls, 4)
    changed_data = _config(tmp_path, "cls")
    changed_data.data.revision = "different"
    spatial = _config(tmp_path, "cls_spatial")
    assert cache_fingerprint(cls_spec) != cache_fingerprint(cache_spec(changed_data, 4))
    assert cache_fingerprint(cls_spec) != cache_fingerprint(cache_spec(spatial, 4))
    first_spatial = cache_fingerprint(cache_spec(spatial, 4))
    spatial.model.reg.pool_size = 3
    assert first_spatial != cache_fingerprint(cache_spec(spatial, 4))
    repa = _config(tmp_path, "cls", repa_enabled=True)
    assert cache_fingerprint(cls_spec) != cache_fingerprint(cache_spec(repa, 4))
    first_repa = cache_fingerprint(cache_spec(repa, 4))
    repa.model.patch_size = 8
    assert first_repa != cache_fingerprint(cache_spec(repa, 4))
    cls.model.patch_size = 8
    assert cache_fingerprint(cls_spec) == cache_fingerprint(cache_spec(cls, 4))


@pytest.mark.parametrize("corrupt_file", [TOKEN_FILE, REPA_FILE])
def test_reg_cache_reader_rejects_incomplete_or_corrupt_files(
    tmp_path: Path,
    corrupt_file: str,
) -> None:
    config = _config(tmp_path, repa_enabled=True)
    spec = cache_spec(config, 2)
    directory = cache_path(config, 2)
    REGTokenCacheWriter.initialize(directory, spec)
    with pytest.raises(ValueError, match="incomplete"):
        REGTokenCache(config, 2)
    writer = REGTokenCacheWriter(directory, spec)
    writer.write(
        torch.tensor([0, 1]),
        REGTargets(torch.zeros(2, 1, 8), torch.zeros(2, 4, 8)),
    )
    writer.finalize()
    tokens = np.memmap(directory / corrupt_file, mode="r+", dtype=np.uint16)
    del tokens
    with (directory / corrupt_file).open("r+b") as handle:
        handle.truncate(2)
    with pytest.raises(ValueError, match="wrong size"):
        REGTokenCache(config, 2)


def test_reg_cache_writer_requires_matching_repa_presence_and_shape(tmp_path: Path) -> None:
    repa_config = _config(tmp_path, repa_enabled=True)
    repa_spec = cache_spec(repa_config, 2)
    repa_directory = cache_path(repa_config, 2)
    REGTokenCacheWriter.initialize(repa_directory, repa_spec)
    repa_writer = REGTokenCacheWriter(repa_directory, repa_spec)
    with pytest.raises(ValueError, match="REPA targets"):
        repa_writer.write(
            torch.tensor([0, 1]),
            REGTargets(torch.zeros(2, 1, 8), None),
        )
    with pytest.raises(ValueError, match="REPA targets"):
        repa_writer.write(
            torch.tensor([0, 1]),
            REGTargets(torch.zeros(2, 1, 8), torch.zeros(2, 1, 8)),
        )

    no_repa_config = _config(tmp_path)
    no_repa_spec = cache_spec(no_repa_config, 2)
    no_repa_directory = cache_path(no_repa_config, 2)
    REGTokenCacheWriter.initialize(no_repa_directory, no_repa_spec)
    no_repa_writer = REGTokenCacheWriter(no_repa_directory, no_repa_spec)
    with pytest.raises(ValueError, match="REPA-disabled"):
        no_repa_writer.write(
            torch.tensor([0, 1]),
            REGTargets(torch.zeros(2, 1, 8), torch.zeros(2, 4, 8)),
        )
