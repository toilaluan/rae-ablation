from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from jit_reg.checkpoint import EMA, CheckpointManager
from jit_reg.config import ExperimentConfig
from jit_reg.latent_cache import (
    AsyncLatentCacheWriter,
    LatentCache,
    LatentCacheWriter,
    latent_cache_fingerprint,
    latent_cache_path,
    latent_cache_spec,
    required_latent_cache_bytes,
)
from jit_reg.model.ddt import DDT
from jit_reg.optim import build_optimizer
from jit_reg.rng import RNGStreams
from jit_reg.training import make_train_loader, train
from jit_reg.vae import Flux2Codec, patchify_latents, unpatchify_latents
from jit_reg.validation import create_validation_plan


class _Posterior:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def mode(self) -> torch.Tensor:
        return self.value


class FakeFlux2VAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            latent_channels=32,
            block_out_channels=(8, 16, 32, 32),
            patch_size=(2, 2),
            batch_norm_eps=1e-4,
        )
        self.bn = nn.BatchNorm2d(128, affine=False)
        self.bn.running_mean.copy_(torch.linspace(-0.25, 0.25, 128))
        self.bn.running_var.copy_(torch.linspace(0.5, 1.5, 128))
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, images: torch.Tensor) -> SimpleNamespace:
        self.encode_calls += 1
        pooled = F.avg_pool2d(images.mean(dim=1, keepdim=True), kernel_size=8)
        return SimpleNamespace(latent_dist=_Posterior(pooled.repeat(1, 32, 1, 1)))

    def decode(
        self, latents: torch.Tensor, return_dict: bool = False
    ) -> tuple[torch.Tensor]:
        assert not return_dict
        self.decode_calls += 1
        image = F.interpolate(
            latents[:, :3],
            scale_factor=8,
            mode="nearest",
        ).clamp(-1, 1)
        return (image,)


def latent_config(tmp_path: Path, *, image_size: int = 32) -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.input_space = "flux2_latents"
    config.model.image_size = image_size
    config.model.in_channels = 128
    config.model.patch_size = 1
    config.vae.cache.directory = str(tmp_path)
    config.validate()
    return config


def test_flux2_patchification_is_exactly_invertible() -> None:
    raw = torch.arange(2 * 32 * 4 * 6).reshape(2, 32, 4, 6)
    packed = patchify_latents(raw)
    assert packed.shape == (2, 128, 2, 3)
    assert torch.equal(unpatchify_latents(packed), raw)


def test_flux2_codec_uses_mode_bn_and_micro_batches() -> None:
    vae = FakeFlux2VAE()
    codec = Flux2Codec(vae, image_size=32, micro_batch_size=2)
    images = torch.rand(3, 3, 32, 32)
    latents = codec.encode(images)
    assert latents.shape == (3, 128, 2, 2)
    assert latents.dtype == torch.float32
    assert vae.encode_calls == 2
    decoded = codec.decode(latents)
    assert decoded.shape == images.shape
    assert decoded.min() >= 0 and decoded.max() <= 1
    assert vae.decode_calls == 2
    assert not codec.training and not vae.training
    assert not any(parameter.requires_grad for parameter in codec.parameters())


def test_ddt_supports_two_by_two_flux2_latent_patches(tmp_path: Path) -> None:
    config = latent_config(tmp_path)
    config.model.patch_size = 2
    config.model.hidden_size = 16
    config.model.depth = 1
    config.model.num_heads = 4
    config.model.mlp_ratio = 2
    config.model.decoder.hidden_size = 32
    config.model.decoder.depth = 1
    config.model.decoder.num_heads = 4
    config.model.decoder.mlp_ratio = 2
    config.model.num_classes = 2
    config.validate()
    model = DDT(config.model)
    noisy = torch.randn(2, *config.model.input_shape)
    output = model(
        noisy,
        torch.rand(2),
        torch.tensor([0, 1]),
    )
    assert model.grid_size == 1
    assert output.image.shape == noisy.shape


def test_latent_cache_fingerprint_and_exact_capacity(tmp_path: Path) -> None:
    config = latent_config(tmp_path)
    spec = latent_cache_spec(config, 3)
    assert required_latent_cache_bytes(spec) == 3 * 128 * 2 * 2 * 2 + 3
    original = latent_cache_fingerprint(spec)

    changed = copy.deepcopy(config)
    changed.vae.revision = "different"
    assert latent_cache_fingerprint(latent_cache_spec(changed, 3)) != original

    irrelevant = copy.deepcopy(config)
    irrelevant.train.optimizer.lr = 1
    irrelevant.sampling.sampling_steps = 3
    assert latent_cache_fingerprint(latent_cache_spec(irrelevant, 3)) == original


def test_latent_cache_pipeline_and_lookup(tmp_path: Path) -> None:
    config = latent_config(tmp_path)
    spec = latent_cache_spec(config, 3)
    path = latent_cache_path(config, 3)
    assert not LatentCacheWriter.initialize(path, spec)
    writer = LatentCacheWriter(path, spec)
    pipeline = AsyncLatentCacheWriter(writer, batch_size=2, device="cpu")
    for indices in (torch.tensor([2, 0]), torch.tensor([1])):
        values = (
            indices.float()
            .reshape(-1, 1, 1, 1)
            .expand(-1, *writer.shape[1:])
        )
        pipeline.submit(indices, values)
    pipeline.close()
    writer.finalize()
    cache = LatentCache(config, 3)
    actual = cache.lookup(torch.tensor([0, 2]))[:, 0, 0, 0].float()
    assert torch.equal(actual, torch.tensor([0.0, 2.0]))
    assert pipeline.metrics().examples == 3


class _SilentLogger:
    run_id = None

    def event(self, message: str) -> None:
        del message

    def log_training(self, metrics: dict[str, Any], step: int) -> None:
        del metrics, step

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        del metrics, step

    def log_checkpoint(self, path: str, name: str) -> None:
        del path, name


def test_cached_latents_train_without_rgb_pixels(tmp_path: Path) -> None:
    config = latent_config(tmp_path, image_size=16)
    config.vae.cache.enabled = True
    config.model.hidden_size = 16
    config.model.depth = 1
    config.model.num_heads = 4
    config.model.mlp_ratio = 2
    config.model.decoder.hidden_size = 32
    config.model.decoder.depth = 1
    config.model.decoder.num_heads = 4
    config.model.decoder.mlp_ratio = 2
    config.model.num_classes = 2
    config.compile.enabled = False
    config.train.optimizer.kind = "adamw"
    config.train.epochs = 1
    config.train.global_batch_size = 2
    config.train.max_steps = 1
    config.data.workers = 0
    config.validation.interval = 0
    config.validation.periodic_samples = 2
    config.validation.final_samples = 2
    config.checkpoint.directory = str(tmp_path / "checkpoints")
    config.validate()

    spec = latent_cache_spec(config, 2)
    path = latent_cache_path(config, 2)
    LatentCacheWriter.initialize(path, spec)
    writer = LatentCacheWriter(path, spec)
    writer.write(torch.arange(2), torch.randn(writer.shape))
    writer.finalize()
    cache = LatentCache(config, 2)

    dataset = [(None, 0, 0), (None, 1, 1)]
    streams = RNGStreams.create(3)
    loader, sampler = make_train_loader(dataset, config, streams)
    batch = next(iter(loader))
    assert batch.pixels is None
    assert batch.cached_latents is not None

    model = DDT(config.model)
    plan = create_validation_plan(2, 2, 2, seed=5)
    plan_path = tmp_path / "plan.pt"
    plan.save(plan_path)
    state = train(
        canonical_model=model,
        execution_model=model,
        loader=loader,
        sampler=sampler,
        optimizer=build_optimizer(model, config.train.optimizer),
        ema=EMA(model),
        streams=streams,
        config=config,
        checkpoint_manager=CheckpointManager(config.checkpoint.directory),
        data_fingerprint="data",
        validation_plan_hash=plan.plan_hash,
        validation_plan_path=plan_path,
        logger=_SilentLogger(),  # type: ignore[arg-type]
        latent_cache=cache,
    )
    assert state.global_step == 1
