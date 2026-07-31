from __future__ import annotations

import pytest
import torch

from self_flow.checkpoint import EMA
from self_flow.config import ExperimentConfig, ModelConfig, ObjectiveConfig, SIGRegConfig
from self_flow.model import SiT
from self_flow.objective import SIGReg, self_flow_loss, token_times_to_latents
from self_flow.rng import RNGStreams


def experiment(kind: str, *, clean_forward: str = "truncated") -> ExperimentConfig:
    model = ModelConfig(
        image_size=32,
        patch_size=1,
        hidden_size=32,
        depth=3,
        num_heads=4,
        num_classes=10,
        variant="SiT-tiny/1",
    )
    sigreg = (
        SIGRegConfig(
            clean_forward=clean_forward,
            projections=16,
            knots=5,
            max_frequency=2,
            projection_chunk_size=4,
        )
        if kind == "sigreg"
        else SIGRegConfig()
    )
    objective = ObjectiveConfig(
        kind=kind,
        align_layers=[1, 2] if kind in {"ema", "sigreg"} else [],
        projector_hidden_size=48,
        sigreg=sigreg,
    )
    config = ExperimentConfig(model=model, objective=objective)
    config.train.global_batch_size = 2
    config.validation.periodic_samples = 10
    config.validation.final_samples = 10
    config.sampling.num_samples = 10
    config.validate()
    return config


@pytest.mark.parametrize(
    "kind,clean_forward",
    [
        ("vanilla", "truncated"),
        ("dual_timestep", "truncated"),
        ("ema", "truncated"),
        ("sigreg", "truncated"),
        ("sigreg", "full"),
    ],
)
def test_all_experiment_graphs_are_finite(kind: str, clean_forward: str) -> None:
    config = experiment(kind, clean_forward=clean_forward)
    model = SiT(
        config.model,
        align_layers=config.objective.align_layers,
        projector_hidden_size=config.objective.projector_hidden_size,
    )
    torch.nn.init.normal_(model.final.projection.weight, std=0.01)
    ema = EMA(model) if kind == "ema" else None
    losses = self_flow_loss(
        model,
        torch.randn(2, 128, 2, 2),
        torch.tensor([1, 2]),
        RNGStreams.create(7),
        config,
        ema_model=ema.model if ema else None,
    )
    losses.total.backward()
    assert torch.isfinite(losses.total)
    assert model.patch_embed.weight.grad is not None
    if kind == "sigreg" and clean_forward == "full":
        expected = 0.5 * (losses.mixed_velocity + losses.cleaner_velocity)
        expected = expected + config.objective.alignment_weight * losses.alignment
        expected = expected + config.objective.sigreg.weight * losses.sigreg
        assert torch.allclose(losses.total, expected)
        assert losses.cleaner_velocity > 0
    else:
        assert losses.cleaner_velocity == 0


def test_sigreg_truncated_executes_shared_cleaner_only_to_alignment_depth() -> None:
    config = experiment("sigreg")
    model = SiT(config.model, align_layers=[1, 2], projector_hidden_size=48)
    calls = [0, 0, 0]
    handles = [
        block.register_forward_hook(
            lambda _m, _a, _o, index=index: calls.__setitem__(index, calls[index] + 1)
        )
        for index, block in enumerate(model.blocks)
    ]
    loss = self_flow_loss(
        model,
        torch.randn(2, 128, 2, 2),
        torch.tensor([1, 2]),
        RNGStreams.create(4),
        config,
        collect_diagnostics=True,
    )
    loss.total.backward()
    for handle in handles:
        handle.remove()
    assert calls == [2, 2, 1]
    assert len(loss.feature_mean_norms) == 4
    assert len(loss.feature_std_means) == 4
    assert len(loss.feature_covariance_offdiag_rms) == 4


def test_sigreg_is_deterministic_and_requires_local_batch_two() -> None:
    config = SIGRegConfig(projections=8, knots=5, projection_chunk_size=3)
    regularizer = SIGReg(config)
    features = (torch.randn(3, 4, 8), torch.randn(3, 4, 8))
    left = regularizer(features, torch.Generator().manual_seed(9))[0]
    right = regularizer(features, torch.Generator().manual_seed(9))[0]
    assert torch.equal(left, right)
    directions = torch.randn(8, 8, generator=torch.Generator().manual_seed(9))
    directions = torch.nn.functional.normalize(directions, dim=0)
    knots = torch.linspace(0, 3, 5)
    spacing = 3 / 4
    weights = torch.full((5,), 2 * spacing)
    weights[[0, -1]] = spacing
    target = torch.exp(-0.5 * knots.square())
    weights *= target
    reference_groups = []
    for feature in features:
        projected = torch.einsum("bnd,dm->nbm", feature, directions)
        values = projected[..., None] * knots
        empirical_real = values.cos().mean(dim=1)
        empirical_imag = values.sin().mean(dim=1)
        reference_groups.append(
            (((empirical_real - target).square() + empirical_imag.square()) * weights)
            .sum(dim=-1)
            .mul(feature.shape[0])
            .mean()
        )
    assert torch.allclose(left, torch.stack(reference_groups).mean())
    with pytest.raises(ValueError, match="local batch"):
        regularizer((torch.randn(1, 4, 8),), torch.Generator())


def test_token_times_expand_to_corresponding_latent_patches() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expanded = token_times_to_latents(values, grid_size=2, patch_size=2)
    assert expanded.shape == (1, 1, 4, 4)
    assert torch.equal(expanded[0, 0, :2, :2], torch.ones(2, 2))
    assert torch.equal(expanded[0, 0, 2:, 2:], torch.full((2, 2), 4.0))
