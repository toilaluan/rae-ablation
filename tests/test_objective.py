from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cascade_dit.config import ExperimentConfig
from cascade_dit.objective import (
    cascade_losses,
    gaussian_flow_path,
    sample_timesteps,
    select_semantic_condition,
    shift_timestep,
    x0_to_velocity,
)
from cascade_dit.rng import RNGStreams
from cascade_dit.training import compute_training_losses


def test_gaussian_paths_and_x0_velocity() -> None:
    clean = torch.tensor([[[[2.0]]]])
    noise = torch.tensor([[[[-1.0]]]])
    path = gaussian_flow_path(clean, noise, torch.tensor([0.25]))
    assert path.noisy.item() == pytest.approx(1.25)
    assert path.velocity.item() == pytest.approx(-3.0)
    assert x0_to_velocity(path.noisy, clean, path.timestep).item() == pytest.approx(-3.0)
    assert shift_timestep(torch.tensor(0.5), 4).item() == pytest.approx(0.8)
    values = sample_timesteps(32, torch.Generator().manual_seed(4), "cpu")
    assert torch.all((values > 0) & (values < 1))


def test_semantic_reuse_is_detached_and_seeded_fraction_is_quarter() -> None:
    clean = torch.zeros(20_000, 1, 1, 1, requires_grad=True)
    predicted = torch.ones_like(clean, requires_grad=True)
    condition, mask = select_semantic_condition(
        clean, predicted, torch.Generator().manual_seed(9), probability=0.25
    )
    assert mask.float().mean().item() == pytest.approx(0.25, abs=0.01)
    assert condition.requires_grad is False


def test_cascade_loss_is_one_sum_of_three_routes() -> None:
    semantic = gaussian_flow_path(
        torch.zeros(2, 1, 1, 1), torch.ones(2, 1, 1, 1), torch.ones(2)
    )
    image = gaussian_flow_path(
        torch.zeros(2, 1, 1, 1), torch.ones(2, 1, 1, 1), torch.ones(2)
    )
    losses = cascade_losses(
        semantic_path=semantic,
        image_latent_path=image,
        semantic_full_prediction=torch.zeros_like(semantic.noisy),
        semantic_base_prediction=torch.zeros_like(semantic.noisy),
        image_latent_prediction=torch.zeros_like(image.noisy),
        predicted_semantic_mask=torch.tensor([True, False]),
        semantic_base_loss_weight=0.5,
    )
    assert losses.total == (
        losses.image_latent_velocity
        + losses.semantic_full_velocity
        + 0.5 * losses.semantic_base_velocity
    )
    assert losses.predicted_semantic_condition_fraction == 0.5


def test_training_uses_route_specific_timestep_distributions() -> None:
    class RecordingCascade(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.semantic_timestep: torch.Tensor | None = None
            self.image_timestep: torch.Tensor | None = None

        def forward(
            self,
            noisy_semantic: torch.Tensor,
            semantic_timestep: torch.Tensor,
            semantic_labels: torch.Tensor,
            noisy_image: torch.Tensor,
            image_timestep: torch.Tensor,
            image_labels: torch.Tensor,
            clean_semantic: torch.Tensor,
            predicted_mask: torch.Tensor,
        ) -> SimpleNamespace:
            del semantic_labels, image_labels, clean_semantic, predicted_mask
            self.semantic_timestep = semantic_timestep.detach()
            self.image_timestep = image_timestep.detach()
            return SimpleNamespace(
                semantic=SimpleNamespace(
                    full=torch.zeros_like(noisy_semantic),
                    base=torch.zeros_like(noisy_semantic),
                ),
                image_latent=torch.zeros_like(noisy_image),
            )

    config = ExperimentConfig()
    config.model.class_dropout_probability = 0
    config.train.semantic_timestep_loc = -8
    config.train.semantic_timestep_scale = 0.01
    config.train.semantic_timestep_shift = 1
    config.train.image_timestep_loc = 8
    config.train.image_timestep_scale = 0.01
    config.train.image_timestep_shift = 1
    model = RecordingCascade()
    compute_training_losses(
        model,
        torch.zeros(4, 1, 1, 1),
        torch.zeros(4, 1, 1, 1),
        torch.arange(4),
        RNGStreams.create(3),
        config,
    )
    assert model.semantic_timestep is not None
    assert model.image_timestep is not None
    assert torch.all(model.semantic_timestep < 0.01)
    assert torch.all(model.image_timestep > 0.99)
