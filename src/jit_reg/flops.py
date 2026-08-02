from __future__ import annotations

from .config import ExperimentConfig


def _sit_forward_flops(
    config: ExperimentConfig,
    *,
    blocks: int,
    repa: bool = False,
    early_prediction: bool = False,
) -> int:
    """Static matmul/attention FLOPs for one sample; elementwise ops are excluded."""
    model = config.model
    tokens = model.grid_size**2
    hidden = model.hidden_size
    head_dim = hidden // model.num_heads
    intermediate = max(1, int(2 * hidden * model.mlp_ratio / 3))
    patch_values = model.patch_size**2 * model.in_channels
    patch = 2 * tokens * patch_values * hidden
    attention = (
        2 * tokens * hidden * (3 * hidden)
        + 4 * model.num_heads * tokens * tokens * head_dim
        + 2 * tokens * hidden * hidden
    )
    mlp = 2 * tokens * hidden * (2 * intermediate) + 2 * tokens * intermediate * hidden
    modulation = 2 * hidden * (6 * hidden)
    total = patch + blocks * (attention + mlp + modulation)
    if blocks == model.depth:
        total += 2 * hidden * (2 * hidden) + 2 * tokens * hidden * patch_values
    if repa:
        projection = config.repa.projection_hidden_size
        feature = config.repa.encoder.feature_dim
        total += 2 * tokens * (hidden * projection + projection**2 + projection * feature)
    if early_prediction:
        early_hidden = config.early_prediction.mlp_hidden_size or hidden
        total += 2 * tokens * (hidden * early_hidden + early_hidden * patch_values)
    return total


def model_flops_per_step(config: ExperimentConfig) -> int:
    """Approximate global denoiser FLOPs including selector and winner backward."""
    batch = config.train.global_batch_size
    training_forward = _sit_forward_flops(
        config,
        blocks=config.model.depth,
        repa=config.repa.enabled,
        early_prediction=config.early_prediction.enabled,
    )
    # A dense transformer backward is approximately two forward passes.
    total = batch * 3 * training_forward
    explore = config.exploration
    if explore.k_early > 1:
        if explore.early_selector == "early_prediction":
            blocks = config.early_prediction.block
            repa = False
            early_prediction = True
        else:
            blocks = config.repa.block
            repa = True
            early_prediction = False
        total += batch * explore.k_total * _sit_forward_flops(
            config,
            blocks=blocks,
            repa=repa,
            early_prediction=early_prediction,
        )
    if explore.k_full > 1:
        total += batch * explore.k_full * _sit_forward_flops(
            config,
            blocks=config.model.depth,
        )
    return total
