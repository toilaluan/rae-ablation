from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from .checkpoint import EMA
from .config import ExperimentConfig
from .data import load_imagenet, patchify_rgb, prepare_pixels, unpatchify_rgb
from .decoder_training import _save_grid
from .evaluation import (
    FeatureMoments,
    _default_feature_extractor,
    inception_features,
    reconstruction_fid,
    save_evaluation,
)
from .latent_statistics import EndpointLatentNormalizer
from .model import SemanticDiT
from .pixel_decoder import load_pixel_decoder
from .rng import indexed_standard_normal, preserving_global_rng
from .sampling import flow_sample, one_step_sample
from .validation import ValidationPlan
from .vision import load_dino_encoder


def _distributed() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _ssim_sum(target: Tensor, output: Tensor) -> Tensor:
    left = target.float().flatten(1)
    right = output.float().flatten(1)
    mu_x, mu_y = left.mean(1), right.mean(1)
    var_x, var_y = left.var(1, unbiased=False), right.var(1, unbiased=False)
    covariance = ((left - mu_x[:, None]) * (right - mu_y[:, None])).mean(1)
    return (
        ((2 * mu_x * mu_y + 0.01**2) * (2 * covariance + 0.03**2))
        / ((mu_x.square() + mu_y.square() + 0.01**2) * (var_x + var_y + 0.03**2))
    ).sum()


def _accumulate(
    total: Tensor,
    offset: int,
    target: Tensor,
    output: Tensor,
    endpoint_prediction: Tensor,
    endpoint_target: Tensor,
    perceptual: torch.nn.Module | None,
) -> None:
    difference = output.float() - target.float()
    batch = target.shape[0]
    total[offset] += difference.abs().flatten(1).mean(1).sum()
    total[offset + 1] += difference.square().flatten(1).mean(1).sum()
    total[offset + 2] += _ssim_sum(target, output)
    if perceptual is not None:
        total[offset + 3] += perceptual(
            2 * output.float() - 1, 2 * target.float() - 1
        ).sum()
    total[offset + 4] += (
        endpoint_prediction.float().sub(endpoint_target.float()).square().flatten(1).mean(1).sum()
    )
    total[offset + 5] += batch


@torch.inference_mode()
@preserving_global_rng
def validate_ema(
    *,
    ema: EMA,
    config: ExperimentConfig,
    plan: ValidationPlan,
    completed_epoch: int,
    final: bool,
    output_root: str | Path,
    normalizer: EndpointLatentNormalizer,
) -> tuple[dict[str, float], Path]:
    rank, world_size = _distributed()
    device = torch.device(
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )
    sample_count = config.validation.final_samples if final else config.validation.periodic_samples
    selected = plan.prefix(sample_count)[rank::world_size].tolist()
    dataset = load_imagenet(
        config.data,
        config.vision.image_size,
        load_pixels=True,
        split=config.data.validation_split,
    )
    expected_dataset_size = min(
        config.data.validation_size,
        config.data.max_samples or config.data.validation_size,
    )
    if len(dataset) != expected_dataset_size:
        raise ValueError(
            f"validation dataset has {len(dataset)} samples, expected "
            f"{expected_dataset_size} from data.validation_size/max_samples"
        )
    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=config.sampling.batch_size,
        shuffle=False,
        num_workers=config.data.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.workers > 0,
        prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,
    )
    model = SemanticDiT(config.model).to(device)
    ema.copy_to(model)
    model.eval()
    encoder = load_dino_encoder(
        config.vision,
        device,
        include_semantic=True,
        normalizer=normalizer,
    )
    decoder = None
    if config.flow.endpoint == "target":
        decoder, _digest = load_pixel_decoder(config.decoder, device)
    perceptual = None
    if config.decoder.loss.lpips_weight > 0:
        import lpips

        perceptual = lpips.LPIPS(net="alex").to(device).eval()
    inception = (
        _default_feature_extractor(device).to(device).eval()
        if config.evaluation.reconstruction_fid
        else None
    )
    feature_moments = (
        tuple(FeatureMoments(2048, device) for _ in range(3))
        if inception is not None
        else None
    )
    total = torch.zeros(12, dtype=torch.float64, device=device)
    preview: tuple[Tensor, Tensor] | None = None

    def decode(endpoint: Tensor) -> Tensor:
        if config.flow.endpoint == "target":
            assert decoder is not None
            return decoder(endpoint.to(next(decoder.parameters()).dtype)).float().clamp(0, 1)
        return unpatchify_rgb(normalizer.denormalize_pixel(endpoint)).float().clamp(0, 1)

    for pixels, _labels, indices in loader:
        raw_pixels = pixels.to(device, non_blocking=True)
        target_pixels = prepare_pixels(raw_pixels, device)
        endpoints = encoder.encode_pair(raw_pixels)
        endpoint_target = (
            endpoints.target
            if config.flow.endpoint == "target"
            else normalizer.normalize_pixel(patchify_rgb(target_pixels))
        )
        noise = indexed_standard_normal(
            indices,
            tuple(endpoints.semantic.shape[1:]),
            config.sampling.noise_seed,
            device,
            dtype=endpoints.semantic.dtype,
        )
        multi = flow_sample(
            model,
            endpoints.semantic,
            noise,
            steps=config.sampling.steps,
            timestep_shift=config.sampling.timestep_shift,
            solver=config.sampling.solver,  # type: ignore[arg-type]
        )
        single = one_step_sample(model, endpoints.semantic, noise)
        multi_pixels = decode(multi.endpoint)
        single_pixels = decode(single.endpoint)
        if inception is not None and feature_moments is not None:
            real_moments, multi_moments, single_moments = feature_moments
            real_moments.update(inception_features(inception, target_pixels))
            multi_moments.update(inception_features(inception, multi_pixels))
            single_moments.update(inception_features(inception, single_pixels))
        _accumulate(
            total,
            0,
            target_pixels,
            multi_pixels,
            multi.endpoint,
            endpoint_target,
            perceptual,
        )
        _accumulate(
            total,
            6,
            target_pixels,
            single_pixels,
            single.endpoint,
            endpoint_target,
            perceptual,
        )
        if rank == 0 and preview is None:
            preview = target_pixels, multi_pixels
    if world_size > 1:
        torch.distributed.all_reduce(total)
    if feature_moments is not None:
        for moments in feature_moments:
            moments.all_reduce()
    metrics: dict[str, float] = {}
    multi_name = f"{config.sampling.steps}step"
    for name, offset in ((multi_name, 0), ("1step", 6)):
        count = max(total[offset + 5].item(), 1)
        mse = total[offset + 1].item() / count
        metrics.update(
            {
                f"validation/l1_{name}": total[offset].item() / count,
                f"validation/mse_{name}": mse,
                f"validation/psnr_{name}": -10 * torch.log10(
                    torch.tensor(max(mse, 1e-12))
                ).item(),
                f"validation/ssim_{name}": total[offset + 2].item() / count,
                f"validation/lpips_{name}": total[offset + 3].item() / count,
                f"validation/endpoint_mse_{name}": total[offset + 4].item() / count,
            }
        )
    if feature_moments is not None:
        real_moments, multi_moments, single_moments = feature_moments
        metrics[f"validation/rfid_{multi_name}"] = reconstruction_fid(
            real_moments, multi_moments
        )
        metrics["validation/rfid_1step"] = reconstruction_fid(
            real_moments, single_moments
        )
    kind = "final" if final else "periodic"
    output = Path(output_root) / f"epoch-{completed_epoch:04d}-{kind}"
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if preview is not None:
            _save_grid(*preview, output / "preview.png")
        save_evaluation(
            output / "metrics.json",
            metrics,
            plan.plan_hash,
            {
                "endpoint": config.flow.endpoint,
                "sample_count": sample_count,
                "steps": config.sampling.steps,
                "one_step": True,
                "solver": config.sampling.solver,
                "timestep_shift": config.sampling.timestep_shift,
                "prior": "standard_normal",
                "noise_seed": config.sampling.noise_seed,
            },
        )
    return metrics, output
