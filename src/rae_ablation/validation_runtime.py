from __future__ import annotations

import time
from pathlib import Path

import torch

from .attnres_visualization import save_attnres_visualization
from .checkpoint import EMA
from .codecs import load_codecs
from .config import ExperimentConfig
from .distributed import broadcast_string
from .evaluation import evaluate_npz, save_evaluation
from .flow import shift_timestep
from .latents import LatentStatistics
from .model.compile import compile_model_regions
from .model.factory import build_diffusion_model
from .rng import preserving_global_rng
from .sample_runtime import SamplingParameters, sample_to_archive, state_dict_sha256
from .sampling import euler_sample
from .validation import ValidationPlan, noise_for_samples


def _distributed() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _capture_attnres_probe(
    *,
    model: torch.nn.Module,
    config: ExperimentConfig,
    plan: ValidationPlan,
    sampling: SamplingParameters,
    device: torch.device,
    output: Path,
) -> None:
    block_size = config.model.attnres_block_size
    if block_size is None:
        return
    capture_context = getattr(model, "capture_attnres_weights", None)
    if capture_context is None:
        raise TypeError("AttnRes model does not expose weight capture")
    channels, height, width = config.latent_shape
    planned_noise = (
        plan.visualization_noise_16 if (height, width) == (16, 16) else plan.visualization_noise_8
    )
    if planned_noise.shape != (16, channels, height, width):
        raise ValueError("validation visualization noise does not match latent shape")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    noise = planned_noise.to(device=device, dtype=dtype)
    labels = plan.visualization_labels.to(device)
    with (
        capture_context(height, width) as capture,
        torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
    ):
        euler_sample(
            model,
            noise,
            labels,
            steps=sampling.sampling_steps,
            guidance_scale=sampling.guidance_scale,
            timestep_shift=sampling.timestep_shift,
        )
    schedule = shift_timestep(
        torch.linspace(0, 1, sampling.sampling_steps + 1),
        sampling.timestep_shift,
    )[:-1]
    save_attnres_visualization(
        capture=capture,
        sampling_timesteps=schedule.numpy(),
        labels=plan.visualization_labels.numpy(),
        plan_hash=plan.plan_hash,
        block_size=block_size,
        output_dir=output,
    )


@torch.inference_mode()
@preserving_global_rng
def validate_ema(
    *,
    ema: EMA,
    config: ExperimentConfig,
    statistics: LatentStatistics,
    plan: ValidationPlan,
    completed_epoch: int,
    final: bool,
    output_root: str | Path,
    reference_npz: str | Path,
    sampling: SamplingParameters,
    eval_batch_size: int = 64,
    eval_log_interval: int = 10,
) -> tuple[dict[str, float], Path]:
    rank, _world_size = _distributed()
    device = torch.device(
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )
    count = config.validation.final_samples if final else config.validation.periodic_samples
    name = "final-50k" if final else "periodic-5k"
    output = Path(output_root) / f"epoch-{completed_epoch:04d}-{name}"
    output.mkdir(parents=True, exist_ok=True)

    model = build_diffusion_model(config.model).to(device)
    model.load_state_dict(ema.state_dict())
    model.eval()
    execution_model = compile_model_regions(model, config.compile)
    codec = load_codecs(config.codec, str(device))
    if isinstance(codec, tuple):
        codec = codec[0] if config.codec == "rae" else codec[1]

    @torch.inference_mode()
    def generate_batch(
        _positions: torch.Tensor, labels: torch.Tensor, noise_seeds: torch.Tensor
    ) -> torch.Tensor:
        batch_labels = labels.to(device)
        noise = noise_for_samples(
            noise_seeds,
            config.latent_shape,
            device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            normalized = euler_sample(
                execution_model,
                noise,
                batch_labels,
                steps=sampling.sampling_steps,
                guidance_scale=sampling.guidance_scale,
                timestep_shift=sampling.timestep_shift,
            )
            return codec.decode(statistics.denormalize(normalized))

    digest = state_dict_sha256(ema.state_dict()) if rank == 0 else None
    digest = broadcast_string(digest, rank)
    started = time.perf_counter()
    archive = sample_to_archive(
        output_dir=output,
        plan=plan,
        sample_count=count,
        num_classes=config.model.num_classes,
        codec=config.codec,
        checkpoint_hash=digest,
        parameters=sampling,
        generate_batch=generate_batch,
    )
    if rank == 0:
        _capture_attnres_probe(
            model=model,
            config=config,
            plan=plan,
            sampling=sampling,
            device=device,
            output=output,
        )
    del generate_batch
    execution_model = None
    model = None
    codec = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics: dict[str, float] = {}
    if rank == 0:
        raw = evaluate_npz(
            archive,
            reference_npz,
            batch_size=eval_batch_size,
            device=device,
            log_interval=eval_log_interval,
        )
        save_evaluation(output / "evaluation.json", raw, plan.plan_hash)
        suffix = "50K" if final else "5K"
        metrics[f"validation/FID-{suffix}"] = raw["FID"]
        metrics["validation/images_per_second"] = count / max(time.perf_counter() - started, 1e-9)
        metrics["validation/sampling_nfe"] = float(sampling.sampling_steps)
        metrics["validation/peak_memory_gib"] = (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
        )
    _barrier()
    return metrics, output
