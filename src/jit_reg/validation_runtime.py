from __future__ import annotations

import time
from pathlib import Path

import torch

from .checkpoint import EMA
from .config import ExperimentConfig
from .distributed import broadcast_string
from .evaluation import evaluate_npz, save_evaluation
from .model.compile import compile_model_regions
from .model.sit import SiT
from .rng import preserving_global_rng
from .sample_runtime import SamplingParameters, sample_to_archive, state_dict_sha256
from .sampling import EulerSample, euler_sample
from .validation import ValidationPlan, noise_for_samples


def _distributed() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


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

    model = SiT(config.model).to(device)
    model.load_state_dict(ema.state_dict())
    model.eval()
    execution_model = compile_model_regions(model, config.compile)

    @torch.inference_mode()
    def generate_batch(
        _positions: torch.Tensor, labels: torch.Tensor, noise_seeds: torch.Tensor
    ) -> torch.Tensor:
        batch_labels = labels.to(device)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        noise = noise_for_samples(
            noise_seeds,
            (3, config.model.image_size, config.model.image_size),
            device,
            dtype=dtype,
        )
        reg_noise = (
            noise_for_samples(
                noise_seeds,
                (config.model.reg.num_tokens, config.model.reg.feature_dim),
                device,
                dtype=dtype,
                seed_offset=1_000_003,
            )
            if config.model.reg.enabled
            else None
        )
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            generated = euler_sample(
                execution_model,
                noise,
                batch_labels,
                reg_noise=reg_noise,
                steps=sampling.sampling_steps,
                guidance_scale=sampling.guidance_scale,
                timestep_shift=sampling.timestep_shift,
                reg_timestep_shift=sampling.reg_timestep_shift,
                cfg_interval=(sampling.cfg_start, sampling.cfg_end),
                num_classes=config.model.num_classes,
            )
        image = generated.image if isinstance(generated, EulerSample) else generated
        return image.float().add(1).div(2)

    digest = state_dict_sha256(ema.state_dict()) if rank == 0 else None
    digest = broadcast_string(digest, rank)
    started = time.perf_counter()
    archive = sample_to_archive(
        output_dir=output,
        plan=plan,
        sample_count=count,
        num_classes=config.model.num_classes,
        image_size=config.model.image_size,
        patch_size=config.model.patch_size,
        model_variant=config.model.model_variant,
        reg_enabled=config.model.reg.enabled,
        reg_variant=config.model.reg.cls_predictor if config.model.reg.enabled else "disabled",
        reg_token_mode=config.model.reg.token_mode if config.model.reg.enabled else "disabled",
        reg_pool_size=config.model.reg.pool_size,
        checkpoint_hash=digest,
        parameters=sampling,
        generate_batch=generate_batch,
    )
    del generate_batch
    execution_model = None
    model = None
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
