from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from .checkpoint import EMA
from .config import ExperimentConfig
from .distributed import broadcast_string
from .evaluation import evaluate_npz, save_evaluation
from .model.compile import compile_model_regions
from .model.ddt import DDT
from .rng import preserving_global_rng
from .sample_runtime import (
    SamplingParameters,
    feature_schema_for_config,
    sample_to_archive,
    sampling_output_name,
    state_dict_sha256,
)
from .sampling import ODESample, ode_sample
from .vae import load_flux2_codec
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
    sampling: SamplingParameters | Sequence[SamplingParameters],
) -> tuple[dict[str, float], Path]:
    rank, _world_size = _distributed()
    device = torch.device(
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )
    candidates = (
        [sampling] if isinstance(sampling, SamplingParameters) else list(sampling)
    )
    if not candidates:
        raise ValueError("validation requires at least one sampling candidate")
    if not config.model.features.reg.enabled:
        candidates = candidates[:1]
    for candidate in candidates:
        candidate.validate()

    tune_count = config.validation.periodic_samples
    run_kind = "final" if final else "periodic"
    root = Path(output_root) / f"epoch-{completed_epoch:04d}-{run_kind}"
    root.mkdir(parents=True, exist_ok=True)
    reg = config.model.features.reg
    digest = state_dict_sha256(ema.state_dict()) if rank == 0 else None
    digest = broadcast_string(digest, rank)
    feature_schema = feature_schema_for_config(config)
    codec = (
        load_flux2_codec(
            config.vae,
            image_size=config.model.image_size,
            device=device,
        )
        if config.model.input_space == "flux2_latents"
        else None
    )
    started = time.perf_counter()

    def build_model() -> tuple[DDT, torch.nn.Module]:
        model = DDT(config.model).to(device)
        model.load_state_dict(ema.state_dict())
        model.eval()
        return model, compile_model_regions(model, config.compile)

    def generate_archive(
        *,
        execution_model: torch.nn.Module,
        parameters: SamplingParameters,
        count: int,
        parent: Path,
    ) -> tuple[Path, Path]:
        output = parent / sampling_output_name(count, parameters)

        @torch.inference_mode()
        def generate_batch(
            _positions: torch.Tensor, labels: torch.Tensor, noise_seeds: torch.Tensor
        ) -> torch.Tensor:
            batch_labels = labels.to(device)
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            noise = noise_for_samples(
                noise_seeds,
                config.model.input_shape,
                device,
                dtype=dtype,
            )
            reg_noise = (
                noise_for_samples(
                    noise_seeds,
                    (reg.num_tokens, config.model.features.encoder.feature_dim),
                    device,
                    dtype=dtype,
                    seed_offset=1_000_003,
                )
                if reg.enabled
                else None
            )
            with torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                generated = ode_sample(
                    execution_model,
                    noise,
                    batch_labels,
                    reg_noise=reg_noise,
                    steps=parameters.sampling_steps,
                    guidance_scale=parameters.guidance_scale,
                    reg_guidance_scale=parameters.reg_guidance_scale,
                    timestep_shift=parameters.timestep_shift,
                    reg_timestep_shift=parameters.reg_timestep_shift,
                    img_denoising_interval=parameters.img_denoising_interval,
                    reg_denoising_interval=parameters.reg_denoising_interval,
                    cfg_interval=(parameters.cfg_start, parameters.cfg_end),
                    solver=parameters.solver,
                    num_classes=config.model.num_classes,
                )
            image = generated.image if isinstance(generated, ODESample) else generated
            if codec is not None:
                return codec.decode(image)
            return image.float().add(1).div(2)

        archive = sample_to_archive(
            output_dir=output,
            plan=plan,
            sample_count=count,
            num_classes=config.model.num_classes,
            image_size=config.model.image_size,
            patch_size=config.model.patch_size,
            grid_size=config.model.grid_size,
            model_variant=config.model.model_variant,
            feature_schema=feature_schema,
            checkpoint_hash=digest,
            parameters=parameters,
            generate_batch=generate_batch,
        )
        return archive, output

    def release_model(model: DDT, execution_model: torch.nn.Module) -> None:
        model.cpu()
        del execution_model, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    sample_count = (
        tune_count
        if reg.enabled
        else config.validation.final_samples
        if final
        else tune_count
    )
    model, execution_model = build_model()
    tune_results = [
        (
            candidate,
            *generate_archive(
                execution_model=execution_model,
                parameters=candidate,
                count=sample_count,
                parent=root,
            ),
        )
        for candidate in candidates
    ]
    release_model(model, execution_model)
    execution_model = None
    model = None
    if codec is not None:
        codec.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics: dict[str, float] = {}
    fid_values: list[float] = []
    if rank == 0:
        eval_device = (
            device
            if config.evaluation.device == "auto"
            else torch.device(config.evaluation.device)
        )
        for candidate, archive, output in tune_results:
            raw = evaluate_npz(
                archive,
                config.evaluation.reference_npz,
                batch_size=config.evaluation.batch_size,
                device=eval_device,
                log_interval=config.evaluation.log_interval,
            )
            save_evaluation(output / "evaluation.json", raw, plan.plan_hash)
            fid = float(raw["FID"])
            fid_values.append(fid)
            if reg.enabled:
                shift_slug = repr(candidate.reg_timestep_shift).replace(".", "p")
                metrics[f"validation/reg_shift_sweep/FID-{shift_slug}"] = fid
    winner_state: list[int | None] = [
        min(range(len(fid_values)), key=lambda index: (fid_values[index], index))
        if rank == 0
        else None
    ]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(winner_state, src=0)
    winner_index = winner_state[0]
    if winner_index is None:
        raise RuntimeError("validation failed to select a REG timestep shift")
    winner, _winner_archive, winner_output = tune_results[winner_index]

    if rank == 0:
        suffix = (
            f"{sample_count // 1000}K"
            if sample_count % 1000 == 0
            else str(sample_count)
        )
        metrics[f"validation/FID-{suffix}"] = fid_values[winner_index]
        if reg.enabled:
            metrics["validation/best_reg_timestep_shift"] = (
                winner.reg_timestep_shift
            )
            summary = {
                "sample_count": sample_count,
                "candidates": [
                    {
                        "reg_timestep_shift": candidate.reg_timestep_shift,
                        "FID": fid,
                    }
                    for (candidate, _archive, _output), fid in zip(
                        tune_results, fid_values, strict=True
                    )
                ],
                "best_reg_timestep_shift": winner.reg_timestep_shift,
                "best_fid": fid_values[winner_index],
            }
            (root / "sweep.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )

    total_generated = sample_count * len(candidates)
    if final and reg.enabled:
        if codec is not None:
            codec.to(device)
        model, execution_model = build_model()
        final_archive, final_output = generate_archive(
            execution_model=execution_model,
            parameters=winner,
            count=config.validation.final_samples,
            parent=root / "selected-final",
        )
        release_model(model, execution_model)
        execution_model = None
        model = None
        if codec is not None:
            codec.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if rank == 0:
            raw = evaluate_npz(
                final_archive,
                config.evaluation.reference_npz,
                batch_size=config.evaluation.batch_size,
                device=eval_device,
                log_interval=config.evaluation.log_interval,
            )
            save_evaluation(final_output / "evaluation.json", raw, plan.plan_hash)
            final_count = config.validation.final_samples
            suffix = (
                f"{final_count // 1000}K"
                if final_count % 1000 == 0
                else str(final_count)
            )
            metrics[f"validation/FID-{suffix}"] = float(raw["FID"])
        total_generated += config.validation.final_samples
        winner_output = final_output

    if rank == 0:
        metrics["validation/images_per_second"] = total_generated / max(
            time.perf_counter() - started, 1e-9
        )
        metrics["validation/sampling_nfe"] = float(winner.sampling_nfe)
        metrics["validation/peak_memory_gib"] = (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
        )
    _barrier()
    return metrics, winner_output
