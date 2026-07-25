from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from jit_reg.checkpoint_bundle import load_checkpoint_bundle
from jit_reg.distributed import broadcast_string, initialize_distributed
from jit_reg.model.compile import compile_model_regions
from jit_reg.model.sit import SiT
from jit_reg.sample_runtime import (
    SamplingParameters,
    checkpoint_sha256,
    sample_to_archive,
    sampling_output_name,
)
from jit_reg.sampling import ODESample, ode_sample
from jit_reg.validation import noise_for_samples

torch._dynamo.config.cache_size_limit = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a self-contained JiT checkpoint into an ADM-compatible NPZ"
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--solver", choices=("euler", "heun"), default="euler")
    parser.add_argument("--guidance-scale", type=float, default=1.78)
    parser.add_argument("--timestep-shift", type=float, default=1.0)
    parser.add_argument("--cfg-start", type=float, default=0.0)
    parser.add_argument("--cfg-end", type=float, default=0.9)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = load_checkpoint_bundle(args.checkpoint_dir)
    config = bundle.config
    parameters = SamplingParameters(
        batch_size=args.batch_size,
        sampling_steps=args.sampling_steps,
        solver=args.solver,
        guidance_scale=args.guidance_scale,
        timestep_shift=args.timestep_shift,
        cfg_start=args.cfg_start,
        cfg_end=args.cfg_end,
        log_interval=args.log_interval,
    )
    parameters.validate()
    rank, _world_size, device = initialize_distributed()
    if rank == 0:
        print(f"initializing sampler on {device}")
    plan = bundle.plan
    digest = checkpoint_sha256(bundle.weights) if rank == 0 else None
    digest = broadcast_string(digest, rank)

    model = SiT(config.model).to(device)
    model.load_state_dict(load_file(str(bundle.weights), device=str(device)))
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
            generated = ode_sample(
                execution_model,
                noise,
                batch_labels,
                reg_noise=reg_noise,
                steps=parameters.sampling_steps,
                guidance_scale=parameters.guidance_scale,
                timestep_shift=parameters.timestep_shift,
                cfg_interval=(parameters.cfg_start, parameters.cfg_end),
                solver=parameters.solver,
                num_classes=config.model.num_classes,
            )
        image = generated.image if isinstance(generated, ODESample) else generated
        return image.float().add(1).div(2)

    wide_extensions_active = config.model.reg.wide_extensions_active
    sample_to_archive(
        output_dir=bundle.directory / sampling_output_name(args.num_samples, parameters),
        plan=plan,
        sample_count=args.num_samples,
        num_classes=config.model.num_classes,
        image_size=config.model.image_size,
        patch_size=config.model.patch_size,
        model_variant=config.model.model_variant,
        reg_enabled=config.model.reg.enabled,
        reg_variant=config.model.reg.cls_predictor if config.model.reg.enabled else "disabled",
        repa_enabled=config.model.reg.enabled and config.model.reg.repa_enabled,
        reg_token_mode=config.model.reg.token_mode if config.model.reg.enabled else "disabled",
        reg_pool_size=config.model.reg.pool_size,
        reg_x0_conditioning=(
            config.model.reg.x0_conditioning if wide_extensions_active else False
        ),
        reg_continuation_tokens=(
            config.model.reg.continuation_tokens if wide_extensions_active else None
        ),
        reg_patch_reprojection_enabled=(
            config.model.reg.patch_reprojection.enabled
            if wide_extensions_active
            else False
        ),
        reg_patch_reprojection_source=config.model.reg.patch_reprojection.source,
        reg_patch_reprojection_intermediate_size=(
            config.model.reg.patch_reprojection.intermediate_size
        ),
        reg_patch_reprojection_include_timestep=(
            config.model.reg.patch_reprojection.include_timestep
        ),
        checkpoint_hash=digest,
        parameters=parameters,
        generate_batch=generate_batch,
    )


if __name__ == "__main__":
    main()
