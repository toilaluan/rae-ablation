from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from jit_reg.config import load_config
from jit_reg.distributed import broadcast_string, initialize_distributed
from jit_reg.model.compile import compile_model_regions
from jit_reg.model.sit import SiT
from jit_reg.sample_runtime import (
    SamplingParameters,
    checkpoint_sha256,
    sample_to_archive,
)
from jit_reg.sampling import EulerSample, euler_sample
from jit_reg.validation import ValidationPlan, noise_for_samples

torch._dynamo.config.cache_size_limit = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a JiT EMA export into an ADM-compatible NPZ"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/canonical.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    reg = parser.add_mutually_exclusive_group()
    reg.add_argument(
        "--reg", action="store_true", help="sample a standard REG-enabled checkpoint"
    )
    reg.add_argument(
        "--reg-widehead",
        action="store_true",
        help="sample a wide-intermediate REG checkpoint",
    )
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.78)
    parser.add_argument("--timestep-shift", type=float, default=1.0)
    parser.add_argument("--cfg-start", type=float, default=0.0)
    parser.add_argument("--cfg-end", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.override)
    if args.reg:
        config.model.reg.enabled = True
        config.model.reg.cls_predictor = "standard"
    elif args.reg_widehead:
        config.model.reg.enabled = True
        config.model.reg.cls_predictor = "wide_intermediate"
    config.validate()
    parameters = SamplingParameters(
        batch_size=args.batch_size,
        sampling_steps=args.sampling_steps,
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
    plan = ValidationPlan.load(args.plan)
    digest = checkpoint_sha256(args.checkpoint) if rank == 0 else None
    digest = broadcast_string(digest, rank)

    model = SiT(config.model).to(device)
    model.load_state_dict(load_file(str(args.checkpoint), device=str(device)))
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
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            generated = euler_sample(
                execution_model,
                noise,
                batch_labels,
                reg_noise=reg_noise,
                steps=parameters.sampling_steps,
                guidance_scale=parameters.guidance_scale,
                timestep_shift=parameters.timestep_shift,
                cfg_interval=(parameters.cfg_start, parameters.cfg_end),
                num_classes=config.model.num_classes,
            )
        image = generated.image if isinstance(generated, EulerSample) else generated
        return image.float().add(1).div(2)

    sample_to_archive(
        output_dir=args.output_dir,
        plan=plan,
        sample_count=args.num_samples,
        num_classes=config.model.num_classes,
        image_size=config.model.image_size,
        patch_size=config.model.patch_size,
        model_variant=config.model.model_variant,
        reg_enabled=config.model.reg.enabled,
        reg_variant=config.model.reg.cls_predictor
        if config.model.reg.enabled
        else "disabled",
        reg_token_mode=config.model.reg.token_mode
        if config.model.reg.enabled
        else "disabled",
        reg_pool_size=config.model.reg.pool_size,
        checkpoint_hash=digest,
        parameters=parameters,
        generate_batch=generate_batch,
    )


if __name__ == "__main__":
    main()
