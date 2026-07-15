from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from rae_ablation.codecs import load_codecs
from rae_ablation.config import load_config
from rae_ablation.distributed import broadcast_string, initialize_distributed
from rae_ablation.latents import LatentStatistics
from rae_ablation.model.compile import compile_model_regions
from rae_ablation.model.factory import build_diffusion_model
from rae_ablation.sample_runtime import (
    SamplingParameters,
    checkpoint_sha256,
    sample_to_archive,
)
from rae_ablation.sampling import euler_sample
from rae_ablation.validation import ValidationPlan, noise_for_samples

torch._dynamo.config.cache_size_limit = 1000

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample an EMA export into an ADM-compatible NPZ")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/canonical.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--codec", choices=("rae", "rae_compact"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.78)
    parser.add_argument("--timestep-shift", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.override)
    if args.codec:
        config.codec = args.codec
    if args.cache_dir:
        config.cache.directory = str(args.cache_dir)
    config.validate()
    parameters = SamplingParameters(
        batch_size=args.batch_size,
        sampling_steps=args.sampling_steps,
        guidance_scale=args.guidance_scale,
        timestep_shift=args.timestep_shift,
        log_interval=args.log_interval,
    )
    parameters.validate()
    rank, _world_size, device = initialize_distributed()
    if rank == 0:
        print(f"initializing sampler on {device}")
    plan = ValidationPlan.load(args.plan)
    digest = checkpoint_sha256(args.checkpoint) if rank == 0 else None
    digest = broadcast_string(digest, rank)

    model = build_diffusion_model(config.model).to(device)
    model.load_state_dict(load_file(str(args.checkpoint), device=str(device)))
    model.eval()
    execution_model = compile_model_regions(model, config.compile)
    statistics = LatentStatistics.load(
        Path(config.cache.directory) / f"stats-{config.codec}.safetensors"
    )
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
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ), torch.no_grad():
            normalized = euler_sample(
                execution_model,
                noise,
                batch_labels,
                steps=parameters.sampling_steps,
                guidance_scale=parameters.guidance_scale,
                timestep_shift=parameters.timestep_shift,
            )
            return codec.decode(statistics.denormalize(normalized))

    sample_to_archive(
        output_dir=args.output_dir,
        plan=plan,
        sample_count=args.num_samples,
        num_classes=config.model.num_classes,
        codec=config.codec,
        checkpoint_hash=digest,
        parameters=parameters,
        generate_batch=generate_batch,
    )


if __name__ == "__main__":
    main()
