from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from jit_reg.checkpoint_bundle import load_checkpoint_bundle
from jit_reg.distributed import broadcast_string, initialize_distributed
from jit_reg.evaluation import evaluate_npz, save_evaluation
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
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--solver", choices=("euler", "heun"))
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--timestep-shift", type=float)
    parser.add_argument("--reg-timestep-ratio", type=float)
    parser.add_argument("--cfg-start", type=float)
    parser.add_argument("--cfg-end", type=float)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="compute FID on rank zero and write evaluation.json beside samples.npz",
    )
    parser.add_argument("--reference-npz", type=Path)
    return parser


def _value(explicit: object | None, inherited: object) -> object:
    return inherited if explicit is None else explicit


def main() -> None:
    args = build_parser().parse_args()
    bundle = load_checkpoint_bundle(args.checkpoint_dir)
    config = bundle.config
    defaults = config.sampling
    reg = config.model.features.reg
    sample_count = int(_value(args.num_samples, defaults.num_samples))
    parameters = SamplingParameters(
        batch_size=int(_value(args.batch_size, defaults.batch_size)),
        sampling_steps=int(
            _value(args.sampling_steps, defaults.sampling_steps)
        ),
        solver=str(_value(args.solver, defaults.solver)),
        guidance_scale=float(
            _value(args.guidance_scale, defaults.guidance_scale)
        ),
        timestep_shift=float(
            _value(args.timestep_shift, defaults.timestep_shift)
        ),
        reg_timestep_ratio=float(
            _value(
                args.reg_timestep_ratio,
                reg.timestep.sample_ratio if reg.enabled else 1.0,
            )
        ),
        cfg_start=float(_value(args.cfg_start, defaults.cfg_start)),
        cfg_end=float(_value(args.cfg_end, defaults.cfg_end)),
        log_interval=int(_value(args.log_interval, defaults.log_interval)),
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
                (
                    reg.num_tokens,
                    config.model.features.encoder.feature_dim,
                ),
                device,
                dtype=dtype,
                seed_offset=1_000_003,
            )
            if reg.enabled
            else None
        )
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            generated = ode_sample(
                execution_model,
                noise,
                batch_labels,
                reg_noise=reg_noise,
                steps=parameters.sampling_steps,
                guidance_scale=parameters.guidance_scale,
                timestep_shift=parameters.timestep_shift,
                reg_timestep_ratio=parameters.reg_timestep_ratio,
                cfg_interval=(parameters.cfg_start, parameters.cfg_end),
                solver=parameters.solver,
                num_classes=config.model.num_classes,
            )
        image = generated.image if isinstance(generated, ODESample) else generated
        return image.float().add(1).div(2)

    output_dir = bundle.directory / sampling_output_name(sample_count, parameters)
    archive = sample_to_archive(
        output_dir=output_dir,
        plan=plan,
        sample_count=sample_count,
        num_classes=config.model.num_classes,
        image_size=config.model.image_size,
        patch_size=config.model.patch_size,
        model_variant=config.model.model_variant,
        feature_schema={
            "schema_version": 2,
            **asdict(config.model.features),
        },
        checkpoint_hash=digest,
        parameters=parameters,
        generate_batch=generate_batch,
    )
    if args.evaluate and rank == 0:
        del execution_model, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        reference = args.reference_npz or Path(config.evaluation.reference_npz)
        eval_device = (
            device
            if config.evaluation.device == "auto"
            else torch.device(config.evaluation.device)
        )
        metrics = evaluate_npz(
            archive,
            reference,
            batch_size=config.evaluation.batch_size,
            device=eval_device,
            log_interval=config.evaluation.log_interval,
        )
        save_evaluation(
            output_dir / "evaluation.json",
            metrics,
            plan.plan_hash,
        )
        print(f"evaluation: {output_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
