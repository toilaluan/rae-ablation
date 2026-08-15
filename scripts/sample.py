from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from jit_reg.checkpoint_bundle import load_checkpoint_bundle
from jit_reg.distributed import broadcast_string, initialize_distributed
from jit_reg.evaluation import evaluate_npz, save_evaluation
from jit_reg.model import DDT
from jit_reg.model.compile import compile_model_regions
from jit_reg.prior import (
    load_checkpoint_prior,
    prior_provenance,
    source_from_gaussian,
    validate_prior_provenance,
)
from jit_reg.rae import load_rae_codec
from jit_reg.sample_runtime import (
    FEATURE_SCHEMA_VERSION,
    SamplingParameters,
    checkpoint_sha256,
    sample_to_archive,
    sampling_output_name,
)
from jit_reg.sampling import ode_sample
from jit_reg.token_cache import load_token_statistics
from jit_reg.validation import noise_for_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample a schema-v13 RAE-DDT checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--solver", choices=("euler", "heun"))
    parser.add_argument("--timestep-shift", type=float)
    parser.add_argument("--ig-scale", type=float)
    parser.add_argument("--ig-start", type=float)
    parser.add_argument("--ig-end", type=float)
    parser.add_argument("--cfg-scale", type=float)
    parser.add_argument("--cfg-start", type=float)
    parser.add_argument("--cfg-end", type=float)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--reference-npz", type=Path)
    return parser


def _value(explicit: object | None, inherited: object) -> object:
    return inherited if explicit is None else explicit


def main() -> None:
    args = build_parser().parse_args()
    bundle = load_checkpoint_bundle(args.checkpoint_dir)
    config, defaults = bundle.config, bundle.config.sampling
    sample_count = int(_value(args.num_samples, defaults.num_samples))
    parameters = SamplingParameters(
        batch_size=int(_value(args.batch_size, defaults.batch_size)),
        sampling_steps=int(_value(args.sampling_steps, defaults.sampling_steps)),
        solver=str(_value(args.solver, defaults.solver)),
        timestep_shift=float(_value(args.timestep_shift, defaults.timestep_shift)),
        ig_scale=float(_value(args.ig_scale, defaults.ig_scale)),
        ig_start=float(_value(args.ig_start, defaults.ig_start)),
        ig_end=float(_value(args.ig_end, defaults.ig_end)),
        cfg_scale=float(_value(args.cfg_scale, defaults.cfg_scale)),
        cfg_start=float(_value(args.cfg_start, defaults.cfg_start)),
        cfg_end=float(_value(args.cfg_end, defaults.cfg_end)),
        log_interval=int(_value(args.log_interval, defaults.log_interval)),
    )
    parameters.validate()
    rank, _world_size, device = initialize_distributed()
    digest = checkpoint_sha256(bundle.weights) if rank == 0 else None
    digest = broadcast_string(digest, rank)
    metadata = __import__("json").loads((bundle.directory / "metadata.json").read_text())
    cache_fingerprint = metadata.get("token_cache_fingerprint")
    if not isinstance(cache_fingerprint, str):
        raise ValueError("checkpoint has no token-cache provenance")
    statistics = load_token_statistics(config.rae.cache.directory, cache_fingerprint)
    prior = load_checkpoint_prior(
        config,
        device,
        token_cache_fingerprint=cache_fingerprint,
    )
    validate_prior_provenance(prior_provenance(prior), metadata.get("prior"))

    model = DDT(config.model).to(device)
    model.load_state_dict(load_file(str(bundle.weights), device=str(device)))
    model.eval()
    execution_model = compile_model_regions(model, config.compile)
    codec = load_rae_codec(config.rae, device)

    @torch.inference_mode()
    def generate_batch(
        _positions: torch.Tensor, labels: torch.Tensor, noise_seeds: torch.Tensor
    ) -> torch.Tensor:
        batch_labels = labels.to(device)
        noise = noise_for_samples(noise_seeds, config.model.input_shape, device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
        source = source_from_gaussian(prior, noise, batch_labels)
        generated = ode_sample(
            execution_model,
            source,
            batch_labels,
            steps=parameters.sampling_steps,
            timestep_shift=parameters.timestep_shift,
            ig_scale=parameters.ig_scale,
            ig_interval=(parameters.ig_start, parameters.ig_end),
            cfg_scale=parameters.cfg_scale,
            cfg_interval=(parameters.cfg_start, parameters.cfg_end),
            solver=parameters.solver,
            num_classes=config.model.num_classes,
            velocity_denominator_floor=config.train.velocity_denominator_floor,
        )
        return codec.decode(statistics.denormalize(generated)).clamp(0, 1)

    output_dir = bundle.directory / sampling_output_name(sample_count, parameters)
    archive = sample_to_archive(
        output_dir=output_dir,
        plan=bundle.plan,
        sample_count=sample_count,
        num_classes=config.model.num_classes,
        image_size=256,
        patch_size=config.model.patch_size[0],
        grid_size=config.model.grid_size,
        model_variant=config.model.model_variant,
        feature_schema={
            "schema_version": FEATURE_SCHEMA_VERSION,
            "input_space": "normalized-raev2-compressor-tokens",
            "input_shape": list(config.model.input_shape),
            "conditioning": "four-timestep-plus-eight-class-tokens",
            "prediction": "clean-x-full-and-base",
            "token_cache_fingerprint": cache_fingerprint,
            "prior": prior_provenance(prior),
            "rae": {
                "base_repository": config.rae.base_repository,
                "base_revision": config.rae.base_revision,
                "compressor_repository": config.rae.compressor_repository,
                "compressor_revision": config.rae.compressor_revision,
            },
        },
        checkpoint_hash=digest,
        parameters=parameters,
        generate_batch=generate_batch,
    )
    if args.evaluate and rank == 0:
        reference = args.reference_npz or Path(config.evaluation.reference_npz)
        eval_device = device if config.evaluation.device == "auto" else torch.device(config.evaluation.device)
        metrics = evaluate_npz(archive, reference, batch_size=config.evaluation.batch_size, device=eval_device, log_interval=config.evaluation.log_interval)
        save_evaluation(output_dir / "evaluation.json", metrics, bundle.plan.plan_hash)


if __name__ == "__main__":
    main()
