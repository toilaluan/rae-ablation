from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file

from jit_reg.checkpoint import EMA
from jit_reg.checkpoint_bundle import load_checkpoint_bundle
from jit_reg.data import data_fingerprint
from jit_reg.distributed import initialize_distributed
from jit_reg.latent_statistics import (
    EndpointLatentNormalizer,
    latent_statistics_sha256,
    load_latent_statistics,
)
from jit_reg.model import SemanticDiT
from jit_reg.pixel_decoder import (
    decoder_latent_statistics_sha256,
    decoder_weights_path,
)
from jit_reg.sample_runtime import checkpoint_sha256
from jit_reg.validation_runtime import validate_ema
from jit_reg.vision import vision_fingerprint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a semantic bridge checkpoint on paired validation images"
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--solver", choices=("euler", "heun"))
    parser.add_argument("--timestep-shift", type=float)
    parser.add_argument("--noise-seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-rfid", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = load_checkpoint_bundle(args.checkpoint_dir)
    config = bundle.config
    if args.steps is not None:
        config.sampling.steps = args.steps
    if args.solver is not None:
        config.sampling.solver = args.solver
    if args.timestep_shift is not None:
        config.sampling.timestep_shift = args.timestep_shift
    if args.noise_seed is not None:
        config.sampling.noise_seed = args.noise_seed
    if args.num_samples is not None:
        config.validation.final_samples = args.num_samples
        config.validation.periodic_samples = min(
            config.validation.periodic_samples, args.num_samples
        )
    if args.no_rfid:
        config.evaluation.reconstruction_fid = False
    config.validate()

    rank, _world_size, device = initialize_distributed()
    statistics_path = Path(config.vision.statistics_path)
    statistics = load_latent_statistics(
        statistics_path,
        expected_channels=config.vision.hidden_size,
        expected_vision_fingerprint=vision_fingerprint(config.vision),
        expected_data_fingerprint=data_fingerprint(config.data, config.vision.image_size),
    )
    normalizer = EndpointLatentNormalizer(statistics).to(device)
    metadata = json.loads((bundle.directory / "metadata.json").read_text())
    statistics_digest = latent_statistics_sha256(statistics_path)
    if metadata.get("latent_statistics_sha256") != statistics_digest:
        raise ValueError("latent statistics do not match checkpoint provenance")
    if metadata.get("feature_fingerprint") != vision_fingerprint(config.vision):
        raise ValueError("checkpoint DINO provenance does not match its configuration")
    if bundle.endpoint_kind == "target":
        if config.decoder.checkpoint is None:
            raise ValueError("target checkpoint has no configured pixel decoder")
        decoder_digest = checkpoint_sha256(decoder_weights_path(config.decoder.checkpoint))
        if metadata.get("decoder_sha256") != decoder_digest:
            raise ValueError("configured pixel decoder does not match checkpoint provenance")
        if decoder_latent_statistics_sha256(config.decoder.checkpoint) != statistics_digest:
            raise ValueError("pixel decoder and flow use different latent statistics")

    model = SemanticDiT(config.model).to(device)
    ema = EMA(model, config.flow.train.ema_decay)
    ema.load_state_dict(load_file(str(bundle.weights), device=str(device)))
    output_root = args.output or bundle.directory / "paired-evaluation"
    metrics, output = validate_ema(
        ema=ema,
        config=config,
        plan=bundle.plan,
        completed_epoch=int(metadata["completed_epoch"]),
        final=True,
        output_root=output_root,
        normalizer=normalizer,
    )
    if rank == 0:
        print(json.dumps(metrics, indent=2, sort_keys=True))
        print(f"paired evaluation: {output / 'metrics.json'}")


if __name__ == "__main__":
    main()
