from __future__ import annotations

import argparse
from pathlib import Path

import torch

from jit_reg.checkpoint import EMA
from jit_reg.config import load_config
from jit_reg.data import data_fingerprint, load_imagenet, preprocessing_metadata
from jit_reg.decoder_training import (
    DecoderState,
    load_decoder_checkpoint,
    make_decoder_loader,
    train_decoder,
)
from jit_reg.distributed import initialize_distributed
from jit_reg.latent_statistics import (
    SharedLatentNormalizer,
    latent_statistics_sha256,
    load_latent_statistics,
)
from jit_reg.model.compile import compile_model_regions
from jit_reg.pixel_decoder import (
    PatchDiscriminator,
    ReconstructionObjective,
    TransformerPixelDecoder,
)
from jit_reg.vision import load_dino_encoder, vision_fingerprint
from jit_reg.wandb_utils import RunLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the DINO target-token pixel decoder")
    parser.add_argument("--config", type=Path, default=Path("configs/dino-tokenmix.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path, default=Path("decoder-runs/default"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.override)
    rank, world_size, device = initialize_distributed()
    statistics_path = Path(config.vision.statistics_path).resolve()
    config.vision.statistics_path = str(statistics_path)
    statistics = load_latent_statistics(
        statistics_path,
        expected_channels=config.vision.hidden_size,
        expected_vision_fingerprint=vision_fingerprint(config.vision),
        expected_data_fingerprint=data_fingerprint(config.data, config.vision.image_size),
    )
    statistics_digest = latent_statistics_sha256(statistics_path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.decoder.train.seed)
        decoder = TransformerPixelDecoder(config.decoder).to(device)
        discriminator = PatchDiscriminator(config.decoder.discriminator_channels).to(device)
    feature_encoder = load_dino_encoder(
        config.vision,
        device,
        include_semantic=False,
        normalizer=SharedLatentNormalizer(statistics),
    )
    objective = ReconstructionObjective(config.decoder).to(device)
    generator_optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=config.decoder.train.lr, betas=(0.9, 0.95)
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=config.decoder.train.lr, betas=(0.5, 0.9)
    )
    ema = EMA(decoder, config.decoder.train.ema_decay)
    state = DecoderState()
    if args.resume:
        state = load_decoder_checkpoint(
            args.resume,
            decoder,
            ema,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            config,
            statistics_digest,
        )
    else:
        torch.manual_seed(config.decoder.train.seed + rank)
        if device.type == "cuda":
            torch.cuda.manual_seed(config.decoder.train.seed + rank)
    execution_decoder = compile_model_regions(decoder, config.compile)
    execution_discriminator: torch.nn.Module = discriminator
    if world_size > 1:
        execution_decoder = torch.nn.parallel.DistributedDataParallel(
            execution_decoder,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
        execution_discriminator = torch.nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    train_dataset = load_imagenet(
        config.data, config.vision.image_size, load_pixels=True, split=config.data.split
    )
    validation_dataset = load_imagenet(
        config.data,
        config.vision.image_size,
        load_pixels=True,
        split=config.data.validation_split,
    )
    train_loader, sampler = make_decoder_loader(train_dataset, config, training=True)
    validation_loader, _ = make_decoder_loader(validation_dataset, config, training=False)
    logged_config = config.as_dict()
    logged_config["stage"] = "pixel_decoder"
    logged_config["provenance"] = {
        "dataset": preprocessing_metadata(config.data, config.vision.image_size),
        "data_fingerprint": data_fingerprint(config.data, config.vision.image_size),
        "vision_fingerprint": vision_fingerprint(config.vision),
        "latent_statistics_sha256": statistics_digest,
    }
    logger = RunLogger(
        config.wandb,
        logged_config,
        rank == 0,
        state.wandb_run_id,
        args.log_interval,
    )
    try:
        train_decoder(
            canonical_decoder=decoder,
            execution_decoder=execution_decoder,
            feature_encoder=feature_encoder,
            canonical_discriminator=discriminator,
            execution_discriminator=execution_discriminator,
            objective=objective,
            train_loader=train_loader,
            validation_loader=validation_loader,
            sampler=sampler,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            ema=ema,
            config=config,
            output=args.output,
            logger=logger,
            state=state,
            latent_statistics_sha256=statistics_digest,
        )
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
