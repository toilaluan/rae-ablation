from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from copy import deepcopy
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, DistributedSampler

from jit_reg.codec_training import (
    LPIPSReconstructionLoss,
    codec_learning_rate,
    codec_training_step,
    compute_pixel_token_statistics,
    make_codec_loader,
    make_statistics_loader,
    validate_codec,
)
from jit_reg.config import load_config
from jit_reg.data import data_fingerprint, load_imagenet, preprocessing_metadata
from jit_reg.distributed import initialize_distributed
from jit_reg.pixel_codec import PixelCodec, save_pixel_codec_artifact
from jit_reg.token_cache import TokenCache
from jit_reg.wandb_utils import RunLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the aligned 1024x8x8 pixel codec")
    parser.add_argument("--config", type=Path, default=Path("configs/pixel-codec.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pixel-codec"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def _validation_loader(config: object, dataset: object, rank: int, world_size: int) -> DataLoader:
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    batch_size = max(1, config.codec_train.global_batch_size // world_size)  # type: ignore[attr-defined]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=config.data.workers,  # type: ignore[attr-defined]
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.data.workers > 0,  # type: ignore[attr-defined]
        prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,  # type: ignore[attr-defined]
    )


def _save_training_checkpoint(
    output: Path,
    epoch: int,
    global_step: int,
    model: PixelCodec,
    optimizer: torch.optim.Optimizer,
    best_lpips: float,
) -> None:
    target = output / "checkpoints" / f"epoch-{epoch:04d}"
    if target.exists():
        raise FileExistsError(f"codec checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.partial-{os.getpid()}"
    temporary.mkdir()
    try:
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
            str(temporary / "model.safetensors"),
        )
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "best_lpips": best_lpips,
                "cpu_rng": torch.random.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            temporary / "training.pt",
        )
        os.replace(temporary, target)
        link = output / "checkpoints" / "last"
        pending = output / "checkpoints" / f".last-{os.getpid()}"
        pending.symlink_to(target.name, target_is_directory=True)
        os.replace(pending, link)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.override)
    config.validate()
    rank, world_size, device = initialize_distributed()
    torch.manual_seed(config.train.seed + rank)

    train_dataset = load_imagenet(config.data, 256, load_pixels=True)
    token_cache = TokenCache(config, len(train_dataset))
    validation_data = deepcopy(config.data)
    validation_data.split = "validation"
    validation_data.max_samples = None
    validation_dataset = load_imagenet(validation_data, 256, load_pixels=True)
    train_loader, train_sampler = make_codec_loader(train_dataset, token_cache, config)
    validation_loader = _validation_loader(config, validation_dataset, rank, world_size)

    model = PixelCodec(config.pixel_codec).to(device)
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.codec_train.lr,
        betas=(config.codec_train.beta1, config.codec_train.beta2),
        weight_decay=config.codec_train.weight_decay,
    )
    reconstruction_loss = LPIPSReconstructionLoss(config.pixel_codec.lpips_weight).to(device)
    start_epoch = 0
    global_step = 0
    best_lpips = float("inf")
    if args.resume is not None:
        model.load_state_dict(load_file(str(args.resume / "model.safetensors"), device=str(device)))
        state = torch.load(args.resume / "training.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
        best_lpips = float(state["best_lpips"])
        torch.random.set_rng_state(state["cpu_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    execution_model: torch.nn.Module = model
    if world_size > 1:
        execution_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    logger = RunLogger(config.wandb, config.as_dict(), rank == 0, log_interval=args.log_interval)
    args.output.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad()
    for epoch_index in range(start_epoch, config.codec_train.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_index)
        execution_model.train()
        started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            progress = epoch_index + batch_index / max(len(train_loader), 1)
            lr = codec_learning_rate(config, progress)
            for group in optimizer.param_groups:
                group["lr"] = lr
            metrics = codec_training_step(
                model,
                execution_model,
                batch,
                reconstruction_loss,
                optimizer,
                config,
                token_cache,
                device,
            )
            global_step += 1
            logger.log(
                {
                    **{f"codec/train_{name}": value for name, value in metrics.__dict__.items()},
                    "codec/lr": lr,
                    "codec/epoch": progress,
                },
                step=global_step,
            )
            if rank == 0 and global_step % args.log_interval == 0:
                logger.event(
                    f"codec: step={global_step} epoch={progress:.3f} "
                    f"reconstruction={metrics.reconstruction:.6f} vf={metrics.vf:.6f} "
                    f"vf_weight={metrics.alignment_weight:.4f}"
                )
            if config.codec_train.max_steps is not None and global_step >= config.codec_train.max_steps:
                break
        completed_epoch = epoch_index + 1
        final = completed_epoch == config.codec_train.epochs or (
            config.codec_train.max_steps is not None and global_step >= config.codec_train.max_steps
        )
        if completed_epoch % config.codec_train.validation_interval == 0 or final:
            validation = validate_codec(model, validation_loader, reconstruction_loss, device)
            logger.log(validation, step=global_step)
            if rank == 0 and validation["codec/validation_lpips"] < best_lpips:
                best_lpips = validation["codec/validation_lpips"]
                save_file(
                    {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
                    str(args.output / "best-model.safetensors"),
                )
                (args.output / "best.json").write_text(
                    json.dumps({"epoch": completed_epoch, **validation}, indent=2, sort_keys=True) + "\n"
                )
        if rank == 0 and (
            completed_epoch % config.codec_train.checkpoint_interval == 0 or final
        ):
            _save_training_checkpoint(
                args.output, completed_epoch, global_step, model, optimizer, best_lpips
            )
        logger.log({"codec/epoch_seconds": time.perf_counter() - started}, step=global_step)
        if world_size > 1:
            torch.distributed.barrier()
        if final:
            break

    if not (args.output / "best-model.safetensors").is_file():
        raise RuntimeError("codec training produced no validated best checkpoint")
    if world_size > 1:
        torch.distributed.barrier()
    model.load_state_dict(load_file(str(args.output / "best-model.safetensors"), device=str(device)))
    statistics_loader = make_statistics_loader(train_dataset, token_cache, config)
    statistics = compute_pixel_token_statistics(model, statistics_loader, token_cache, device)
    if rank == 0:
        artifact = save_pixel_codec_artifact(
            args.output / "artifact",
            model,
            statistics,
            metadata={
                "data_fingerprint": data_fingerprint(config.data, 256),
                "preprocessing": preprocessing_metadata(config.data, 256),
                "token_cache_fingerprint": token_cache.fingerprint,
                "best_validation": json.loads((args.output / "best.json").read_text()),
            },
        )
        link = args.output / "best"
        pending = args.output / f".best-{os.getpid()}"
        pending.symlink_to(artifact.name, target_is_directory=True)
        os.replace(pending, link)
        logger.event(f"pixel codec artifact complete: {artifact}")
    if world_size > 1:
        torch.distributed.barrier()
    logger.finish()


if __name__ == "__main__":
    main()
