from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from .checkpoint import EMA
from .config import ExperimentConfig
from .data import unlabeled_data_fingerprint
from .model import SemanticPixelDecoder
from .model.compile import compile_model_regions
from .objective import build_lpips, pixel_reconstruction_loss
from .optim import build_optimizer, build_training_scheduler
from .pixel_artifact import (
    PIXEL_DECODER_TRAINING_SCHEMA_VERSION,
    PixelDecoderCheckpointManager,
    PixelDecoderResumeState,
    pixel_decoder_architecture_fingerprint,
    resolve_pixel_decoder_checkpoint,
)
from .pixel_data import (
    PixelBatchPreprocessor,
    load_pixel_semantic_statistics,
    make_pixel_pretraining_loader,
    set_pixel_loader_epoch,
)
from .pixel_preparation import prepare_pixel_pretraining_inputs
from .vision import encoder_fingerprint, load_dino_encoder

PixelLog = Callable[[dict[str, float | int | str]], None]
PixelImageLog = Callable[[str, str, int, str], None]


def _all_reduce(values: Tensor) -> Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(values)
    return values


@torch.inference_mode()
def validate_pixel_decoder(
    model: nn.Module,
    loader: object,
    preprocessor: PixelBatchPreprocessor,
    lpips_model: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, float], Tensor | None]:
    was_training = model.training
    model.eval()
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    preview: Tensor | None = None
    try:
        for batch_index, raw_batch in enumerate(loader):  # type: ignore[union-attr]
            if batch_index >= config.pixel_pretraining.validation_batches:
                break
            semantic, pixels = preprocessor(raw_batch)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(semantic)
                loss = pixel_reconstruction_loss(
                    prediction, pixels, lpips_model, config.objective
                )
            mse = (prediction.float() - pixels.float()).square().flatten(1).mean(1)
            psnr = -10 * torch.log10(mse.clamp_min(1e-12))
            count = pixels.shape[0]
            totals += torch.tensor(
                [
                    float(loss.total) * count,
                    float(loss.pixel_l1) * count,
                    float(loss.pixel_lpips) * count,
                    float(psnr.sum()),
                    count,
                ],
                dtype=torch.float64,
                device=device,
            )
            if preview is None:
                preview = prediction[:16].float().clamp(0, 1).cpu()
    finally:
        model.train(was_training)
    _all_reduce(totals)
    count = totals[-1].item()
    if count < 1:
        raise ValueError("pixel validation loader produced no batches")
    return (
        {
            "loss": totals[0].item() / count,
            "pixel_l1": totals[1].item() / count,
            "pixel_lpips": totals[2].item() / count,
            "psnr": totals[3].item() / count,
        },
        preview,
    )


def _save_preview(images: Tensor, path: Path) -> None:
    if images.ndim != 4 or images.shape[1:] != (3, 128, 128):
        raise ValueError("pixel preview requires [B,3,128,128] images")
    count = min(images.shape[0], 16)
    canvas = np.full((4 * 128, 4 * 128, 3), 255, dtype=np.uint8)
    values = images[:count].permute(0, 2, 3, 1).mul(255).round().byte().numpy()
    for index, image in enumerate(values):
        row, column = divmod(index, 4)
        canvas[row * 128 : (row + 1) * 128, column * 128 : (column + 1) * 128] = image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


class PixelPretrainingRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device | str = "cpu",
        log: PixelLog | None = None,
        log_image: PixelImageLog | None = None,
        resume: str | Path | None = None,
    ) -> None:
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.log = log or (lambda _metrics: None)
        self.log_image = log_image or (
            lambda _path, _caption, _step, _key: None
        )
        prepare_pixel_pretraining_inputs(
            config, rank=rank, world_size=world_size, device=self.device
        )
        self.statistics = load_pixel_semantic_statistics(config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.experiment.seed + 70_001)
            self.model = SemanticPixelDecoder(config.model).to(self.device)
        torch.manual_seed(config.experiment.seed + 70_002)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.experiment.seed + 70_002)
        self.ema = EMA(self.model, config.pixel_pretraining.optimizer.ema_decay)
        self.optimizer = build_optimizer(
            self.model, config.pixel_pretraining.optimizer
        )
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.pending_scheduler_state: dict[str, object] | None = None
        self.lpips_model = build_lpips(self.device)
        self.encoder: nn.Module | None = None
        if config.pixel_pretraining.semantic_input_mode == "online":
            semantic = config.representations.semantic
            self.encoder = load_dino_encoder(
                config.encoder,
                [semantic.layer],
                self.device,
                compile_config=config.compile,
            )
        self.preprocessor = PixelBatchPreprocessor(
            self.statistics,
            self.device,
            self.encoder,
            config.representations.semantic.layer,
        )
        output = Path(config.experiment.output_directory)
        self.manager = PixelDecoderCheckpointManager(output / "checkpoints")
        self.completed_epoch = 0
        self.global_step = 0
        self.started_at = time.perf_counter()
        self.elapsed_before_resume = 0.0
        if resume is not None:
            self.resume(resume)

    def resume(self, checkpoint: str | Path) -> None:
        state, saved_config, saved_statistics = self.manager.load_training(
            resolve_pixel_decoder_checkpoint(checkpoint), self.model, self.ema
        )
        if (
            pixel_decoder_architecture_fingerprint(saved_config.model)
            != pixel_decoder_architecture_fingerprint(self.config.model)
            or saved_config.pixel_pretraining != self.config.pixel_pretraining
            or saved_config.encoder != self.config.encoder
            or saved_config.objective != self.config.objective
        ):
            raise ValueError("pixel pretraining resume configuration mismatch")
        if not torch.equal(saved_statistics.mean, self.statistics.mean) or not torch.equal(
            saved_statistics.std, self.statistics.std
        ):
            raise ValueError("pixel pretraining resume statistics mismatch")
        if state.completed_epoch > self.config.pixel_pretraining.epochs:
            raise ValueError("pixel checkpoint epoch exceeds the active schedule")
        data = self.config.pixel_pretraining.data
        expected_data = unlabeled_data_fingerprint(
            data,
            split=data.train_split,
            max_samples=data.max_train_samples,
        )
        expected_encoder = encoder_fingerprint(
            self.config.encoder, [self.config.representations.semantic.layer]
        )
        if (
            state.data_fingerprint != expected_data
            or state.encoder_fingerprint != expected_encoder
        ):
            raise ValueError("pixel pretraining resume provenance mismatch")
        self.completed_epoch = state.completed_epoch
        self.global_step = state.global_step
        self.elapsed_before_resume = state.wall_clock_seconds
        self.started_at = time.perf_counter()
        cpu_state = state.rng_state.get("cpu")
        if not isinstance(cpu_state, Tensor):
            raise ValueError("pixel checkpoint CPU RNG state is invalid")
        torch.set_rng_state(cpu_state)
        cuda_state = state.rng_state.get("cuda")
        if self.device.type == "cuda" and cuda_state is not None:
            if not isinstance(cuda_state, list):
                raise ValueError("pixel checkpoint CUDA RNG state is invalid")
            torch.cuda.set_rng_state_all(cuda_state)
        if state.optimizer is not None:
            self.optimizer.load_state_dict(state.optimizer)
            self.pending_scheduler_state = state.scheduler

    def _resume_state(self, complete: bool) -> PixelDecoderResumeState:
        data = self.config.pixel_pretraining.data
        return PixelDecoderResumeState(
            schema_version=PIXEL_DECODER_TRAINING_SCHEMA_VERSION,
            completed_epoch=self.completed_epoch,
            global_step=self.global_step,
            training_complete=complete,
            optimizer=self.optimizer.state_dict(),
            scheduler=None if self.scheduler is None else self.scheduler.state_dict(),
            rng_state={
                "cpu": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if self.device.type == "cuda"
                    else None
                ),
            },
            wall_clock_seconds=(
                self.elapsed_before_resume + time.perf_counter() - self.started_at
            ),
            data_fingerprint=(
                unlabeled_data_fingerprint(
                    data,
                    split=data.train_split,
                    max_samples=data.max_train_samples,
                )
            ),
            encoder_fingerprint=encoder_fingerprint(
                self.config.encoder, [self.config.representations.semantic.layer]
            ),
        )

    def _build_execution(self, steps_per_epoch: int) -> nn.Module:
        if self.scheduler is None:
            values = self.config.pixel_pretraining.optimizer.scheduler
            self.scheduler = build_training_scheduler(
                self.optimizer,
                kind=values.kind,
                steps_per_epoch=steps_per_epoch,
                total_epochs=self.config.pixel_pretraining.epochs,
                warmup_epochs=values.warmup_epochs,
                min_lr_ratio=values.min_lr_ratio,
            )
            if self.pending_scheduler_state is not None:
                self.scheduler.load_state_dict(self.pending_scheduler_state)
                self.pending_scheduler_state = None
        execution = compile_model_regions(self.model, self.config.compile)
        if self.world_size > 1:
            execution = torch.nn.parallel.DistributedDataParallel(
                execution,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
            )
        return execution

    def run(self, *, max_steps: int | None = None) -> None:
        train_loader = make_pixel_pretraining_loader(
            self.config,
            split_kind="train",
            rank=self.rank,
            world_size=self.world_size,
            epoch=self.completed_epoch,
        )
        validation_loader = make_pixel_pretraining_loader(
            self.config,
            split_kind="validation",
            rank=self.rank,
            world_size=self.world_size,
        )
        execution = self._build_execution(len(train_loader))
        for epoch in range(
            self.completed_epoch, self.config.pixel_pretraining.epochs
        ):
            set_pixel_loader_epoch(train_loader, epoch)
            execution.train()
            for raw_batch in train_loader:
                step_started = time.perf_counter()
                semantic, pixels = self.preprocessor(raw_batch)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    prediction = execution(semantic)
                    loss = pixel_reconstruction_loss(
                        prediction, pixels, self.lpips_model, self.config.objective
                    )
                loss.total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.pixel_pretraining.optimizer.gradient_clip,
                )
                self.optimizer.step()
                self.ema.update(self.model)
                assert self.scheduler is not None
                self.scheduler.step()
                self.global_step += 1
                self.log(
                    {
                        "epoch": epoch,
                        "step": self.global_step,
                        "loss": float(loss.total.detach()),
                        "pixel_l1": float(loss.pixel_l1.detach()),
                        "pixel_lpips": float(loss.pixel_lpips.detach()),
                        "gradient_norm": float(gradient_norm),
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "images_per_second": (
                            pixels.shape[0]
                            * self.world_size
                            / max(time.perf_counter() - step_started, 1e-9)
                        ),
                        "semantic_input_mode": self.config.pixel_pretraining.semantic_input_mode,
                    }
                )
                if max_steps is not None and self.global_step >= max_steps:
                    self._write_metrics(complete=False)
                    return
            self.completed_epoch = epoch + 1
            complete = self.completed_epoch == self.config.pixel_pretraining.epochs
            validation_interval = self.config.pixel_pretraining.validation_interval_epochs
            if complete or self.completed_epoch % validation_interval == 0:
                raw_metrics, _raw_preview = validate_pixel_decoder(
                    self.model,
                    validation_loader,
                    self.preprocessor,
                    self.lpips_model,
                    self.config,
                    self.device,
                )
                ema_model = self.ema.to_model(self.model).to(self.device)
                ema_metrics, preview = validate_pixel_decoder(
                    ema_model,
                    validation_loader,
                    self.preprocessor,
                    self.lpips_model,
                    self.config,
                    self.device,
                )
                del ema_model
                self.log(
                    {"epoch": epoch, "step": self.global_step}
                    | {
                        f"validation/raw_{name}": value
                        for name, value in raw_metrics.items()
                    }
                    | {
                        f"validation/ema_{name}": value
                        for name, value in ema_metrics.items()
                    }
                )
                if self.rank == 0 and preview is not None:
                    path = (
                        Path(self.config.experiment.output_directory)
                        / "validation"
                        / f"epoch-{self.completed_epoch:04d}.png"
                    )
                    _save_preview(preview, path)
                    self.log_image(
                        str(path),
                        f"Pixel decoder EMA reconstructions, epoch={self.completed_epoch}",
                        self.global_step,
                        "validation/pixel_reconstructions",
                    )
            checkpoint_interval = self.config.pixel_pretraining.checkpoint_interval_epochs
            if self.rank == 0 and (
                complete or self.completed_epoch % checkpoint_interval == 0
            ):
                self.manager.save(
                    model=self.model,
                    ema=self.ema,
                    state=self._resume_state(complete),
                    config=self.config,
                    statistics=self.statistics,
                )
            if self.world_size > 1:
                torch.distributed.barrier()
        self._write_metrics(complete=True)

    def _write_metrics(self, *, complete: bool) -> None:
        if self.rank != 0:
            return
        output = Path(self.config.experiment.output_directory)
        output.mkdir(parents=True, exist_ok=True)
        metrics = {
            "complete": complete,
            "completed_epoch": self.completed_epoch,
            "global_step": self.global_step,
            "semantic_input_mode": self.config.pixel_pretraining.semantic_input_mode,
            "wall_clock_seconds": (
                self.elapsed_before_resume + time.perf_counter() - self.started_at
            ),
        }
        (output / "pixel-pretraining-metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
