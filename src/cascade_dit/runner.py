from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from .checkpoint import EMA, CheckpointManager, ResumeState
from .config import ExperimentConfig, load_checkpoint_config
from .curriculum import configure_trainability, stage_at_epoch
from .data import (
    MIXTURE_MODE_NAMES,
    RepresentationBatchPreprocessor,
    data_fingerprint,
    make_stage_loader,
)
from .feature_pca import save_feature_pca_preview
from .flops import model_flops_per_step
from .model import DDT, build_model
from .model.compile import compile_model_regions
from .optim import (
    build_optimizer,
    build_training_scheduler,
    parameter_count,
    trainable_parameter_count,
)
from .preparation import preparation_fingerprint, representation_manifest_fingerprint
from .representations import RepresentationManifest
from .rng import RNGStreams
from .training import compute_training_loss
from .validation import validate_stage_loss


def load_representation_manifest(
    config: ExperimentConfig,
) -> tuple[RepresentationManifest, str]:
    root = Path(config.representations.directory)
    manifest = RepresentationManifest.load(
        root / config.representations.manifest,
        require_complete=config.representations.require_complete,
    )
    if manifest.preparation_fingerprint != preparation_fingerprint(config):
        raise ValueError("prepared representation settings do not match experiment config")
    return manifest, representation_manifest_fingerprint(manifest)


class ExperimentRunner:
    """Dynamic adjacent-flow curriculum or architecture-matched RGB baseline."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device | str = "cpu",
        log: Callable[[dict[str, float | int | str]], None] | None = None,
    ) -> None:
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.log = log or (lambda _metrics: None)
        manifest, self.representation_fingerprint = load_representation_manifest(config)
        self.data_fingerprint = data_fingerprint(config.data)
        self.batch_preprocessor = RepresentationBatchPreprocessor(
            config, manifest, self.device
        )
        self.rng = RNGStreams.create(config.experiment.seed, self.device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.rng.initialization.initial_seed())
            self.model: DDT = build_model(
                config.model, config.num_representation_levels
            ).to(self.device)
        configure_trainability(self.model)
        torch.manual_seed(config.experiment.seed + 90_000)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.experiment.seed + 90_000)
        self.ema = EMA(self.model, config.optimizer.ema_decay)
        self.manager = CheckpointManager(
            Path(config.experiment.output_directory) / config.checkpoint.directory,
            keep_last_n=config.checkpoint.keep_last_n,
        )
        self.completed_epoch = 0
        self.global_step = 0
        self.stage_index = (
            config.num_representation_levels - 1
            if config.experiment.mode == "pixel_baseline"
            else 0
        )
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._pending_scheduler_state: dict[str, object] | None = None
        self.images_seen = 0
        self.estimated_flops = 0
        self.started_at = time.perf_counter()
        self.elapsed_before_resume = 0.0

    @property
    def stage_name(self) -> str:
        return self.config.stage_names[self.stage_index]

    def _epoch_within_stage(self) -> int:
        if self.config.experiment.mode == "pixel_baseline":
            return self.completed_epoch
        beginning = sum(stage.epochs for stage in self.config.stages[: self.stage_index])
        return min(
            max(self.completed_epoch - beginning, 0),
            self.config.stages[self.stage_index].epochs,
        )

    def _resume_state(
        self, transition_state: str, *, training_complete: bool
    ) -> ResumeState:
        return ResumeState(
            schema_version=4,
            completed_epoch=self.completed_epoch,
            global_step=self.global_step,
            stage_index=self.stage_index,
            stage=self.stage_name,
            epoch_within_stage=self._epoch_within_stage(),
            transition_state=transition_state,
            training_complete=training_complete,
            images_seen=self.images_seen,
            estimated_flops=self.estimated_flops,
            wall_clock_seconds=(
                self.elapsed_before_resume + time.perf_counter() - self.started_at
            ),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict() if self.scheduler is not None else None,
            rng_streams=self.rng.state_dict(),
            data_fingerprint=self.data_fingerprint,
            representation_fingerprint=self.representation_fingerprint,
        )

    def resume(self, checkpoint: str | Path) -> None:
        path = Path(checkpoint)
        if path.name == "last" or not (path / "metadata.json").is_file():
            if (path / "last").is_dir():
                path = (path / "last").resolve()
            elif path.name == "last":
                path = path.resolve()
        saved_config = load_checkpoint_config(path / "config.json")
        for name in (
            "data",
            "encoder",
            "representations",
            "model",
            "curriculum",
            "objective",
            "optimizer",
        ):
            if getattr(self.config, name) != getattr(saved_config, name):
                raise ValueError(f"checkpoint {name} configuration mismatch")
        if self.config.experiment.mode != saved_config.experiment.mode:
            raise ValueError("checkpoint experiment mode mismatch")
        self.ema = EMA(self.model, self.config.optimizer.ema_decay)
        state = self.manager.load(
            path,
            self.model,
            self.ema,
            expected_data_fingerprint=self.data_fingerprint,
            expected_representation_fingerprint=self.representation_fingerprint,
        )
        self.completed_epoch = state.completed_epoch
        self.global_step = state.global_step
        self.stage_index = state.stage_index
        self.images_seen = state.images_seen
        self.estimated_flops = state.estimated_flops
        self.elapsed_before_resume = state.wall_clock_seconds
        self.started_at = time.perf_counter()
        self.rng.load_state_dict(state.rng_streams)
        if state.optimizer is not None:
            self.optimizer.load_state_dict(state.optimizer)
        self._pending_scheduler_state = state.scheduler

    def _save_stage_boundary(self, target_index: int) -> None:
        self.stage_index = target_index
        if self.rank == 0 and self.config.checkpoint.save_stage_boundaries:
            self.manager.save_boundary(
                target_stage_index=target_index,
                position="after",
                model=self.model,
                ema=self.ema,
                state=self._resume_state("after", training_complete=False),
                config=self.config,
            )
        if self.world_size > 1:
            torch.distributed.barrier()

    def _build_execution(self, steps_per_epoch: int) -> nn.Module:
        if self.scheduler is None:
            scheduler = self.config.optimizer.scheduler
            self.scheduler = build_training_scheduler(
                self.optimizer,
                steps_per_epoch=steps_per_epoch,
                total_epochs=self.config.total_epochs,
                warmup_epochs=scheduler.warmup_epochs,
                min_lr_ratio=scheduler.min_lr_ratio,
            )
            if self._pending_scheduler_state is not None:
                self.scheduler.load_state_dict(self._pending_scheduler_state)
                self._pending_scheduler_state = None
        execution: nn.Module = compile_model_regions(self.model, self.config.compile)
        if self.world_size > 1:
            execution = torch.nn.parallel.DistributedDataParallel(
                execution,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
            )
        return execution

    def run(self, *, max_steps: int | None = None) -> None:
        execution: nn.Module | None = None
        for epoch in range(self.completed_epoch, self.config.total_epochs):
            pixel_baseline = self.config.experiment.mode == "pixel_baseline"
            if pixel_baseline:
                stage_index = self.config.num_representation_levels - 1
                stage_name = self.config.stage_names[-1]
                epoch_in_stage = epoch
            else:
                position = stage_at_epoch(epoch, self.config)
                stage_index = position.index
                stage_name = position.name
                epoch_in_stage = position.epoch_within_stage
                if stage_index != self.stage_index:
                    self._save_stage_boundary(stage_index)
            loader = make_stage_loader(
                self.config,
                stage_index,
                rank=self.rank,
                world_size=self.world_size,
                epoch=epoch,
                pixel_baseline=pixel_baseline,
            )
            if execution is None:
                execution = self._build_execution(len(loader))
            execution.train()
            source_model = None
            if self.config.curriculum.source_consistency_weight:
                source_model = self.ema.to_model(self.model).to(self.device).eval()

            loader_iterator = iter(loader)
            while True:
                load_started_at = time.perf_counter()
                try:
                    raw_batch = next(loader_iterator)
                except StopIteration:
                    break
                load_time = time.perf_counter() - load_started_at
                batch = self.batch_preprocessor(raw_batch)
                tokens = batch.tokens
                labels = batch.labels
                weights = batch.representation_weights
                modes = batch.modes
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    result = compute_training_loss(
                        execution,
                        tokens,
                        labels,
                        weights,
                        self.rng,
                        self.config,
                        source_model=source_model,
                    )
                result.loss.total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimizer.gradient_clip
                )
                self.optimizer.step()
                assert self.scheduler is not None
                self.scheduler.step()
                self.ema.update(self.model)
                self.global_step += 1
                self.images_seen += tokens.shape[0] * self.world_size
                self.estimated_flops += model_flops_per_step(self.config)
                metrics: dict[str, float | int | str] = {
                    "stage": stage_name,
                    "epoch": epoch,
                    "epoch_within_stage": epoch_in_stage,
                    "step": self.global_step,
                    "loss": float(result.loss.total.detach()),
                    "velocity_mse": float(result.loss.velocity_mse.detach()),
                    "target_rms": float(tokens.float().square().mean().sqrt()),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(gradient_norm),
                    "load_time": float(load_time),
                }
                for mode, name in enumerate(MIXTURE_MODE_NAMES):
                    metrics[f"mode/{name}"] = int((modes == mode).sum())
                self.log(metrics)
                if max_steps is not None and self.global_step >= max_steps:
                    self._write_run_metrics(complete=False)
                    return
            self.completed_epoch = epoch + 1
            complete = self.completed_epoch == self.config.total_epochs
            interval = self.config.validation.interval_epochs
            if complete or (interval > 0 and self.completed_epoch % interval == 0):
                ema_model = self.ema.to_model(self.model).to(self.device)
                validation, preview, preview_labels = validate_stage_loss(
                    ema_model,
                    loader,
                    self.batch_preprocessor,
                    self.config,
                    stage_index=stage_index,
                    epoch=epoch,
                    device=self.device,
                )
                validation_metrics: dict[str, float | int | str] = {
                    "stage": stage_name,
                    "epoch": epoch,
                    "step": self.global_step,
                }
                validation_metrics.update(
                    {f"validation/{name}": value for name, value in validation.items()}
                )
                self.log(validation_metrics)
                if self.rank == 0 and preview is not None and preview_labels is not None:
                    preview_directory = (
                        Path(self.config.experiment.output_directory) / "validation"
                    )
                    preview_directory.mkdir(parents=True, exist_ok=True)
                    features = preview.transpose(1, 2).reshape(-1, 768, 8, 8)
                    save_feature_pca_preview(
                        features,
                        preview_labels,
                        preview_directory
                        / f"epoch-{self.completed_epoch:04d}-feature-pca.png",
                        128,
                    )
            checkpoint_interval = self.config.checkpoint.interval_epochs
            periodic = (
                checkpoint_interval
                and self.completed_epoch % checkpoint_interval == 0
            )
            if self.rank == 0 and (complete or periodic):
                self.manager.save(
                    model=self.model,
                    ema=self.ema,
                    state=self._resume_state("stable", training_complete=complete),
                    config=self.config,
                )
            if self.world_size > 1:
                torch.distributed.barrier()
        self._write_run_metrics(complete=True)

    def _write_run_metrics(self, *, complete: bool) -> None:
        if self.rank != 0:
            return
        output = Path(self.config.experiment.output_directory)
        output.mkdir(parents=True, exist_ok=True)
        metrics = {
            "complete": complete,
            "mode": self.config.experiment.mode,
            "global_step": self.global_step,
            "completed_epoch": self.completed_epoch,
            "images_seen": self.images_seen,
            "estimated_flops": self.estimated_flops,
            "wall_clock_seconds": (
                self.elapsed_before_resume + time.perf_counter() - self.started_at
            ),
            "peak_memory_bytes": (
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda"
                else 0
            ),
            "total_parameters": parameter_count(self.model),
            "trainable_parameters": trainable_parameter_count(self.model),
        }
        target = output / "training-metrics.json"
        target.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
