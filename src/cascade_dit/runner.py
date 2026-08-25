from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from .checkpoint import EMA, CheckpointManager, ResumeState
from .config import ExperimentConfig, load_checkpoint_config
from .curriculum import configure_trainability, stage_at_epoch, transition_to_stage
from .data import data_fingerprint, make_stage_loader
from .feature_pca import save_feature_pca_preview
from .flops import model_flops_per_step
from .model import LightningDiT, build_model
from .model.compile import compile_model_regions
from .optim import (
    build_optimizer,
    build_stage_scheduler,
    parameter_count,
    trainable_parameter_count,
)
from .preparation import preparation_fingerprint, representation_manifest_fingerprint
from .representations import RepresentationManifest
from .rng import RNGStreams
from .training import compute_training_loss
from .validation import validate_stage_loss


def load_cumulative_projectors(config: ExperimentConfig) -> tuple[dict[str, Tensor], str]:
    root = Path(config.representations.directory)
    manifest = RepresentationManifest.load(
        root / config.representations.manifest,
        require_complete=config.representations.require_complete,
    )
    if manifest.preparation_fingerprint != preparation_fingerprint(config):
        raise ValueError("prepared representation settings do not match the experiment config")
    tensors = load_file(root / manifest.tensor_file)
    projectors = {
        stage: tensors[key].float()
        for stage, key in manifest.cumulative_projectors.items()
    }
    if set(projectors) != set(config.curriculum.stages):
        raise ValueError("prepared cumulative projectors do not cover the curriculum")
    return projectors, representation_manifest_fingerprint(manifest)


def configure_model_for_resume(
    model: LightningDiT,
    metadata: dict[str, object],
    projectors: dict[str, Tensor],
    config: ExperimentConfig,
) -> None:
    stage = str(metadata["stage"])
    stage_index = int(metadata["stage_index"])
    training_complete = bool(metadata.get("training_complete"))
    projector = torch.eye(model.config.token_dim) if training_complete else projectors[stage]
    model.set_stage(stage, projector)
    if config.experiment.mode == "pixel_baseline":
        configure_trainability(model, 3, config.curriculum, pixel_baseline=True)
    else:
        if stage_index > 0 and not training_complete:
            model.attach_lora(
                config.curriculum.lora_blocks,
                config.curriculum.lora_rank,
                config.curriculum.lora_alpha,
                config.curriculum.lora_dropout,
            )
        configure_trainability(model, stage_index, config.curriculum)


class ExperimentRunner:
    """Four-stage progressive runner, or the matched 80-epoch pixel baseline."""

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
        self.projectors, self.representation_fingerprint = load_cumulative_projectors(config)
        self.data_fingerprint = data_fingerprint(config.data)
        self.rng = RNGStreams.create(config.experiment.seed, self.device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.rng.initialization.initial_seed())
            self.model = build_model(config.model, config.detail_heads).to(self.device)
        torch.manual_seed(config.experiment.seed + 90_000)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.experiment.seed + 90_000)
        if config.experiment.mode == "pixel_baseline":
            self.model.set_stage("rgb", torch.eye(config.model.token_dim))
            configure_trainability(
                self.model, 3, config.curriculum, pixel_baseline=True
            )
            self.stage_index = 3
        else:
            self.model.set_stage("dino12", self.projectors["dino12"])
            configure_trainability(self.model, 0, config.curriculum)
            self.stage_index = 0
        self.ema = EMA(self.model, config.optimizer.ema_decay)
        self.manager = CheckpointManager(
            Path(config.experiment.output_directory) / config.checkpoint.directory
        )
        self.completed_epoch = 0
        self.global_step = 0
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._pending_scheduler_state: dict[str, object] | None = None
        self.images_seen = 0
        self.estimated_flops = 0
        self.started_at = time.perf_counter()
        self.elapsed_before_resume = 0.0

    def _resume_state(self, transition_state: str, *, training_complete: bool) -> ResumeState:
        stage_epochs = self.config.curriculum.epochs[self.stage_index]
        stage_begin = sum(self.config.curriculum.epochs[: self.stage_index])
        within = min(max(self.completed_epoch - stage_begin, 0), stage_epochs)
        return ResumeState(
            schema_version=3,
            completed_epoch=self.completed_epoch,
            global_step=self.global_step,
            stage_index=self.stage_index,
            stage=self.model.stage,
            epoch_within_stage=within,
            transition_state=transition_state,
            training_complete=training_complete,
            images_seen=self.images_seen,
            estimated_flops=self.estimated_flops,
            wall_clock_seconds=self.elapsed_before_resume + time.perf_counter() - self.started_at,
            optimizer=self.optimizer.state_dict() if self.optimizer is not None else None,
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
        metadata = json.loads((path / "metadata.json").read_text())
        saved_config = load_checkpoint_config(path / "config.json")
        for name in (
            "experiment",
            "data",
            "encoder",
            "representations",
            "alignment",
            "model",
            "detail_heads",
            "curriculum",
            "objective",
            "optimizer",
        ):
            current_value = getattr(self.config, name)
            saved_value = getattr(saved_config, name)
            if name == "experiment":
                if current_value.mode != saved_value.mode:
                    raise ValueError("checkpoint experiment mode mismatch")
            elif current_value != saved_value:
                raise ValueError(f"checkpoint {name} configuration mismatch")
        configure_model_for_resume(self.model, metadata, self.projectors, self.config)
        self.ema = EMA(self.model, self.config.optimizer.ema_decay)
        self.optimizer = build_optimizer(
            self.model, self.config.optimizer, transitioned=int(metadata["stage_index"]) > 0
        )
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

    def _finalize_export(self) -> None:
        if self.config.experiment.mode != "progressive":
            return
        ema_model = self.ema.to_model(self.model)
        self.model.merge_lora()
        ema_model.merge_lora()
        self.model.fold_projector()
        ema_model.fold_projector()
        configure_trainability(
            self.model, 3, self.config.curriculum, pixel_baseline=True
        )
        configure_trainability(
            ema_model, 3, self.config.curriculum, pixel_baseline=True
        )
        self.ema.replace_from_model(ema_model)
        self.optimizer = build_optimizer(
            self.model, self.config.optimizer, transitioned=True
        )
        self.scheduler = None

    def _transition(self, target_index: int) -> None:
        if self.rank == 0 and self.config.checkpoint.save_boundary_before:
            self.manager.save_boundary(
                target_stage_index=target_index,
                position="before",
                model=self.model,
                ema=self.ema,
                state=self._resume_state("before", training_complete=False),
                config=self.config,
            )
        if self.world_size > 1:
            torch.distributed.barrier()
        transition_to_stage(
            self.model,
            self.ema,
            target_index,
            self.projectors,
            self.config.curriculum,
        )
        self.stage_index = target_index
        self.optimizer = build_optimizer(self.model, self.config.optimizer, transitioned=True)
        self.scheduler = None
        if self.rank == 0 and self.config.checkpoint.save_boundary_after:
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

    def run(self, *, max_steps: int | None = None) -> None:
        total_epochs = self.config.total_epochs
        for epoch in range(self.completed_epoch, total_epochs):
            if self.config.experiment.mode == "pixel_baseline":
                stage, stage_index, epoch_in_stage = "rgb", 3, epoch
                stage_epochs = total_epochs
            else:
                position = stage_at_epoch(epoch, self.config.curriculum)
                stage, stage_index = position.name, position.index
                epoch_in_stage = position.epoch_within_stage
                stage_epochs = self.config.curriculum.epochs[stage_index]
                if stage_index != self.stage_index:
                    self._transition(stage_index)
            loader = make_stage_loader(
                self.config,
                stage,
                rank=self.rank,
                world_size=self.world_size,
                epoch=epoch,
            )
            if self.scheduler is None:
                warmup = (
                    self.config.curriculum.transition_warmup_epochs
                    if stage_index > 0 and self.config.experiment.mode == "progressive"
                    else 0
                )
                self.scheduler = build_stage_scheduler(
                    self.optimizer,
                    steps_per_epoch=len(loader),
                    stage_epochs=stage_epochs,
                    warmup_epochs=warmup,
                    min_lr_ratio=self.config.optimizer.min_lr_ratio,
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
            execution.train()
            source_model = None
            if (
                self.config.experiment.mode == "progressive"
                and stage_index > 0
                and self.config.curriculum.source_consistency_weight
            ):
                source_model = self.ema.to_model(self.model).to(self.device).eval()
            for batch in loader:
                tokens = batch.tokens.to(self.device, non_blocking=True)
                labels = batch.labels.to(self.device, non_blocking=True)
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
                        self.rng,
                        self.config,
                        stage=stage,
                        source_model=source_model,
                    )
                result.loss.total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.requires_grad),
                    self.config.optimizer.gradient_clip,
                )
                self.optimizer.step()
                assert self.scheduler is not None
                self.scheduler.step()
                self.ema.update(self.model)
                self.global_step += 1
                self.images_seen += tokens.shape[0] * self.world_size
                self.estimated_flops += model_flops_per_step(self.config, stage=stage)
                self.log(
                    {
                        "stage": stage,
                        "epoch": epoch,
                        "epoch_within_stage": epoch_in_stage,
                        "step": self.global_step,
                        "loss": float(result.loss.total.detach()),
                        "velocity_mse": float(result.loss.velocity_mse.detach()),
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "gradient_norm": float(gradient_norm),
                    }
                )
                if max_steps is not None and self.global_step >= max_steps:
                    self._write_run_metrics(complete=False)
                    return
            self.completed_epoch = epoch + 1
            complete = self.completed_epoch == total_epochs
            validation_interval = self.config.validation.interval_epochs
            validate = complete or (
                validation_interval > 0
                and self.completed_epoch % validation_interval == 0
            )
            if validate:
                ema_model = self.ema.to_model(self.model).to(self.device)
                validation_loss, preview, preview_labels = validate_stage_loss(
                    ema_model,
                    loader,
                    self.config,
                    stage=stage,
                    epoch=epoch,
                    device=self.device,
                )
                self.log(
                    {
                        "stage": stage,
                        "epoch": epoch,
                        "step": self.global_step,
                        "validation_loss": validation_loss,
                    }
                )
                if self.rank == 0 and preview is not None and preview_labels is not None:
                    preview_directory = (
                        Path(self.config.experiment.output_directory) / "validation"
                    )
                    preview_directory.mkdir(parents=True, exist_ok=True)
                    features = preview.transpose(1, 2).reshape(-1, 768, 8, 8)
                    save_feature_pca_preview(
                        features,
                        preview_labels,
                        preview_directory / f"epoch-{self.completed_epoch:04d}-feature-pca.png",
                        128,
                    )
            if complete:
                self._finalize_export()
            interval = self.config.checkpoint.interval_epochs
            periodic = interval and self.completed_epoch % interval == 0
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
            "wall_clock_seconds": self.elapsed_before_resume + time.perf_counter() - self.started_at,
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
