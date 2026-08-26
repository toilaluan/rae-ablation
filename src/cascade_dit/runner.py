from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from .checkpoint import (
    EMA,
    TRAINING_STATE_SCHEMA_VERSION,
    CheckpointManager,
    ResumeState,
)
from .config import IMAGE_SIZE, TOKEN_COUNT, TOKEN_DIM, ExperimentConfig, load_checkpoint_config
from .data import (
    EndpointBatchPreprocessor,
    EndpointPhase,
    data_fingerprint,
    load_endpoint_manifest,
    make_endpoint_loader,
    set_endpoint_loader_epoch,
)
from .distributed import broadcast_string
from .evaluation import evaluate_npz, save_evaluation
from .flops import model_flops_per_step
from .model import DDT, build_model
from .model.compile import compile_model_regions
from .optim import (
    build_optimizer,
    build_training_scheduler,
    parameter_count,
    trainable_parameter_count,
)
from .preparation import (
    representation_manifest_fingerprint,
    statistics_from_manifest,
    validate_manifest_compatibility,
)
from .representations import ChannelStatistics, RepresentationManifest, unpatchify_rgb
from .rng import RNGStreams
from .sample_runtime import (
    SamplingParameters,
    noise_for_seeds,
    prune_sample_archive,
    sample_to_archive,
    state_dict_sha256,
)
from .sampling import sample_semantic_then_pixel, sample_tokens
from .training import compute_training_loss
from .validation import ValidationPlan, create_validation_plan, validate_phase_loss

ImageLog = Callable[[str, str, int, str], None]


def load_representation_manifest(
    config: ExperimentConfig,
) -> tuple[RepresentationManifest, str]:
    manifest = load_endpoint_manifest(
        config, load_semantic=config.training.mode == "progressive"
    )
    if config.training.mode == "progressive":
        validate_manifest_compatibility(config, manifest)
    return manifest, representation_manifest_fingerprint(manifest)


def _resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    if path.name == "last" or not (path / "metadata.json").is_file():
        if (path / "last").is_dir():
            path = (path / "last").resolve()
        elif path.name == "last":
            path = path.resolve()
    return path.resolve()


def _different_config_fields(active: object, saved: object, prefix: str) -> list[str]:
    if isinstance(active, dict) and isinstance(saved, dict):
        differences: list[str] = []
        for name in sorted(set(active) | set(saved)):
            child = f"{prefix}.{name}" if prefix else name
            differences.extend(
                _different_config_fields(active.get(name), saved.get(name), child)
            )
        return differences
    if isinstance(active, list) and isinstance(saved, list):
        return [] if active == saved else [prefix]
    return [] if active == saved else [prefix]


class ExperimentRunner:
    """Progressive semantic-to-pixel or direct-pixel DDT training."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device | str = "cpu",
        log: Callable[[dict[str, float | int | str]], None] | None = None,
        log_image: ImageLog | None = None,
        resume: str | Path | None = None,
    ) -> None:
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.log = log or (lambda _metrics: None)
        self.log_image = log_image or (lambda _path, _caption, _step, _key: None)
        manifest, self.representation_fingerprint = load_representation_manifest(config)
        self.rgb_statistics: ChannelStatistics = statistics_from_manifest(
            manifest, "rgb"
        )
        self.data_fingerprint = data_fingerprint(config.data)
        self.batch_preprocessor = EndpointBatchPreprocessor(
            config,
            manifest,
            self.device,
            load_semantic=config.training.mode == "progressive",
        )
        self.rng = RNGStreams.create(config.experiment.seed, self.device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.rng.initialization.initial_seed())
            self.model: DDT = build_model(config.model).to(self.device)
        torch.manual_seed(config.experiment.seed + 90_000)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.experiment.seed + 90_000)
        self.ema = EMA(self.model, config.optimizer.ema_decay)
        self.manager = CheckpointManager(
            Path(config.experiment.output_directory) / config.checkpoint.directory
        )
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._pending_scheduler_state: dict[str, object] | None = None
        self.completed_epoch = 0
        self.global_step = 0
        self.images_seen = 0
        self.estimated_flops = 0
        self.started_at = time.perf_counter()
        self.elapsed_before_resume = 0.0
        if resume is not None:
            self.resume(_resolve_checkpoint_path(resume))
        self.validation_plan = self._initialize_validation_plan()

    @property
    def phase(self) -> EndpointPhase:
        if self.completed_epoch >= self.config.total_epochs:
            return "pixel"
        return self.config.phase_for_epoch(self.completed_epoch)[0]

    @property
    def epoch_within_phase(self) -> int:
        if self.completed_epoch >= self.config.total_epochs:
            return self.config.training.pixel.epochs
        return self.config.phase_for_epoch(self.completed_epoch)[1]

    def _initialize_validation_plan(self) -> ValidationPlan:
        plan = create_validation_plan(
            final_samples=self.config.evaluation.num_samples,
            periodic_samples=self.config.validation.periodic_samples,
            num_classes=self.config.model.num_classes,
            seed=self.config.sampling.seed,
        )
        error: list[str | None] = [None]
        if self.rank == 0:
            try:
                plan.save(
                    Path(self.config.experiment.output_directory)
                    / "validation"
                    / "sampling-plan.pt"
                )
            except Exception as exception:
                error[0] = f"{type(exception).__name__}: {exception}"
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(error[0])
        return plan

    def _sampling_parameters(self) -> SamplingParameters:
        values = self.config.sampling
        return SamplingParameters(
            steps=values.steps,
            solver=values.solver,
            timestep_shift=values.timestep_shift,
            guidance_scale=values.guidance_scale,
            guidance_start=values.guidance_start,
            guidance_end=values.guidance_end,
            batch_size=values.batch_size,
            seed=values.seed,
        )

    def _cached_evaluation(
        self,
        output: Path,
        sample_count: int,
        checkpoint_hash: str,
        parameters: SamplingParameters,
        phase: EndpointPhase,
    ) -> dict[str, float] | None:
        payload: list[object | None] = [None, None]
        if self.rank == 0:
            try:
                evaluation_path = output / "evaluation.json"
                preview_path = output / "preview.png"
                manifest_path = output / "samples.json"
                if evaluation_path.is_file() and preview_path.is_file() and manifest_path.is_file():
                    evaluation = json.loads(evaluation_path.read_text())
                    manifest = json.loads(manifest_path.read_text())
                    if (
                        evaluation.get("sampling_plan_hash") != self.validation_plan.plan_hash
                        or evaluation.get("num_samples") != sample_count
                        or evaluation.get("phase") != phase
                        or manifest.get("num_samples") != sample_count
                        or manifest.get("checkpoint_sha256") != checkpoint_hash
                        or manifest.get("parameters") != vars(parameters)
                    ):
                        raise ValueError("cached evaluation does not match this validation")
                    metrics = evaluation.get("metrics")
                    if not isinstance(metrics, dict):
                        raise ValueError("cached evaluation metrics are invalid")
                    if manifest.get("archive_retained", True):
                        prune_sample_archive(output)
                    payload[0] = {str(name): float(value) for name, value in metrics.items()}
            except Exception as exception:
                payload[1] = f"{type(exception).__name__}: {exception}"
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(payload, src=0)
        if payload[1] is not None:
            raise RuntimeError(str(payload[1]))
        return payload[0] if isinstance(payload[0], dict) else None

    def _generate_evaluation(
        self,
        model: nn.Module,
        *,
        completed_epoch: int,
        training_complete: bool,
        phase: EndpointPhase,
    ) -> tuple[Path | None, Path, int, dict[str, float] | None]:
        sample_count = (
            self.config.evaluation.num_samples
            if training_complete
            else self.config.validation.periodic_samples
        )
        output = (
            Path(self.config.experiment.output_directory)
            / "validation"
            / f"epoch-{completed_epoch:04d}"
        )
        parameters = self._sampling_parameters()
        digest = state_dict_sha256(self.ema.state_dict()) if self.rank == 0 else None
        digest = broadcast_string(digest, self.rank)
        cached = self._cached_evaluation(
            output, sample_count, digest, parameters, phase
        )
        if cached is not None:
            return None, output, sample_count, cached
        labels, seeds = self.validation_plan.prefix(sample_count)
        execution = compile_model_regions(model, self.config.compile)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        @torch.inference_mode()
        def generate(batch_labels: torch.Tensor, batch_seeds: torch.Tensor) -> torch.Tensor:
            noise = noise_for_seeds(
                batch_seeds,
                (TOKEN_COUNT, TOKEN_DIM),
                self.device,
                dtype=dtype,
            )
            device_labels = batch_labels.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                if phase == "semantic":
                    normalized = sample_semantic_then_pixel(
                        execution,
                        noise,
                        device_labels,
                        steps=parameters.steps,
                        solver=parameters.solver,
                        timestep_shift=parameters.timestep_shift,
                        guidance_scale=parameters.guidance_scale,
                        guidance_start=parameters.guidance_start,
                        guidance_end=parameters.guidance_end,
                        denominator_floor=self.config.objective.denominator_floor,
                        num_classes=self.config.model.num_classes,
                    )
                else:
                    ratio = torch.ones(
                        batch_labels.shape[0], device=self.device, dtype=torch.float32
                    )
                    normalized = sample_tokens(
                        execution,
                        noise,
                        device_labels,
                        ratio,
                        prediction_mode="pixel",
                        steps=parameters.steps,
                        solver=parameters.solver,
                        timestep_shift=parameters.timestep_shift,
                        guidance_scale=parameters.guidance_scale,
                        guidance_start=parameters.guidance_start,
                        guidance_end=parameters.guidance_end,
                        denominator_floor=self.config.objective.denominator_floor,
                        num_classes=self.config.model.num_classes,
                    )
            tokens = self.rgb_statistics.denormalize(normalized.float())
            return unpatchify_rgb(tokens).clamp(0, 1)

        archive = sample_to_archive(
            output_directory=output,
            num_samples=sample_count,
            num_classes=self.config.model.num_classes,
            image_size=IMAGE_SIZE,
            checkpoint_hash=digest,
            parameters=parameters,
            generate_batch=generate,
            labels=labels,
            seeds=seeds,
            plan_hash=self.validation_plan.plan_hash,
        )
        return archive, output, sample_count, None

    def _evaluate_archive(
        self,
        archive: Path | None,
        output: Path,
        sample_count: int,
        cached: dict[str, float] | None,
        phase: EndpointPhase,
    ) -> dict[str, float]:
        payload: list[object | None] = [cached, None]
        if self.rank == 0 and cached is None:
            try:
                if archive is None:
                    raise RuntimeError("new evaluation has no sample archive")
                metrics = evaluate_npz(
                    archive,
                    self.config.evaluation.reference_npz,
                    batch_size=self.config.evaluation.batch_size,
                    device=self.config.evaluation.device,
                )
                save_evaluation(
                    output / "evaluation.json",
                    metrics,
                    self.validation_plan.plan_hash,
                    num_samples=sample_count,
                    phase=phase,
                )
                prune_sample_archive(output)
                payload[0] = metrics
            except Exception as exception:
                payload[1] = f"{type(exception).__name__}: {exception}"
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(payload, src=0)
        if payload[1] is not None:
            raise RuntimeError(str(payload[1]))
        if not isinstance(payload[0], dict):
            raise RuntimeError("evaluation did not produce metrics")
        return {str(name): float(value) for name, value in payload[0].items()}

    def _resume_state(self, *, training_complete: bool) -> ResumeState:
        return ResumeState(
            schema_version=TRAINING_STATE_SCHEMA_VERSION,
            completed_epoch=self.completed_epoch,
            global_step=self.global_step,
            phase=self.phase,
            epoch_within_phase=self.epoch_within_phase,
            training_complete=training_complete,
            images_seen=self.images_seen,
            estimated_flops=self.estimated_flops,
            wall_clock_seconds=(
                self.elapsed_before_resume + time.perf_counter() - self.started_at
            ),
            optimizer=self.optimizer.state_dict(),
            scheduler=None if self.scheduler is None else self.scheduler.state_dict(),
            rng_streams=self.rng.state_dict(),
            data_fingerprint=self.data_fingerprint,
            representation_fingerprint=self.representation_fingerprint,
        )

    def resume(self, checkpoint: str | Path) -> None:
        path = _resolve_checkpoint_path(checkpoint)
        saved_config = load_checkpoint_config(path / "config.json")
        compared_fields = (
            "data",
            "encoder",
            "training",
            "representations",
            "model",
            "objective",
            "optimizer",
        )
        active_values = self.config.as_dict()
        saved_values = saved_config.as_dict()
        differences: list[str] = []
        for name in compared_fields:
            differences.extend(
                _different_config_fields(active_values[name], saved_values[name], name)
            )
        if differences and self.rank == 0:
            warnings.warn(
                "active training configuration overrides checkpoint fields: "
                + ", ".join(differences),
                stacklevel=2,
            )
        state = self.manager.load(path, self.model, self.ema)
        if self.rank == 0 and state.data_fingerprint != self.data_fingerprint:
            warnings.warn(
                "active dataset differs from the checkpoint; active data will be used",
                stacklevel=2,
            )
        if self.rank == 0 and state.representation_fingerprint != self.representation_fingerprint:
            warnings.warn(
                "active representation manifest differs from the checkpoint",
                stacklevel=2,
            )
        self.completed_epoch = state.completed_epoch
        self.global_step = state.global_step
        self.images_seen = state.images_seen
        self.estimated_flops = state.estimated_flops
        self.elapsed_before_resume = state.wall_clock_seconds
        self.started_at = time.perf_counter()
        self.rng.load_state_dict(state.rng_streams)
        if state.phase != self.phase or state.epoch_within_phase != self.epoch_within_phase:
            raise ValueError("checkpoint phase does not match its completed epoch")
        if not differences and state.optimizer is not None:
            self.optimizer.load_state_dict(state.optimizer)
            self._pending_scheduler_state = state.scheduler
        elif self.rank == 0 and (state.optimizer is not None or state.scheduler is not None):
            warnings.warn(
                "checkpoint optimizer/scheduler state is incompatible and will be reset",
                stacklevel=2,
            )

    def _build_execution(self, steps_per_epoch: int, phase: EndpointPhase) -> nn.Module:
        if self.scheduler is None:
            scheduler = self.config.optimizer.scheduler
            self.scheduler = build_training_scheduler(
                self.optimizer,
                kind=scheduler.kind,
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
                find_unused_parameters=phase == "pixel",
            )
        return execution

    def run(self, *, max_steps: int | None = None) -> None:
        execution: nn.Module | None = None
        loader = None
        previous_phase: EndpointPhase | None = None
        for epoch in range(self.completed_epoch, self.config.total_epochs):
            phase, epoch_in_phase = self.config.phase_for_epoch(epoch)
            if previous_phase is not None and phase != previous_phase:
                execution = None
                if self.rank == 0 and self.config.checkpoint.save_phase_boundary:
                    self.manager.save_phase_boundary(
                        model=self.model,
                        ema=self.ema,
                        state=self._resume_state(training_complete=False),
                        config=self.config,
                    )
                if self.world_size > 1:
                    torch.distributed.barrier()
            if loader is None:
                loader = make_endpoint_loader(
                    self.config,
                    phase,
                    rank=self.rank,
                    world_size=self.world_size,
                    epoch=epoch_in_phase,
                )
            else:
                set_endpoint_loader_epoch(loader, phase, epoch_in_phase)
            if execution is None:
                execution = self._build_execution(len(loader), phase)
            execution.train()
            for raw_batch in loader:
                batch = self.batch_preprocessor(raw_batch)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    result = compute_training_loss(
                        execution,
                        batch,
                        self.rng,
                        self.config,
                        phase=phase,
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
                self.images_seen += batch.clean.shape[0] * self.world_size
                self.estimated_flops += model_flops_per_step(self.config, phase)
                ratios = batch.pixel_ratio.detach().float()
                metrics: dict[str, float | int | str] = {
                    "phase": phase,
                    "epoch": epoch,
                    "epoch_within_phase": epoch_in_phase,
                    "step": self.global_step,
                    "loss": float(result.loss.total.detach()),
                    "velocity_mse": float(result.loss.velocity_mse.detach()),
                    "pixel_auxiliary_mse": float(
                        result.loss.pixel_auxiliary_mse.detach()
                    ),
                    "weighted_pixel_auxiliary_mse": float(
                        result.loss.weighted_pixel_auxiliary_mse.detach()
                    ),
                    "pure_semantic_count": result.loss.pure_semantic_count,
                    "mixed_count": int(((ratios > 0) & (ratios < 1)).sum()),
                    "pixel_ratio_mean": float(ratios.mean()),
                    "pixel_ratio_min": float(ratios.min()),
                    "pixel_ratio_max": float(ratios.max()),
                    "target_rms": float(batch.clean.float().square().mean().sqrt()),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(gradient_norm),
                }
                self.log(metrics)
                if max_steps is not None and self.global_step >= max_steps:
                    self._write_run_metrics(complete=False)
                    return
            self.completed_epoch = epoch + 1
            complete = self.completed_epoch == self.config.total_epochs
            interval = self.config.validation.interval_epochs
            final_evaluation: tuple[dict[str, float], int] | None = None
            if complete or (interval > 0 and self.completed_epoch % interval == 0):
                ema_model = self.ema.to_model(self.model).to(self.device)
                validation = validate_phase_loss(
                    ema_model,
                    loader,
                    self.batch_preprocessor,
                    self.config,
                    phase=phase,
                    epoch=epoch,
                    device=self.device,
                )
                self.log(
                    {
                        "phase": phase,
                        "epoch": epoch,
                        "step": self.global_step,
                    }
                    | {f"validation/{name}": value for name, value in validation.items()}
                )
                archive, evaluation_output, sample_count, cached = (
                    self._generate_evaluation(
                        ema_model,
                        completed_epoch=self.completed_epoch,
                        training_complete=complete,
                        phase=phase,
                    )
                )
                del ema_model
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                generated_metrics = self._evaluate_archive(
                    archive, evaluation_output, sample_count, cached, phase
                )
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                if self.rank == 0:
                    self.log(
                        {
                            "phase": phase,
                            "epoch": epoch,
                            "step": self.global_step,
                            "validation/pixel_sample_count": sample_count,
                        }
                        | {
                            f"validation/pixel_{name.lower()}_{sample_count}": value
                            for name, value in generated_metrics.items()
                        }
                    )
                    self.log_image(
                        str(evaluation_output / "preview.png"),
                        f"RGB EMA samples, phase={phase}, epoch={self.completed_epoch}",
                        self.global_step,
                        "validation/pixel_samples",
                    )
                if complete:
                    final_evaluation = (generated_metrics, sample_count)
            checkpoint_interval = self.config.checkpoint.interval_epochs
            periodic = checkpoint_interval and self.completed_epoch % checkpoint_interval == 0
            if self.rank == 0 and (complete or periodic):
                checkpoint_path = self.manager.save(
                    model=self.model,
                    ema=self.ema,
                    state=self._resume_state(training_complete=complete),
                    config=self.config,
                )
                if complete and final_evaluation is not None:
                    metrics, sample_count = final_evaluation
                    save_evaluation(
                        checkpoint_path / "evaluation.json",
                        metrics,
                        self.validation_plan.plan_hash,
                        num_samples=sample_count,
                        phase=phase,
                    )
            if self.world_size > 1:
                torch.distributed.barrier()
            previous_phase = phase
        self._write_run_metrics(complete=True)

    def _write_run_metrics(self, *, complete: bool) -> None:
        if self.rank != 0:
            return
        output = Path(self.config.experiment.output_directory)
        output.mkdir(parents=True, exist_ok=True)
        metrics = {
            "complete": complete,
            "mode": self.config.training.mode,
            "phase": self.phase,
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
        (output / "training-metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
