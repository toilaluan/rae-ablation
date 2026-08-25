from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from .checkpoint import EMA, CheckpointManager, ResumeState
from .config import (
    IMAGE_SIZE,
    TOKEN_COUNT,
    TOKEN_DIM,
    ExperimentConfig,
    load_checkpoint_config,
)
from .curriculum import configure_trainability, stage_at_epoch
from .data import (
    MIXTURE_MODE_NAMES,
    RepresentationBatchPreprocessor,
    data_fingerprint,
    make_stage_loader,
)
from .distributed import broadcast_string
from .evaluation import evaluate_npz, save_evaluation
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
from .preparation import (
    preparation_fingerprint,
    representation_manifest_fingerprint,
    statistics_from_manifest,
)
from .representations import RepresentationManifest, unpatchify_rgb
from .rng import RNGStreams
from .sample_runtime import (
    SamplingParameters,
    noise_for_seeds,
    prune_sample_archive,
    sample_to_archive,
    state_dict_sha256,
)
from .sampling import sample_tokens
from .training import compute_training_loss
from .validation import ValidationPlan, create_validation_plan, validate_stage_loss

ImageLog = Callable[[str, str, int, str], None]


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
        log_image: ImageLog | None = None,
    ) -> None:
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.log = log or (lambda _metrics: None)
        self.log_image = log_image or (lambda _path, _caption, _step, _key: None)
        manifest, self.representation_fingerprint = load_representation_manifest(config)
        self.rgb_statistics = statistics_from_manifest(manifest, config.stage_names[-1])
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
        self.validation_plan = self._initialize_validation_plan()

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

    def _cached_pixel_evaluation(
        self,
        output: Path,
        sample_count: int,
        checkpoint_hash: str,
        parameters: SamplingParameters,
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
                        or manifest.get("num_samples") != sample_count
                        or manifest.get("checkpoint_sha256") != checkpoint_hash
                        or manifest.get("parameters") != vars(parameters)
                    ):
                        raise ValueError("cached pixel evaluation does not match this validation")
                    metrics = evaluation.get("metrics")
                    if not isinstance(metrics, dict):
                        raise ValueError("cached pixel evaluation metrics are invalid")
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

    def _generate_pixel_evaluation(
        self,
        model: nn.Module,
        *,
        completed_epoch: int,
        training_complete: bool,
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
        cached = self._cached_pixel_evaluation(output, sample_count, digest, parameters)
        if cached is not None:
            return None, output, sample_count, cached
        labels, _sample_ids, seeds = self.validation_plan.prefix(sample_count)
        execution = compile_model_regions(model, self.config.compile)
        execution_holder = {"model": execution}
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        @torch.inference_mode()
        def generate(batch_labels: torch.Tensor, batch_seeds: torch.Tensor) -> torch.Tensor:
            noise = noise_for_seeds(
                batch_seeds,
                (TOKEN_COUNT, TOKEN_DIM),
                self.device,
                dtype=dtype,
            )
            weights = torch.zeros(
                batch_labels.shape[0],
                self.config.num_representation_levels,
                device=self.device,
            )
            weights[:, -1] = 1
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                normalized = sample_tokens(
                    execution_holder["model"],
                    noise,
                    batch_labels.to(self.device),
                    weights,
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
        execution_holder.clear()
        del execution
        return archive, output, sample_count, None

    def _evaluate_pixel_archive(
        self,
        archive: Path | None,
        output: Path,
        sample_count: int,
        cached: dict[str, float] | None,
    ) -> dict[str, float]:
        payload: list[object | None] = [cached, None]
        if self.rank == 0 and cached is None:
            try:
                if archive is None:
                    raise RuntimeError("new pixel evaluation has no sample archive")
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
            raise RuntimeError("pixel evaluation did not produce metrics")
        return {str(name): float(value) for name, value in payload[0].items()}

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
                        timestep_shift=self.config.stages[stage_index].timestep_shift,
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
                    "timestep_shift": self.config.stages[stage_index].timestep_shift,
                }
                for mode, name in enumerate(MIXTURE_MODE_NAMES):
                    metrics[f"mode/{name}"] = int((modes == mode).sum())
                self.log(metrics)
                if max_steps is not None and self.global_step >= max_steps:
                    self._write_run_metrics(complete=False)
                    return
            del source_model
            self.completed_epoch = epoch + 1
            complete = self.completed_epoch == self.config.total_epochs
            interval = self.config.validation.interval_epochs
            final_evaluation: tuple[dict[str, float], int] | None = None
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
                    pca_path = (
                        preview_directory
                        / f"epoch-{self.completed_epoch:04d}-feature-pca.png"
                    )
                    save_feature_pca_preview(
                        features,
                        preview_labels,
                        pca_path,
                        128,
                    )
                    self.log_image(
                        str(pca_path),
                        f"stage={stage_name}, epoch={self.completed_epoch}",
                        self.global_step,
                        "validation/feature_pca",
                    )
                archive, evaluation_output, sample_count, cached = (
                    self._generate_pixel_evaluation(
                        ema_model,
                        completed_epoch=self.completed_epoch,
                        training_complete=complete,
                    )
                )
                del ema_model
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                pixel_metrics = self._evaluate_pixel_archive(
                    archive, evaluation_output, sample_count, cached
                )
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                if self.rank == 0:
                    qualified = {
                        f"validation/pixel_{name.lower()}_{sample_count}": value
                        for name, value in pixel_metrics.items()
                    }
                    self.log(
                        {
                            "stage": stage_name,
                            "epoch": epoch,
                            "step": self.global_step,
                            "validation/pixel_sample_count": sample_count,
                        }
                        | qualified
                    )
                    self.log_image(
                        str(evaluation_output / "preview.png"),
                        (
                            f"RGB EMA samples, stage={stage_name}, "
                            f"epoch={self.completed_epoch}, n={sample_count}"
                        ),
                        self.global_step,
                        "validation/pixel_samples",
                    )
                if complete:
                    final_evaluation = (pixel_metrics, sample_count)
            checkpoint_interval = self.config.checkpoint.interval_epochs
            periodic = (
                checkpoint_interval
                and self.completed_epoch % checkpoint_interval == 0
            )
            if self.rank == 0 and (complete or periodic):
                checkpoint_path = self.manager.save(
                    model=self.model,
                    ema=self.ema,
                    state=self._resume_state("stable", training_complete=complete),
                    config=self.config,
                )
                if complete and final_evaluation is not None:
                    metrics, sample_count = final_evaluation
                    save_evaluation(
                        checkpoint_path / "evaluation.json",
                        metrics,
                        self.validation_plan.plan_hash,
                        num_samples=sample_count,
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
