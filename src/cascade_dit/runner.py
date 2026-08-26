from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open
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
    set_stage_loader_epoch,
)
from .distributed import broadcast_string
from .evaluation import evaluate_npz, save_evaluation
from .feature_pca import save_feature_pca_preview
from .flops import model_flops_per_step
from .model import DDT, apply_lora, build_model, has_lora, merged_state_dict
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
from .representations import RepresentationManifest, cached_stages_include, unpatchify_rgb
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
        required_stages=(
            None
            if config.pixel_training.use_representation_guiding
            else [config.pixel_training.name]
        ),
    )
    if config.pixel_training.use_representation_guiding:
        validate_manifest_compatibility(config, manifest)
    else:
        pixel = {
            "name": config.pixel_training.name,
            "source": {"kind": "rgb", "layer": None},
        }
        if not cached_stages_include(manifest.stages, [pixel]):
            raise ValueError("representation manifest does not include the configured RGB stage")
        if manifest.dataset_fingerprint != data_fingerprint(config.data, config.data.image_size):
            raise ValueError("representation manifest uses a different dataset")
    return manifest, representation_manifest_fingerprint(manifest)


def _resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    if path.name == "last" or not (path / "metadata.json").is_file():
        if (path / "last").is_dir():
            path = (path / "last").resolve()
        elif path.name == "last":
            path = path.resolve()
    return path.resolve()


def _checkpoint_representation_levels(path: Path) -> int:
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as tensors:
        name = "representation_embedder.embedding"
        if name not in set(tensors.keys()):
            raise ValueError("checkpoint has no representation embedding")
        shape = tensors.get_slice(name).get_shape()
    if len(shape) != 2 or shape[0] < 1:
        raise ValueError("checkpoint representation embedding has an invalid shape")
    return int(shape[0])


def _different_config_fields(
    active: object,
    saved: object,
    prefix: str,
) -> list[str]:
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
    """Representation-guided or pixel-only DDT training."""

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
        resume_path = _resolve_checkpoint_path(resume) if resume is not None else None
        pixel_only = not config.pixel_training.use_representation_guiding
        self.representation_levels = (
            _checkpoint_representation_levels(resume_path)
            if pixel_only and resume_path is not None
            else config.num_representation_levels
        )
        manifest, self.representation_fingerprint = load_representation_manifest(config)
        self.rgb_statistics = statistics_from_manifest(manifest, config.pixel_training.name)
        self.data_fingerprint = data_fingerprint(config.data)
        self.batch_preprocessor = RepresentationBatchPreprocessor(
            config,
            manifest,
            self.device,
            pixel_only=pixel_only,
            representation_levels=self.representation_levels,
        )
        self.rng = RNGStreams.create(config.experiment.seed, self.device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.rng.initialization.initial_seed())
            self.model: DDT = build_model(
                config.model, self.representation_levels
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
        self.pixel_completed_epoch = 0
        self.global_step = 0
        self.stage_index = (
            self.representation_levels - 1
            if pixel_only
            else 0
        )
        self.optimizer: torch.optim.AdamW | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._pending_scheduler_state: dict[str, object] | None = None
        self._pending_optimizer_state: dict[str, object] | None = None
        self.images_seen = 0
        self.estimated_flops = 0
        self.started_at = time.perf_counter()
        self.elapsed_before_resume = 0.0
        if resume_path is not None:
            self.resume(resume_path)
        if self._pixel_is_active() and config.pixel_training.lora.use_lora:
            self._activate_lora()
        else:
            configure_trainability(self.model)
        self.optimizer = build_optimizer(self.model, config.optimizer)
        if self._pending_optimizer_state is not None:
            self.optimizer.load_state_dict(self._pending_optimizer_state)
            self._pending_optimizer_state = None
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
                getattr(
                    self,
                    "representation_levels",
                    self.config.num_representation_levels,
                ),
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
        if self.stage_index == self.representation_levels - 1:
            return self.config.pixel_training.name
        return self.config.stage_names[self.stage_index]

    def _pixel_is_active(self) -> bool:
        return (
            not self.config.pixel_training.use_representation_guiding
            or self.stage_index == self.representation_levels - 1
        )

    def _activate_lora(self) -> None:
        if has_lora(self.model):
            return
        ema_state = merged_state_dict(self.model, self.ema.state_dict())
        apply_lora(self.model, self.config.pixel_training.lora)
        reparameterized = EMA(self.model, self.config.optimizer.ema_decay)
        reparameterized.load_merged_state_dict(self.model, ema_state)
        self.ema = reparameterized
        self.optimizer = None
        self.scheduler = None
        self._pending_optimizer_state = None
        self._pending_scheduler_state = None

    def _epoch_within_stage(self) -> int:
        if not self.config.pixel_training.use_representation_guiding:
            return self.pixel_completed_epoch
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
            optimizer=(
                None
                if has_lora(self.model) or self.optimizer is None
                else self.optimizer.state_dict()
            ),
            scheduler=(
                None
                if has_lora(self.model) or self.scheduler is None
                else self.scheduler.state_dict()
            ),
            rng_streams=self.rng.state_dict(),
            data_fingerprint=self.data_fingerprint,
            representation_fingerprint=self.representation_fingerprint,
        )

    def resume(self, checkpoint: str | Path) -> None:
        path = _resolve_checkpoint_path(checkpoint)
        config_matches = False
        config_path = path / "config.json"
        try:
            raw_config = json.loads(config_path.read_text())
            if not isinstance(raw_config, dict):
                raise TypeError("checkpoint config is not an object")
            if raw_config.get("schema_version") == self.config.schema_version:
                saved_config = load_checkpoint_config(config_path)
                fields = (
                    "data",
                    "encoder",
                    "pixel_training",
                    "model",
                    "curriculum",
                    "objective",
                    "optimizer",
                )
                active_values = self.config.as_dict()
                saved_values = saved_config.as_dict()
                differences = []
                for name in fields:
                    differences.extend(
                        _different_config_fields(
                            active_values[name], saved_values[name], name
                        )
                    )
                config_matches = not differences
                if differences and self.rank == 0:
                    warnings.warn(
                        "active training configuration overrides checkpoint fields: "
                        + ", ".join(differences),
                        stacklevel=2,
                    )
            elif self.rank == 0:
                warnings.warn(
                    "checkpoint configuration uses an older schema; active YAML settings "
                    "will be used and optimizer/scheduler state will be reset",
                    stacklevel=2,
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            if self.rank == 0:
                warnings.warn(
                    f"checkpoint configuration could not be compared ({error}); "
                    "active YAML settings will be used",
                    stacklevel=2,
                )
        try:
            metadata = json.loads((path / "metadata.json").read_text())
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            metadata = {}
        if (
            self.rank == 0
            and metadata.get("model_format") == "lora_merged"
            and self.config.pixel_training.lora.use_lora
        ):
            warnings.warn(
                "checkpoint contains merged LoRA weights; they will be used as the new "
                "base with fresh zero-delta adapters",
                stacklevel=2,
            )
        self.ema = EMA(self.model, self.config.optimizer.ema_decay)
        state = self.manager.load(
            path,
            self.model,
            self.ema,
        )
        if self.rank == 0 and state.data_fingerprint != self.data_fingerprint:
            warnings.warn(
                "active dataset differs from the checkpoint; active data will be used",
                stacklevel=2,
            )
        if self.rank == 0 and state.representation_fingerprint != self.representation_fingerprint:
            warnings.warn(
                "active representation manifest differs from the checkpoint; active RGB "
                "statistics and representations will be used",
                stacklevel=2,
            )
        self.completed_epoch = state.completed_epoch
        self.global_step = state.global_step
        self.images_seen = state.images_seen
        self.estimated_flops = state.estimated_flops
        self.elapsed_before_resume = state.wall_clock_seconds
        self.started_at = time.perf_counter()
        self.rng.load_state_dict(state.rng_streams)
        if self.config.pixel_training.use_representation_guiding:
            if self.completed_epoch < self.config.total_epochs:
                position = stage_at_epoch(self.completed_epoch, self.config)
                self.stage_index = position.index
                self.pixel_completed_epoch = (
                    position.epoch_within_stage
                    if position.index == self.config.num_representation_levels - 1
                    else 0
                )
            else:
                self.stage_index = self.config.num_representation_levels - 1
                self.pixel_completed_epoch = self.config.pixel_training.epochs
        else:
            self.stage_index = self.representation_levels - 1
            self.pixel_completed_epoch = (
                min(state.epoch_within_stage, self.config.pixel_training.epochs)
                if state.stage == self.config.pixel_training.name
                else 0
            )
        restorable = (
            config_matches
            and not self.config.pixel_training.lora.use_lora
            and state.optimizer is not None
        )
        if restorable:
            self._pending_optimizer_state = state.optimizer
            self._pending_scheduler_state = state.scheduler
        elif self.rank == 0 and (state.optimizer is not None or state.scheduler is not None):
            warnings.warn(
                "checkpoint optimizer/scheduler state is incompatible with active training "
                "and will be reset",
                stacklevel=2,
            )

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
        if self.optimizer is None:
            self.optimizer = build_optimizer(self.model, self.config.optimizer)
        if self.scheduler is None:
            scheduler = self.config.optimizer.scheduler
            reset_pixel_schedule = (
                has_lora(self.model)
                or (
                    not self.config.pixel_training.use_representation_guiding
                    and self._pending_scheduler_state is None
                )
            )
            scheduler_epochs = (
                max(self.config.pixel_training.epochs - self.pixel_completed_epoch, 1)
                if reset_pixel_schedule
                else self.config.total_epochs
            )
            self.scheduler = build_training_scheduler(
                self.optimizer,
                kind=scheduler.kind,
                steps_per_epoch=steps_per_epoch,
                total_epochs=scheduler_epochs,
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

    def _epochs(self):
        if self.config.pixel_training.use_representation_guiding:
            for epoch in range(self.completed_epoch, self.config.total_epochs):
                position = stage_at_epoch(epoch, self.config)
                yield epoch, position.index, position.name, position.epoch_within_stage
            return
        initial_global = self.completed_epoch
        initial_pixel = self.pixel_completed_epoch
        for pixel_epoch in range(initial_pixel, self.config.pixel_training.epochs):
            yield (
                initial_global + pixel_epoch - initial_pixel,
                self.representation_levels - 1,
                self.config.pixel_training.name,
                pixel_epoch,
            )

    def run(self, *, max_steps: int | None = None) -> None:
        execution: nn.Module | None = None
        loader = None
        pixel_only = not self.config.pixel_training.use_representation_guiding
        for epoch, stage_index, stage_name, epoch_in_stage in self._epochs():
            pixel_stage = stage_index == self.representation_levels - 1
            if not pixel_only and stage_index != self.stage_index:
                self._save_stage_boundary(stage_index)
                if pixel_stage and self.config.pixel_training.lora.use_lora:
                    self._activate_lora()
                    self.optimizer = build_optimizer(self.model, self.config.optimizer)
                    execution = None
            data_epoch = epoch_in_stage if pixel_only else epoch
            if loader is None:
                loader = make_stage_loader(
                    self.config,
                    stage_index,
                    rank=self.rank,
                    world_size=self.world_size,
                    epoch=data_epoch,
                    pixel_only=pixel_only,
                    representation_levels=self.representation_levels,
                )
            else:
                set_stage_loader_epoch(loader, stage_index, data_epoch)
            if execution is None:
                execution = self._build_execution(len(loader))
            execution.train()
            source_model = None
            if (
                self.config.pixel_training.use_representation_guiding
                and self.config.curriculum.source_consistency_weight
            ):
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
                assert self.optimizer is not None
                self.optimizer.zero_grad(set_to_none=True)
                timestep_shift = (
                    self.config.pixel_training.timestep_shift
                    if pixel_stage
                    else self.config.stages[stage_index].timestep_shift
                )
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
                        timestep_shift=timestep_shift,
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
                    "timestep_shift": timestep_shift,
                }
                for mode, name in enumerate(MIXTURE_MODE_NAMES):
                    metrics[f"mode/{name}"] = int((modes == mode).sum())
                self.log(metrics)
                if max_steps is not None and self.global_step >= max_steps:
                    self._write_run_metrics(complete=False)
                    return
            del source_model
            self.completed_epoch = epoch + 1
            if pixel_stage:
                self.pixel_completed_epoch = epoch_in_stage + 1
            complete = (
                self.completed_epoch == self.config.total_epochs
                if not pixel_only
                else self.pixel_completed_epoch == self.config.pixel_training.epochs
            )
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
                    timestep_shift=timestep_shift,
                    preview_features=not pixel_stage,
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
            "mode": (
                "representation_guided"
                if self.config.pixel_training.use_representation_guiding
                else "pixel_only"
            ),
            "use_representation_guiding": (
                self.config.pixel_training.use_representation_guiding
            ),
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
