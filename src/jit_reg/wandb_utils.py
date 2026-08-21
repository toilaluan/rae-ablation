from __future__ import annotations

from typing import Any

from .config import WandbConfig


class RunLogger:
    def __init__(
        self,
        config: WandbConfig,
        resolved_config: dict[str, Any],
        is_main_process: bool,
        run_id: str | None = None,
        log_interval: int = 10,
    ) -> None:
        if log_interval < 1:
            raise ValueError("training log interval must be positive")
        self.is_main_process = is_main_process
        self.log_interval = log_interval
        self.run: Any = None
        if not config.enabled or not is_main_process:
            return
        import wandb

        self.run = wandb.init(
            entity=config.entity,
            project=config.project,
            name=config.name,
            group=config.group,
            tags=config.tags,
            mode=config.mode,
            config=resolved_config,
            id=run_id,
            resume="allow" if run_id else None,
        )

    @property
    def run_id(self) -> str | None:
        return self.run.id if self.run is not None else None

    def event(self, message: str) -> None:
        if self.is_main_process:
            print(message, flush=True)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    @staticmethod
    def _metric(metrics: dict[str, Any], key: str, format_spec: str) -> str:
        """Format console metrics without allowing logging drift to stop training."""
        try:
            return format(metrics[key], format_spec)
        except (KeyError, TypeError, ValueError):
            return "n/a"

    def log_training(self, metrics: dict[str, Any], step: int) -> None:
        self.log(metrics, step=step)
        if not self.is_main_process or step % self.log_interval:
            return
        self.event(
            "train: "
            f"step={step} epoch={self._metric(metrics, 'train/epoch', '.3f')} "
            f"loss={self._metric(metrics, 'train/loss', '.6f')} "
            f"full_velocity={self._metric(metrics, 'train/full_velocity_loss', '.6f')} "
            f"base_semantic_velocity={self._metric(metrics, 'train/base_semantic_velocity_loss', '.6f')} "
            f"repa={self._metric(metrics, 'train/repa_loss', '.6f')} "
            f"repa_cos={self._metric(metrics, 'train/repa_cosine_similarity', '.3f')} "
            f"t={self._metric(metrics, 'train/timestep_mean', '.3f')} "
            f"semantic={self._metric(metrics, 'train/semantic_stage_fraction', '.3f')} "
            f"lr={self._metric(metrics, 'train/lr', '.3e')} "
            f"grad_norm={self._metric(metrics, 'train/gradient_norm', '.4f')} "
            f"throughput={self._metric(metrics, 'throughput/images_per_second', '.2f')} images/s "
            f"step_time={self._metric(metrics, 'throughput/step_seconds', '.3f')}s "
            f"data_wait={self._metric(metrics, 'throughput/data_wait_seconds', '.3f')}s "
            f"peak_memory={self._metric(metrics, 'system/peak_memory_gib', '.2f')} GiB"
        )

    def log_decoder_training(self, metrics: dict[str, Any], step: int) -> None:
        self.log(metrics, step=step)
        if not self.is_main_process or step % self.log_interval:
            return
        self.event(
            "decoder: "
            f"step={step} loss={self._metric(metrics, 'decoder/loss', '.6f')} "
            f"l1={self._metric(metrics, 'decoder/l1', '.6f')} "
            f"lpips={self._metric(metrics, 'decoder/lpips', '.6f')} "
            f"g_adv={self._metric(metrics, 'decoder/generator_adversarial', '.6f')} "
            f"d={self._metric(metrics, 'decoder/discriminator', '.6f')} "
            f"gan={self._metric(metrics, 'decoder/gan_active', '.0f')} "
            f"lr={self._metric(metrics, 'decoder/lr', '.3e')} "
            f"throughput={self._metric(metrics, 'throughput/images_per_second', '.2f')} images/s "
            f"step_time={self._metric(metrics, 'throughput/step_seconds', '.3f')}s"
        )

    def log_grid(
        self, path: str, caption: str, step: int, key: str = "validation/sample_preview"
    ) -> None:
        if self.run is not None:
            import wandb

            self.run.log(
                {key: wandb.Image(path, caption=caption)}, step=step
            )

    def log_checkpoint(self, path: str, name: str) -> None:
        if self.run is not None:
            import wandb

            artifact = wandb.Artifact(name=name, type="checkpoint")
            artifact.add_dir(path)
            self.run.log_artifact(artifact)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


# Kept as an import-compatible alias for callers outside the three script entry points.
WandbLogger = RunLogger
