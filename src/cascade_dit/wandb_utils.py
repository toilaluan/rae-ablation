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

    def should_log_training(self, step: int, *, force: bool = False) -> bool:
        return force or step % self.log_interval == 0

    def log_training(
        self, metrics: dict[str, Any], step: int, *, force: bool = False
    ) -> None:
        if not self.should_log_training(step, force=force):
            return
        self.log(metrics, step=step)
        if not self.is_main_process:
            return
        image_metric = (
            "train/image_target_velocity_loss"
            if "train/image_target_velocity_loss" in metrics
            else "train/image_latent_velocity_loss"
        )
        self.event(
            "train: "
            f"step={step} epoch={self._metric(metrics, 'train/epoch', '.3f')} "
            f"loss={self._metric(metrics, 'train/loss', '.6f')} "
            f"image={self._metric(metrics, image_metric, '.6f')} "
            f"semantic_full={self._metric(metrics, 'train/semantic_full_velocity_loss', '.6f')} "
            f"semantic_base={self._metric(metrics, 'train/semantic_base_velocity_loss', '.6f')} "
            f"predicted_condition={self._metric(metrics, 'train/predicted_semantic_condition_fraction', '.3f')} "
            f"lr={self._metric(metrics, 'train/lr', '.3e')} "
            f"grad_norm={self._metric(metrics, 'train/gradient_norm', '.4f')}"
        )

    def log_grid(
        self, path: str, caption: str, step: int, key: str = "validation/sample_preview"
    ) -> None:
        if self.run is not None:
            import wandb

            self.run.log({key: wandb.Image(path, caption=caption)}, step=step)

    def log_checkpoint(self, path: str, name: str) -> None:
        if self.run is not None:
            import wandb

            artifact = wandb.Artifact(name=name, type="checkpoint")
            artifact.add_dir(path)
            self.run.log_artifact(artifact)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
