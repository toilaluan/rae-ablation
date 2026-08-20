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

    def log_training(self, metrics: dict[str, Any], step: int) -> None:
        self.log(metrics, step=step)
        if not self.is_main_process or step % self.log_interval:
            return
        self.event(
            "train: "
            f"step={step} epoch={metrics['train/epoch']:.3f} "
            f"loss={metrics['train/loss']:.6f} "
            f"velocity={metrics['train/velocity_loss']:.6f} "
            f"repa={metrics['train/repa_loss']:.6f} "
            f"r={metrics['train/r_mean']:.3f} "
            f"lr={metrics['train/lr']:.3e} "
            f"grad_norm={metrics['train/gradient_norm']:.4f} "
            f"throughput={metrics['throughput/images_per_second']:.2f} images/s "
            f"step_time={metrics['throughput/step_seconds']:.3f}s "
            f"data_wait={metrics['throughput/data_wait_seconds']:.3f}s "
            f"peak_memory={metrics['system/peak_memory_gib']:.2f} GiB"
        )

    def log_decoder_training(self, metrics: dict[str, Any], step: int) -> None:
        self.log(metrics, step=step)
        if not self.is_main_process or step % self.log_interval:
            return
        self.event(
            "decoder: "
            f"step={step} loss={metrics['decoder/loss']:.6f} "
            f"l1={metrics['decoder/l1']:.6f} "
            f"lpips={metrics['decoder/lpips']:.6f} "
            f"g_adv={metrics['decoder/generator_adversarial']:.6f} "
            f"d={metrics['decoder/discriminator']:.6f} "
            f"gan={bool(metrics['decoder/gan_active'])} "
            f"lr={metrics['decoder/lr']:.3e} "
            f"throughput={metrics['throughput/images_per_second']:.2f} images/s "
            f"step_time={metrics['throughput/step_seconds']:.3f}s"
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
