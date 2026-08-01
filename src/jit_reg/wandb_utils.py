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
            f"image={metrics['train/image_velocity_loss']:.6f} "
            f"repa={metrics['train/repa_loss']:.6f} "
            f"K={metrics['exploration/k']} "
            f"selection_gain={metrics['exploration/score_improvement']:.6f} "
            f"lr={metrics['train/lr']:.3e} "
            f"grad_norm={metrics['train/gradient_norm']:.4f} "
            f"throughput={metrics['throughput/images_per_second']:.2f} images/s "
            f"step_time={metrics['throughput/step_seconds']:.3f}s "
            f"data_wait={metrics['throughput/data_wait_seconds']:.3f}s "
            f"host_batch={metrics['throughput/host_batch_seconds']:.3f}s "
            f"gpu_input={metrics['throughput/gpu_input_seconds']:.3f}s "
            f"gpu_fwd_bwd={metrics['throughput/gpu_forward_backward_seconds']:.3f}s "
            f"optim_host={metrics['throughput/optimizer_host_seconds']:.3f}s "
            f"gpu_optim={metrics['throughput/gpu_optimizer_seconds']:.3f}s "
            f"logging={metrics['throughput/logging_seconds']:.3f}s "
            f"peak_memory={metrics['system/peak_memory_gib']:.2f} GiB"
        )

    def log_grid(self, path: str, caption: str, step: int) -> None:
        if self.run is not None:
            import wandb

            self.run.log(
                {"validation/sample_preview": wandb.Image(path, caption=caption)}, step=step
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
