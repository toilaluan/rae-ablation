from __future__ import annotations

from typing import Any

from .config import WandbConfig


class WandbLogger:
    def __init__(
        self,
        config: WandbConfig,
        resolved_config: dict[str, Any],
        is_main_process: bool,
        run_id: str | None = None,
    ) -> None:
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

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def log_grid(self, path: str, caption: str, step: int) -> None:
        if self.run is not None:
            import wandb

            self.run.log(
                {"validation/fixed_sample_grid": wandb.Image(path, caption=caption)}, step=step
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
