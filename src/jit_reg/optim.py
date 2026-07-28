from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from .config import OptimizerConfig

TORCH_ADAMW = "torch.optim.AdamW"
TORCH_MUON = "torch.optim.Muon"
FUSED_MUON = "gram_newton_schulz.Muon"
FUSED_MUON_COEFFICIENTS = [(3.4445, -4.7750, 2.0315)] * 5


def _muon_eligible(name: str, parameter: nn.Parameter) -> bool:
    excluded = ("embedding", "norm", "bias", "projection", "input_projection")
    return (
        parameter.ndim == 2
        and "lora" not in name
        and not any(part in name for part in excluded)
    )


class CompositeOptimizer:
    def __init__(
        self,
        optimizers: Iterable[torch.optim.Optimizer],
        implementations: Iterable[str],
    ) -> None:
        self.optimizers = list(optimizers)
        self.implementations = list(implementations)
        if len(self.optimizers) != len(self.implementations):
            raise ValueError(
                "optimizer implementations must identify every optimizer"
            )
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizers": [
                optimizer.state_dict() for optimizer in self.optimizers
            ],
            "implementations": list(self.implementations),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        states = state["optimizers"]
        if len(states) != len(self.optimizers):
            raise ValueError("optimizer count changed across resume")
        saved_implementations = state.get("implementations")
        if saved_implementations is None:
            if FUSED_MUON in self.implementations:
                raise ValueError(
                    "legacy optimizer states cannot be resumed with fused_muon; "
                    "resume with optimizer.kind=muon or start a new fused_muon run"
                )
        elif list(saved_implementations) != self.implementations:
            raise ValueError(
                "optimizer backend changed across resume; PyTorch Muon and "
                "fused Muon optimizer states are not interchangeable"
            )
        for optimizer, optimizer_state in zip(self.optimizers, states, strict=True):
            optimizer.load_state_dict(optimizer_state)


def _cuda_version() -> tuple[int, int] | None:
    version = torch.version.cuda
    if version is None:
        return None
    try:
        major, minor, *_rest = version.split(".")
        return int(major), int(minor)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "optimizer.kind=fused_muon could not determine the CUDA runtime "
            f"version from torch.version.cuda={version!r}"
        ) from error


def _build_fused_muon(
    parameters: list[nn.Parameter],
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "optimizer.kind=fused_muon requires an NVIDIA Hopper or Blackwell "
            "GPU; no CUDA device is available"
        )
    cuda_version = _cuda_version()
    if cuda_version is None or cuda_version < (12, 9):
        detected = torch.version.cuda or "none"
        raise RuntimeError(
            "optimizer.kind=fused_muon requires CUDA 12.9 or newer "
            f"(detected {detected})"
        )
    capability = torch.cuda.get_device_capability()
    if capability < (9, 0):
        raise RuntimeError(
            "optimizer.kind=fused_muon requires compute capability 9.0 or "
            f"newer (detected {capability[0]}.{capability[1]})"
        )
    if any(parameter.device.type != "cuda" for parameter in parameters):
        raise RuntimeError(
            "optimizer.kind=fused_muon requires all Muon parameters to be on CUDA"
        )
    try:
        module = importlib.import_module("gram_newton_schulz")
        fused_muon = module.Muon
    except (AttributeError, ImportError) as error:
        raise RuntimeError(
            "optimizer.kind=fused_muon requires gram-newton-schulz==0.1.6; "
            "install the fused-muon extra with "
            "`pip install 'jit-reg[fused-muon]' --no-build-isolation`"
        ) from error
    return fused_muon(
        params=parameters,
        lr=config.lr,
        weight_decay=config.weight_decay,
        momentum=0.95,
        nesterov=True,
        adjust_lr="rms_norm",
        ns_coefficients=FUSED_MUON_COEFFICIENTS,
        ns_algorithm="gram_newton_schulz",
        ns_use_kernels=True,
        ns_max_batch_size=None,
        gram_newton_schulz_num_restarts=1,
        scalar_optimizer=None,
    )


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> CompositeOptimizer:
    named = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if config.kind == "adamw":
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in named],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
        return CompositeOptimizer([optimizer], [TORCH_ADAMW])

    muon_parameters = [parameter for name, parameter in named if _muon_eligible(name, parameter)]
    adam_parameters = [
        parameter for name, parameter in named if not _muon_eligible(name, parameter)
    ]
    if config.kind == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "optimizer.kind=muon requires torch.optim.Muon (PyTorch 2.12+)"
            )
        muon = torch.optim.Muon(
            muon_parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
            adjust_lr_fn="match_rms_adamw",
        )
        muon_implementation = TORCH_MUON
    else:
        muon = _build_fused_muon(muon_parameters, config)
        muon_implementation = FUSED_MUON
    adam = torch.optim.AdamW(
        adam_parameters,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    return CompositeOptimizer(
        [muon, adam],
        [muon_implementation, TORCH_ADAMW],
    )


def learning_rate_at_epoch(epoch: float, base_lr: float = 2e-4, min_lr: float = 2e-5) -> float:
    if epoch <= 25:
        return base_lr
    if epoch < 50:
        return base_lr + (min_lr - base_lr) * ((epoch - 25) / 25)
    return min_lr


def set_learning_rate(optimizer: CompositeOptimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value
