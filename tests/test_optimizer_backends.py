from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import jit_reg.optim as optim_module
from jit_reg.config import OptimizerConfig
from jit_reg.optim import (
    FUSED_MUON,
    FUSED_MUON_COEFFICIENTS,
    TORCH_ADAMW,
    TORCH_MUON,
    CompositeOptimizer,
    _build_fused_muon,
    _muon_eligible,
    build_optimizer,
)


def _mock_supported_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.version, "cuda", "12.9")


def test_fused_muon_constructor_settings_and_lazy_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_supported_cuda(monkeypatch)
    captured: dict[str, Any] = {}

    class FakeMuon:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    imported: list[str] = []

    def fake_import(name: str) -> object:
        imported.append(name)
        return SimpleNamespace(Muon=FakeMuon)

    monkeypatch.setattr(optim_module.importlib, "import_module", fake_import)
    parameter = SimpleNamespace(device=torch.device("cuda"))
    config = OptimizerConfig(kind="fused_muon", lr=3e-4, weight_decay=0.1)
    optimizer = _build_fused_muon([parameter], config)  # type: ignore[list-item]
    assert isinstance(optimizer, FakeMuon)
    assert imported == ["gram_newton_schulz"]
    assert captured == {
        "params": [parameter],
        "lr": 3e-4,
        "weight_decay": 0.1,
        "momentum": 0.95,
        "nesterov": True,
        "adjust_lr": "rms_norm",
        "ns_coefficients": FUSED_MUON_COEFFICIENTS,
        "ns_algorithm": "gram_newton_schulz",
        "ns_use_kernels": True,
        "ns_max_batch_size": None,
        "gram_newton_schulz_num_restarts": 1,
        "scalar_optimizer": None,
    }


def test_fused_muon_is_not_imported_for_other_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = optim_module.importlib.import_module

    def reject_fused_import(name: str, package: str | None = None) -> Any:
        if name == "gram_newton_schulz":
            raise AssertionError("fused package was imported")
        return original_import(name, package)

    monkeypatch.setattr(
        optim_module.importlib, "import_module", reject_fused_import
    )
    build_optimizer(nn.Linear(4, 4), OptimizerConfig(kind="adamw"))
    build_optimizer(nn.Linear(4, 4), OptimizerConfig(kind="muon"))


def test_fused_muon_hardware_and_installation_errors_are_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = SimpleNamespace(device=torch.device("cuda"))
    config = OptimizerConfig(kind="fused_muon")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        _build_fused_muon([parameter], config)  # type: ignore[list-item]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    with pytest.raises(RuntimeError, match=r"CUDA 12\.9"):
        _build_fused_muon([parameter], config)  # type: ignore[list-item]

    monkeypatch.setattr(torch.version, "cuda", "12.9")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
    with pytest.raises(RuntimeError, match=r"compute capability 9\.0"):
        _build_fused_muon([parameter], config)  # type: ignore[list-item]

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(
        optim_module.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(
            ModuleNotFoundError("gram_newton_schulz")
        ),
    )
    with pytest.raises(RuntimeError, match=r"gram-newton-schulz==0\.1\.6"):
        _build_fused_muon([parameter], config)  # type: ignore[list-item]


def test_fused_muon_keeps_existing_parameter_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.RMSNorm(4))
    captured: list[nn.Parameter] = []

    def fake_fused(
        parameters: list[nn.Parameter],
        config: OptimizerConfig,
    ) -> torch.optim.Optimizer:
        del config
        captured.extend(parameters)
        return torch.optim.SGD(parameters, lr=1e-3)

    monkeypatch.setattr(optim_module, "_build_fused_muon", fake_fused)
    optimizer = build_optimizer(
        model, OptimizerConfig(kind="fused_muon", lr=1e-3)
    )
    expected = [
        parameter
        for name, parameter in model.named_parameters()
        if _muon_eligible(name, parameter)
    ]
    assert captured == expected
    assert optimizer.implementations == [FUSED_MUON, TORCH_ADAMW]
    routed = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(routed) == len(list(model.parameters()))
    assert {id(parameter) for parameter in routed} == {
        id(parameter) for parameter in model.parameters()
    }


def _composite(implementations: list[str]) -> CompositeOptimizer:
    parameters = [nn.Parameter(torch.randn(2, 2)) for _ in implementations]
    return CompositeOptimizer(
        [
            torch.optim.SGD([parameter], lr=0.1)
            for parameter in parameters
        ],
        implementations,
    )


def test_optimizer_state_records_backends_and_supports_compatible_resume() -> None:
    torch_muon = _composite([TORCH_MUON, TORCH_ADAMW])
    state = torch_muon.state_dict()
    assert state["implementations"] == [TORCH_MUON, TORCH_ADAMW]
    _composite([TORCH_MUON, TORCH_ADAMW]).load_state_dict(state)

    legacy = dict(state)
    legacy.pop("implementations")
    _composite([TORCH_MUON, TORCH_ADAMW]).load_state_dict(legacy)

    fused = _composite([FUSED_MUON, TORCH_ADAMW])
    _composite([FUSED_MUON, TORCH_ADAMW]).load_state_dict(fused.state_dict())


def test_optimizer_state_rejects_cross_backend_and_legacy_fused_resume() -> None:
    torch_state = _composite([TORCH_MUON, TORCH_ADAMW]).state_dict()
    with pytest.raises(ValueError, match="not interchangeable"):
        _composite([FUSED_MUON, TORCH_ADAMW]).load_state_dict(torch_state)

    legacy = dict(torch_state)
    legacy.pop("implementations")
    with pytest.raises(ValueError, match="legacy optimizer states"):
        _composite([FUSED_MUON, TORCH_ADAMW]).load_state_dict(legacy)


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fused_and_torch_muon_updates_are_numerically_aligned() -> None:
    package = pytest.importorskip("gram_newton_schulz")
    cuda_version = optim_module._cuda_version()
    if cuda_version is None or cuda_version < (12, 9):
        pytest.skip("CUDA 12.9+ is required")
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("compute capability 9.0+ is required")

    shapes = [(2048, 768), (2304, 768), (4608, 768)]
    initial = [torch.randn(shape, device="cuda") for shape in shapes]
    torch_parameters = [
        nn.Parameter(value.clone()) for value in initial
    ]
    fused_parameters = [
        nn.Parameter(value.clone()) for value in initial
    ]
    generator = torch.Generator(device="cuda").manual_seed(11)
    gradients = [
        torch.randn(shape, device="cuda", generator=generator)
        for shape in shapes
    ]
    for parameter, gradient in zip(
        torch_parameters, gradients, strict=True
    ):
        parameter.grad = gradient.clone()
    for parameter, gradient in zip(
        fused_parameters, gradients, strict=True
    ):
        parameter.grad = gradient.clone()

    torch_muon = torch.optim.Muon(
        torch_parameters,
        lr=2e-4,
        weight_decay=0,
        adjust_lr_fn="match_rms_adamw",
    )
    fused_muon = package.Muon(
        params=fused_parameters,
        lr=2e-4,
        weight_decay=0,
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
    torch_muon.step()
    fused_muon.step()
    for start, torch_parameter, fused_parameter in zip(
        initial, torch_parameters, fused_parameters, strict=True
    ):
        torch_update = torch_parameter.detach() - start
        fused_update = fused_parameter.detach() - start
        assert torch.isfinite(torch_update).all()
        assert torch.isfinite(fused_update).all()
        similarity = torch.nn.functional.cosine_similarity(
            torch_update.flatten(),
            fused_update.flatten(),
            dim=0,
        )
        norm_difference = (
            torch_update.norm() - fused_update.norm()
        ).abs() / torch_update.norm()
        assert similarity >= 0.999
        assert norm_difference <= 0.02
