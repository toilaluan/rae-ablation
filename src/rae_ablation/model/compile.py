from __future__ import annotations

from torch import nn

from ..config import CompileConfig


def compile_model_regions(model: nn.Module, config: CompileConfig) -> nn.Module:
    """Create a compiled execution view while keeping ``model`` canonical for state I/O."""
    if not config.enabled:
        return model
    from accelerate.utils import compile_regions

    return compile_regions(
        model,
        backend=config.backend,
        mode=config.mode,
        fullgraph=config.fullgraph,
        dynamic=config.dynamic,
    )


def clean_state_dict(model: nn.Module) -> dict[str, object]:
    state = model.state_dict()
    if any("_orig_mod" in key for key in state):
        raise ValueError("serialize the canonical uncompiled model, not its compiled view")
    return dict(state)
