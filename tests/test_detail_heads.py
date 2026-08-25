from __future__ import annotations

import torch

from cascade_dit.model import PixelPatchDetailer, TokenDetailHead


def test_token_detail_head_is_pointwise_and_zero_residual_initialized() -> None:
    head = TokenDetailHead(context_dim=24)
    noisy = torch.randn(2, 64, 768, requires_grad=True)
    context = torch.randn(2, 64, 24, requires_grad=True)
    output = head(noisy, context)
    assert output.shape == noisy.shape
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert head.output_projection.weight.grad is not None


def test_pixel_detailer_progression_patch_independence_and_gradient_flow() -> None:
    detailer = PixelPatchDetailer(context_dim=16)
    assert detailer.resolutions == (16, 8, 4, 2, 1, 2, 4, 8, 16)
    assert detailer.channel_progression == (3, 64, 128, 256, 512)
    noisy = torch.randn(1, 64, 768, requires_grad=True)
    context = torch.randn(1, 64, 16, requires_grad=True)
    zero = detailer(noisy, context)
    assert zero.shape == noisy.shape
    assert torch.count_nonzero(zero) == 0
    torch.nn.init.normal_(detailer.output.weight, std=0.01)
    first = detailer(noisy, context)
    changed = noisy.detach().clone()
    changed[:, 0].add_(10)
    second = detailer(changed, context.detach())
    assert torch.equal(first[:, 1:], second[:, 1:])
    first[:, 0].sum().backward()
    assert noisy.grad is not None and torch.count_nonzero(noisy.grad[:, 0]) > 0
    assert torch.count_nonzero(noisy.grad[:, 1:]) == 0
    assert detailer.context_projection.weight.grad is not None
