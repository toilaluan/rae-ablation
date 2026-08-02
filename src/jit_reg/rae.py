from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import IMAGE_SIZE, RAE_CHANNELS, RAE_INPUT_SIZE, RAEConfig

RAE_REMOTE_CODE_SCHEMA = 1


class RAECodec(nn.Module):
    """Frozen RAEv2 base model with its injected 2x spatial compressor."""

    def __init__(
        self,
        model: nn.Module,
        config: RAEConfig,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.config = config
        self.device = torch.device(device)
        self.compute_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return RAE_CHANNELS, RAE_INPUT_SIZE, RAE_INPUT_SIZE

    def train(self, mode: bool = True) -> RAECodec:
        del mode
        super().train(False)
        return self

    @torch.inference_mode()
    def encode(self, pixels: Tensor) -> Tensor:
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"RAE images must have shape [B, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]")
        outputs: list[Tensor] = []
        for chunk in pixels.split(self.config.micro_batch_size):
            with torch.autocast(
                self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                outputs.append(self.model.encode(chunk.to(self.device, dtype=self.compute_dtype)))
        result = torch.cat(outputs)
        if tuple(result.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"RAE compressor returned {tuple(result.shape[1:])}, expected {self.latent_shape}"
            )
        return result

    @torch.inference_mode()
    def decode(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 4 or tuple(tokens.shape[1:]) != self.latent_shape:
            raise ValueError(f"RAE tokens must have shape [B, {self.latent_shape}]")
        outputs: list[Tensor] = []
        for chunk in tokens.split(self.config.micro_batch_size):
            with torch.autocast(
                self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                outputs.append(self.model.decode(chunk.to(self.device, dtype=self.compute_dtype)))
        result = torch.cat(outputs).float()
        expected = (tokens.shape[0], 3, IMAGE_SIZE, IMAGE_SIZE)
        if tuple(result.shape) != expected:
            raise ValueError(f"RAE decoded {tuple(result.shape)}, expected {expected}")
        return result


def load_rae_codec(config: RAEConfig, device: torch.device | str) -> RAECodec:
    if config.remote_code_schema != RAE_REMOTE_CODE_SCHEMA:
        raise ValueError("unsupported RAE remote-code schema")
    config.validate()
    from transformers import AutoModel

    target = torch.device(device)
    dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
    base = AutoModel.from_pretrained(
        config.base_repository,
        revision=config.base_revision,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).eval()
    compressor = AutoModel.from_pretrained(
        config.compressor_repository,
        revision=config.compressor_revision,
        trust_remote_code=True,
        torch_dtype=dtype,
        rae=base,
    ).eval()
    compressor.to(target, dtype=dtype)
    return RAECodec(compressor, config, target)
