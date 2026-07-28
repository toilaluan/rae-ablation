from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .layers import (
    DDTDecoderBlock,
    DDTEncoderBlock,
    DDTFinalLayer,
    DDTInternalREGHead,
    LowRankAdapter,
    REPAProjection,
)
from .tokens import (
    ClassTokenEmbedder,
    DDTTokenLayout,
    TimestepTokenEmbedder,
)


class DDTOutput(NamedTuple):
    image: Tensor
    reg: Tensor | None = None
    repa: Tensor | None = None
    reg_early: Tensor | None = None


class DDTREGOutput(NamedTuple):
    reg: Tensor
    reg_early: Tensor


class DDT(nn.Module):
    """DDT over pixel or normalized FLUX.2 latent tensors."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.grid_size = config.grid_size
        reg_config = config.features.reg
        repa_config = config.features.repa
        decoder = config.decoder
        reg_tokens = (
            reg_config.num_tokens if reg_config.enabled else 0
        )
        reg_pool_size = (
            reg_config.pool_size if reg_config.enabled else None
        )
        attention_mode = (
            reg_config.attention_mode
            if reg_config.enabled
            else "joint"
        )
        self.encoder_layout = DDTTokenLayout(
            reg_data_tokens=reg_tokens,
            image_data_tokens=self.grid_size**2,
            reg_timestep_tokens=config.num_reg_timestep_tokens,
            reg_class_tokens=config.num_reg_class_tokens,
            image_timestep_tokens=config.num_image_timestep_tokens,
            image_class_tokens=config.num_image_class_tokens,
        )
        self.decoder_layout = DDTTokenLayout(
            reg_data_tokens=reg_tokens,
            image_data_tokens=self.grid_size**2,
        )

        self.encoder_patch_projection = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.decoder_patch_projection = nn.Conv2d(
            config.in_channels,
            decoder.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.image_timestep_embedder = TimestepTokenEmbedder(
            config.hidden_size,
            num_tokens=config.num_image_timestep_tokens,
        )
        self.reg_timestep_embedder = (
            TimestepTokenEmbedder(
                config.hidden_size,
                num_tokens=config.num_reg_timestep_tokens,
            )
            if reg_config.enabled
            else None
        )
        self.class_embedder = ClassTokenEmbedder(
            config.hidden_size,
            config.num_classes,
            num_image_tokens=config.num_image_class_tokens,
            num_reg_tokens=config.num_reg_class_tokens,
        )

        spatial_reg_rope = (
            reg_config.enabled and reg_config.spatial_rope
        )
        reg_lora = reg_config.lora if reg_config.enabled else None
        self.encoder_blocks = nn.ModuleList(
            [
                DDTEncoderBlock(
                    config.hidden_size,
                    config.num_heads,
                    config.mlp_ratio,
                    grid_size=self.grid_size,
                    layout=self.encoder_layout,
                    reg_pool_size=reg_pool_size,
                    spatial_reg_rope=spatial_reg_rope,
                    attention_mode=attention_mode,
                    reg_lora=reg_lora,
                    fused_mlp=config.fused_mlp,
                )
                for _ in range(config.depth)
            ]
        )
        self.context_projection: nn.Module = (
            nn.Linear(config.hidden_size, decoder.hidden_size)
            if config.hidden_size != decoder.hidden_size
            else nn.Identity()
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DDTDecoderBlock(
                    decoder.hidden_size,
                    decoder.num_heads,
                    decoder.mlp_ratio,
                    grid_size=self.grid_size,
                    layout=self.decoder_layout,
                    reg_pool_size=reg_pool_size,
                    spatial_reg_rope=spatial_reg_rope,
                    attention_mode=attention_mode,
                    reg_lora=reg_lora,
                    fused_mlp=config.fused_mlp,
                )
                for _ in range(decoder.depth)
            ]
        )
        patch_values = (
            config.patch_size
            * config.patch_size
            * config.in_channels
        )
        feature_dim = config.features.encoder.feature_dim
        self.final_layer = DDTFinalLayer(
            decoder.hidden_size,
            patch_values,
            feature_dim if reg_config.enabled else None,
            self.decoder_layout,
        )
        self.internal_reg_head = (
            DDTInternalREGHead(config.hidden_size, feature_dim)
            if reg_config.enabled
            else None
        )
        if reg_config.enabled:
            self.encoder_reg_projection: nn.Linear | None = nn.Linear(
                feature_dim,
                config.hidden_size,
            )
            self.decoder_reg_projection: nn.Linear | None = nn.Linear(
                feature_dim,
                decoder.hidden_size,
            )
        else:
            self.encoder_reg_projection = None
            self.decoder_reg_projection = None
        self.repa_projection = (
            REPAProjection(
                config.hidden_size,
                repa_config.projection_hidden_size,
                feature_dim,
            )
            if repa_config.enabled
            else None
        )
        self.gradient_checkpointing = config.gradient_checkpointing
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        for module in self.modules():
            if isinstance(module, LowRankAdapter):
                module.reset_parameters()
        for projection in (
            self.encoder_patch_projection,
            self.decoder_patch_projection,
        ):
            nn.init.xavier_uniform_(projection.weight.flatten(1))
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
        nn.init.normal_(
            self.class_embedder.embedding_table.weight,
            std=0.02,
        )
        time_embedders = [self.image_timestep_embedder]
        if self.reg_timestep_embedder is not None:
            time_embedders.append(self.reg_timestep_embedder)
        for embedder in time_embedders:
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)
        for block in self.decoder_blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.modulation[-1].weight)
        nn.init.zeros_(self.final_layer.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.image_projection.weight)
        nn.init.zeros_(self.final_layer.image_projection.bias)
        if self.final_layer.reg_projection is not None:
            nn.init.zeros_(self.final_layer.reg_projection.weight)
            nn.init.zeros_(self.final_layer.reg_projection.bias)
        if self.internal_reg_head is not None:
            nn.init.zeros_(
                self.internal_reg_head.modulation[-1].weight
            )
            nn.init.zeros_(
                self.internal_reg_head.modulation[-1].bias
            )
            nn.init.zeros_(
                self.internal_reg_head.projection.weight
            )
            nn.init.zeros_(
                self.internal_reg_head.projection.bias
            )

    def set_gradient_checkpointing(
        self,
        enabled: bool = True,
    ) -> None:
        self.gradient_checkpointing = enabled

    def _checkpointing(self) -> bool:
        return (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )

    def _unpatchify(self, tokens: Tensor) -> Tensor:
        batch = tokens.shape[0]
        patch = self.config.patch_size
        channels = self.config.in_channels
        values = tokens.reshape(
            batch,
            self.grid_size,
            self.grid_size,
            patch,
            patch,
            channels,
        )
        return torch.einsum(
            "bhwpqc->bchpwq",
            values,
        ).reshape(
            batch,
            channels,
            self.config.input_size,
            self.config.input_size,
        )

    def _validate_reg_input(
        self,
        noisy_reg: Tensor,
        reg_timestep: Tensor,
        labels: Tensor,
    ) -> None:
        batch = noisy_reg.shape[0]
        expected = (
            batch,
            self.encoder_layout.reg_data_tokens,
            self.config.features.encoder.feature_dim,
        )
        if noisy_reg.ndim != 3 or tuple(noisy_reg.shape) != expected:
            raise ValueError(
                "noisy_reg must have shape "
                f"[B, {expected[1]}, {expected[2]}]"
            )
        if reg_timestep.shape != (batch,):
            raise ValueError("reg_timestep must have shape [B]")
        if labels.shape != (batch,):
            raise ValueError("labels must have shape [B]")

    def _reg_predictions(
        self,
        encoded_reg: Tensor,
        early_reg_state: Tensor,
        reg_time_base: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.internal_reg_head is None:
            raise RuntimeError("REG prediction head is not configured")
        early_context = F.silu(
            early_reg_state + reg_time_base
        )
        reg_early = self.internal_reg_head(
            early_context,
            early_context,
        )
        decoder_context = self.context_projection(
            F.silu(encoded_reg + reg_time_base)
        )
        return reg_early, decoder_context

    def _encode(
        self,
        sequence: Tensor,
        *,
        reg_stream: bool,
        reg_dropout: bool = False,
        return_repa: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        reg_config = self.config.features.reg
        early_reg_state: Tensor | None = None
        repa: Tensor | None = None
        for index, block in enumerate(
            self.encoder_blocks,
            start=1,
        ):
            if reg_stream:
                if self._checkpointing():
                    sequence = checkpoint(
                        block.forward_reg,
                        sequence,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    sequence = block.forward_reg(sequence)
            else:
                image_attends_reg = (
                    not reg_dropout
                    and (
                        reg_config.attention_mode != "asymmetric"
                        or index > reg_config.join_after_block
                    )
                )
                if self._checkpointing():
                    sequence = checkpoint(
                        block,
                        sequence,
                        image_attends_reg,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    sequence = block(
                        sequence,
                        image_attends_reg=image_attends_reg,
                    )
            if (
                self.internal_reg_head is not None
                and index == reg_config.internal_guidance_block
            ):
                early_reg_state = sequence[
                    :, self.encoder_layout.reg_data
                ]
            if (
                not reg_stream
                and return_repa
                and self.repa_projection is not None
                and index == self.config.features.repa.block
            ):
                repa_states = (
                    self.encoder_layout.gather_data(sequence)
                    if (
                        self.config.features.repa.token_scope
                        == "image_and_reg"
                    )
                    else sequence[
                        :, self.encoder_layout.image_data
                    ]
                )
                repa = self.repa_projection(repa_states)
        return sequence, early_reg_state, repa

    def _decode_reg(
        self,
        noisy_reg: Tensor,
        decoder_context: Tensor,
    ) -> Tensor:
        if self.decoder_reg_projection is None:
            raise RuntimeError("REG decoder projection is not configured")
        decoded = self.decoder_reg_projection(noisy_reg)
        for block in self.decoder_blocks:
            if self._checkpointing():
                decoded = checkpoint(
                    block.forward_reg,
                    decoded,
                    decoder_context,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                decoded = block.forward_reg(
                    decoded,
                    decoder_context,
                )
        return self.final_layer.forward_reg(
            decoded,
            decoder_context,
        )

    def forward_reg(
        self,
        noisy_reg: Tensor,
        labels: Tensor,
        *,
        reg_timestep: Tensor,
    ) -> DDTREGOutput:
        reg_config = self.config.features.reg
        if (
            not reg_config.enabled
            or reg_config.attention_mode != "asymmetric"
        ):
            raise ValueError(
                "REG-only forward requires asymmetric attention"
            )
        self._validate_reg_input(
            noisy_reg,
            reg_timestep,
            labels,
        )
        assert (
            self.reg_timestep_embedder is not None
            and self.encoder_reg_projection is not None
        )
        reg_time = self.reg_timestep_embedder(reg_timestep)
        class_tokens = self.class_embedder(labels)
        reg_layout = self.encoder_layout.reg_stream_layout
        sequence = reg_layout.pack(
            reg_data=self.encoder_reg_projection(noisy_reg),
            reg_timestep=reg_time.tokens,
            reg_class=class_tokens.reg_tokens,
        )
        sequence, early_reg_state, _repa = self._encode(
            sequence,
            reg_stream=True,
        )
        if early_reg_state is None:
            raise RuntimeError(
                "REG-only forward did not capture its guidance state"
            )
        encoded_reg = sequence[
            :, reg_layout.reg_data
        ]
        reg_early, decoder_context = self._reg_predictions(
            encoded_reg,
            early_reg_state,
            reg_time.base,
        )
        reg = self._decode_reg(noisy_reg, decoder_context)
        return DDTREGOutput(reg=reg, reg_early=reg_early)

    def forward(
        self,
        noisy_image: Tensor,
        timestep: Tensor,
        labels: Tensor,
        noisy_reg: Tensor | None = None,
        reg_timestep: Tensor | None = None,
        return_repa: bool = False,
        reg_dropout: bool = False,
    ) -> DDTOutput:
        batch = noisy_image.shape[0]
        expected_image = (
            batch,
            self.config.in_channels,
            self.config.input_size,
            self.config.input_size,
        )
        if (
            noisy_image.ndim != 4
            or tuple(noisy_image.shape) != expected_image
        ):
            raise ValueError(
                "noisy_image must have shape "
                f"[B, {expected_image[1]}, {expected_image[2]}, "
                f"{expected_image[3]}]"
            )
        if timestep.shape != (batch,) or labels.shape != (batch,):
            raise ValueError(
                "timestep and labels must both have shape [B]"
            )
        reg_config = self.config.features.reg
        if reg_dropout and (
            not reg_config.enabled
            or reg_config.attention_mode != "asymmetric"
        ):
            raise ValueError(
                "REG dropout requires asymmetric attention"
            )

        image_time = self.image_timestep_embedder(timestep)
        class_tokens = self.class_embedder(labels)
        encoder_image = (
            self.encoder_patch_projection(noisy_image)
            .flatten(2)
            .transpose(1, 2)
        )
        decoder_image = (
            self.decoder_patch_projection(noisy_image)
            .flatten(2)
            .transpose(1, 2)
        )

        encoder_reg: Tensor | None = None
        decoder_reg: Tensor | None = None
        reg_time_base: Tensor | None = None
        reg_time_tokens: Tensor | None = None
        if reg_config.enabled:
            if noisy_reg is None or reg_timestep is None:
                raise ValueError(
                    "REG requires noisy_reg and reg_timestep"
                )
            self._validate_reg_input(
                noisy_reg,
                reg_timestep,
                labels,
            )
            assert (
                self.reg_timestep_embedder is not None
                and self.encoder_reg_projection is not None
                and self.decoder_reg_projection is not None
            )
            reg_time = self.reg_timestep_embedder(reg_timestep)
            reg_time_base = reg_time.base
            reg_time_tokens = reg_time.tokens
            encoder_reg = self.encoder_reg_projection(noisy_reg)
            decoder_reg = self.decoder_reg_projection(noisy_reg)
        elif noisy_reg is not None or reg_timestep is not None:
            raise ValueError(
                "noisy_reg and reg_timestep are only valid when REG is enabled"
            )

        sequence = self.encoder_layout.pack(
            reg_data=encoder_reg,
            image_data=encoder_image,
            reg_timestep=reg_time_tokens,
            reg_class=(
                class_tokens.reg_tokens
                if reg_config.enabled
                else None
            ),
            image_timestep=image_time.tokens,
            image_class=class_tokens.image_tokens,
        )
        sequence, early_reg_state, repa = self._encode(
            sequence,
            reg_stream=False,
            reg_dropout=reg_dropout,
            return_repa=return_repa,
        )

        encoded_image = sequence[
            :, self.encoder_layout.image_data
        ]
        image_context = F.silu(
            encoded_image + image_time.base
        )
        reg_early: Tensor | None = None
        reg_context: Tensor | None = None
        if reg_config.enabled:
            if (
                early_reg_state is None
                or reg_time_base is None
            ):
                raise RuntimeError(
                    "REG encoder state was not captured"
                )
            encoded_reg = sequence[
                :, self.encoder_layout.reg_data
            ]
            reg_early, reg_context = self._reg_predictions(
                encoded_reg,
                early_reg_state,
                reg_time_base,
            )
        decoder_context = self.decoder_layout.pack(
            reg_data=reg_context,
            image_data=self.context_projection(image_context),
        )
        decoded = self.decoder_layout.pack(
            reg_data=decoder_reg,
            image_data=decoder_image,
        )
        for block in self.decoder_blocks:
            image_attends_reg = not reg_dropout
            if self._checkpointing():
                decoded = checkpoint(
                    block,
                    decoded,
                    decoder_context,
                    image_attends_reg,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                decoded = block(
                    decoded,
                    decoder_context,
                    image_attends_reg=image_attends_reg,
                )
        image_tokens, reg = self.final_layer(
            decoded,
            decoder_context,
        )
        return DDTOutput(
            image=self._unpatchify(image_tokens),
            reg=reg,
            repa=repa,
            reg_early=reg_early,
        )
