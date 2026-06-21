"""Normalization network node implementations."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from torch import nn

from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node
from spiking_rl_lab.networks.shape import (
    DenseTensorShape,
    ImageTensorShape,
    SequenceTensorShape,
    TensorShape,
    require_shape,
)

if TYPE_CHECKING:
    from spiking_rl_lab.networks.base_network import ListState


@register_node("batch_norm1d")
class BatchNorm1dNode(BaseNode):
    """Batch normalization layer node for dense or sequence tensors."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """1D batch normalization layer configuration."""

        eps: float = 1e-5
        momentum: float | None = 0.1
        affine: bool = False

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        shape = require_shape(
            "Node 'batch_norm1d' input",
            input_shape,
            (DenseTensorShape, SequenceTensorShape),
        )
        num_features = shape.features if isinstance(shape, DenseTensorShape) else shape.channels
        self._layer = nn.BatchNorm1d(
            num_features=num_features,
            eps=cfg.eps,
            momentum=cfg.momentum,
            affine=cfg.affine,
        )

    @property
    def output_shape(self) -> TensorShape:
        """Return output shape."""
        return self._input_shape

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Run the 1D batch normalization layer."""
        return self._layer(inputs), None


@register_node("batch_norm2d")
class BatchNorm2dNode(BaseNode):
    """Batch normalization layer node for image tensors."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """2D batch normalization layer configuration."""

        eps: float = 1e-5
        momentum: float | None = 0.1
        affine: bool = False

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        image_shape = require_shape("Node 'batch_norm2d' input", input_shape, ImageTensorShape)
        self._layer = nn.BatchNorm2d(
            num_features=image_shape.channels,
            eps=cfg.eps,
            momentum=cfg.momentum,
            affine=cfg.affine,
        )

    @property
    def output_shape(self) -> TensorShape:
        """Return output shape."""
        return self._input_shape

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Run the 2D batch normalization layer."""
        return self._layer(inputs), None
