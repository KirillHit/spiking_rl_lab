"""Linear network node implementations."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from torch import nn

from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node
from spiking_rl_lab.networks.shape import DenseTensorShape, TensorShape, require_shape

if TYPE_CHECKING:
    from spiking_rl_lab.networks.base_network import ListState


@register_node("linear")
class LinearNode(BaseNode):
    """Dense linear layer node.

    Linear nodes are for flat MLP-style tensors shaped ``[batch, features]``.
    Use convolutional nodes for channel-first sequence or image tensors.
    """

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """Linear layer configuration."""

        out_features: int
        bias: bool = False

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        dense_shape = require_shape("Node 'linear' input", input_shape, DenseTensorShape)
        self._layer = nn.Linear(
            in_features=dense_shape.features,
            out_features=cfg.out_features,
            bias=cfg.bias,
        )
        self._output_shape = TensorShape.dense(cfg.out_features)

    @property
    def output_shape(self) -> TensorShape:
        """Return output shape."""
        return self._output_shape

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Run the linear layer."""
        return self._layer(inputs), None
