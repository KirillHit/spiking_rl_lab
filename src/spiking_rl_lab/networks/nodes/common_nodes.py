"""Common network node implementations."""

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


@register_node("linear")
class LinearNode(BaseNode):
    """Dense linear layer node.

    Linear nodes are for flat MLP-style tensors shaped ``[batch, features]``.
    Use convolutional nodes for channel-first sequence or image tensors.
    """

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
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


@register_node("conv1d")
class Conv1dNode(BaseNode):
    """Convolution layer node for channel-first sequence tensors."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
        """1D convolution layer configuration."""

        out_channels: int
        kernel_size: int
        stride: int = 1
        padding: int | str = 0
        dilation: int = 1
        groups: int = 1
        bias: bool = False
        padding_mode: str = "zeros"

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        shape = require_shape("Node 'conv1d' input", input_shape, SequenceTensorShape)
        self._layer = nn.Conv1d(
            in_channels=shape.channels,
            out_channels=cfg.out_channels,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            padding=cfg.padding,
            dilation=cfg.dilation,
            groups=cfg.groups,
            bias=cfg.bias,
            padding_mode=cfg.padding_mode,
        )
        with torch.no_grad():
            outputs = self._layer(torch.zeros(1, shape.channels, shape.length))
        self._output_shape = TensorShape.sequence(
            channels=outputs.shape[1],
            length=outputs.shape[2],
        )

    @property
    def output_shape(self) -> TensorShape:
        """Return output shape."""
        return self._output_shape

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Run the 1D convolution layer."""
        return self._layer(inputs), None


@register_node("conv2d")
class Conv2dNode(BaseNode):
    """Convolution layer node for channel-first image tensors."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
        """2D convolution layer configuration."""

        out_channels: int
        kernel_size: int | tuple[int, ...]
        stride: int | tuple[int, ...] = 1
        padding: int | tuple[int, ...] | str = 0
        dilation: int | tuple[int, ...] = 1
        groups: int = 1
        bias: bool = False
        padding_mode: str = "zeros"

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        image_shape = require_shape("Node 'conv2d' input", input_shape, ImageTensorShape)
        self._layer = nn.Conv2d(
            in_channels=image_shape.channels,
            out_channels=cfg.out_channels,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            padding=cfg.padding,
            dilation=cfg.dilation,
            groups=cfg.groups,
            bias=cfg.bias,
            padding_mode=cfg.padding_mode,
        )
        with torch.no_grad():
            outputs = self._layer(
                torch.zeros(1, image_shape.channels, image_shape.height, image_shape.width)
            )
        self._output_shape = TensorShape.image(
            channels=outputs.shape[1],
            height=outputs.shape[2],
            width=outputs.shape[3],
        )

    @property
    def output_shape(self) -> TensorShape:
        """Return output shape."""
        return self._output_shape

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Run the 2D convolution layer."""
        return self._layer(inputs), None


@register_node("batch_norm1d")
class BatchNorm1dNode(BaseNode):
    """Batch normalization layer node for dense or sequence tensors."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
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
    class Config:
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
