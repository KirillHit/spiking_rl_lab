"""Common network node implementations."""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch
from torch import nn

from spiking_rl_lab.core.exception import NetworkCreationError
from spiking_rl_lab.networks.nodes.base_node import BaseNode, BaseNodeCfg, ListState
from spiking_rl_lab.networks.nodes.register import register_node
from spiking_rl_lab.networks.shape import (
    DenseTensorShape,
    ImageTensorShape,
    SequenceTensorShape,
    TensorShape,
)

_PAIR_SIZE = 2


def _as_pair(value: int | tuple[int, ...], *, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != _PAIR_SIZE:
        msg = f"{name} must be an int or a 2-item tuple"
        raise NetworkCreationError(msg)
    return value[0], value[1]


def _conv_out(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return ((size + 2 * padding - dilation * (kernel - 1) - 1) // stride) + 1


def _as_dense_shape(input_shape: TensorShape, node_name: str) -> DenseTensorShape:
    if isinstance(input_shape, DenseTensorShape):
        return input_shape
    msg = f"Node '{node_name}' requires dense input with features only"
    raise NetworkCreationError(msg)


def _as_sequence_shape(input_shape: TensorShape, node_name: str) -> SequenceTensorShape:
    if isinstance(input_shape, SequenceTensorShape):
        return input_shape
    msg = f"Node '{node_name}' requires sequence input with channels and length"
    raise NetworkCreationError(msg)


def _as_image_shape(input_shape: TensorShape, node_name: str) -> ImageTensorShape:
    if isinstance(input_shape, ImageTensorShape):
        return input_shape
    msg = f"Node '{node_name}' requires image input with channels, height and width"
    raise NetworkCreationError(msg)


@dataclasses.dataclass(kw_only=True, slots=True)
class LinearNodeCfg(BaseNodeCfg):
    """Linear layer configuration."""

    out_features: int
    bias: bool = False


@register_node("linear")
class LinearNode(BaseNode[LinearNodeCfg]):
    """Dense linear layer node.

    Linear nodes are for flat MLP-style tensors shaped ``[batch, features]``.
    Use convolutional nodes for channel-first sequence or image tensors.
    """

    def __init__(self, cfg: LinearNodeCfg, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        dense_shape = _as_dense_shape(input_shape, "linear")
        self._layer = nn.Linear(
            in_features=dense_shape.features,
            out_features=cfg.out_features,
            bias=cfg.bias,
        )
        self._output_shape = TensorShape.dense(cfg.out_features)

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Run the linear layer."""
        return self._layer(inputs), state


@dataclasses.dataclass(kw_only=True, slots=True)
class ConvNodeCfg(BaseNodeCfg):
    """Convolution layer configuration."""

    dim: Literal[1, 2] = 2
    out_channels: int
    kernel_size: int | tuple[int, ...]
    stride: int | tuple[int, ...] = 1
    padding: int | tuple[int, ...] | str = 0
    dilation: int | tuple[int, ...] = 1
    groups: int = 1
    bias: bool = False
    padding_mode: str = "zeros"


@register_node("conv")
class ConvNode(BaseNode[ConvNodeCfg]):
    """Convolution layer node for channel-first sequence or image tensors.

    ``dim=1`` expects ``TensorShape.sequence(channels, length)`` and operates on
    tensors shaped ``[batch, channels, length]``. ``dim=2`` expects
    ``TensorShape.image(channels, height, width)`` and operates on tensors shaped
    ``[batch, channels, height, width]``.
    """

    def __init__(self, cfg: ConvNodeCfg, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        conv_cls = {1: nn.Conv1d, 2: nn.Conv2d}[cfg.dim]
        shape = (
            _as_sequence_shape(input_shape, "conv")
            if cfg.dim == 1
            else _as_image_shape(input_shape, "conv")
        )
        self._layer = conv_cls(
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
        self._output_shape = self._get_output_shape(cfg, input_shape)

    def _get_output_shape(self, cfg: ConvNodeCfg, input_shape: TensorShape) -> TensorShape:
        if cfg.dim == 1:
            sequence_shape = _as_sequence_shape(input_shape, "conv")
            kernel = cfg.kernel_size if isinstance(cfg.kernel_size, int) else cfg.kernel_size[0]
            stride = cfg.stride if isinstance(cfg.stride, int) else cfg.stride[0]
            padding = cfg.padding if isinstance(cfg.padding, int) else cfg.padding[0]
            dilation = cfg.dilation if isinstance(cfg.dilation, int) else cfg.dilation[0]
            if not isinstance(padding, int):
                msg = "Conv1d automatic shape requires integer padding"
                raise NetworkCreationError(msg)
            return TensorShape.sequence(
                channels=cfg.out_channels,
                length=_conv_out(sequence_shape.length, kernel, stride, padding, dilation),
            )

        image_shape = _as_image_shape(input_shape, "conv")
        kernel_h, kernel_w = _as_pair(cfg.kernel_size, name="kernel_size")
        stride_h, stride_w = _as_pair(cfg.stride, name="stride")
        dilation_h, dilation_w = _as_pair(cfg.dilation, name="dilation")
        if not isinstance(cfg.padding, int | tuple):
            msg = "Conv2d automatic shape requires integer padding"
            raise NetworkCreationError(msg)
        padding_h, padding_w = _as_pair(cfg.padding, name="padding")
        return TensorShape.image(
            channels=cfg.out_channels,
            height=_conv_out(image_shape.height, kernel_h, stride_h, padding_h, dilation_h),
            width=_conv_out(image_shape.width, kernel_w, stride_w, padding_w, dilation_w),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Run the convolution layer."""
        return self._layer(inputs), state


@dataclasses.dataclass(kw_only=True, slots=True)
class BatchNormNodeCfg(BaseNodeCfg):
    """Batch normalization layer configuration."""

    dim: Literal[1, 2] = 2
    eps: float = 1e-5
    momentum: float | None = 0.1
    affine: bool = False


@register_node("batch_norm")
class BatchNormNode(BaseNode[BatchNormNodeCfg]):
    """Batch normalization layer node for dense, sequence or image tensors."""

    def __init__(self, cfg: BatchNormNodeCfg, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        batch_norm_cls = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d}[cfg.dim]
        self._layer = batch_norm_cls(
            num_features=self._get_num_features(cfg, input_shape),
            eps=cfg.eps,
            momentum=cfg.momentum,
            affine=cfg.affine,
        )

    def _get_num_features(self, cfg: BatchNormNodeCfg, input_shape: TensorShape) -> int:
        if cfg.dim == 1:
            if isinstance(input_shape, DenseTensorShape):
                return input_shape.features
            if isinstance(input_shape, SequenceTensorShape):
                return input_shape.channels
            msg = "Node 'batch_norm' requires dense or sequence input"
            raise NetworkCreationError(msg)
        return _as_image_shape(input_shape, "batch_norm").channels

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Run the batch normalization layer."""
        return self._layer(inputs), state
