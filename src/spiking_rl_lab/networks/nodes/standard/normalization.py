"""Batch normalization with explicitly managed running statistics."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional

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
    from spiking_rl_lab.networks.state import ListState


class _ChannelStatisticsAccumulator:
    """Accumulate channel-wise moments without retaining input tensors."""

    def __init__(self, reference: torch.Tensor) -> None:
        """Create empty moments matching the channel reference tensor."""
        self.sum = torch.zeros_like(reference)
        self.sum_of_squares = torch.zeros_like(reference)
        self.count = 0

    def update(self, inputs: torch.Tensor) -> None:
        """Include another channel-first tensor in the accumulated moments."""
        dimensions = (0, *range(2, inputs.ndim))
        with torch.no_grad():
            self.sum.add_(inputs.sum(dim=dimensions, dtype=self.sum.dtype))
            self.sum_of_squares.add_(inputs.square().sum(dim=dimensions, dtype=self.sum.dtype))
        self.count += inputs.numel() // inputs.shape[1]

    def statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the accumulated mean and unbiased variance."""
        mean = self.sum / self.count
        variance = (self.sum_of_squares - self.sum.square() / self.count) / (self.count - 1)
        return mean, variance.clamp_min(0)


@register_node("batch_norm")
class BatchNormNode(BaseNode):
    """Normalize with fixed running statistics in both train and eval modes.

    Statistics collection is enabled explicitly during an optimization step, so
    rollout replay continues using the previously applied statistics.

    The affine transform uses a learnable scale without a bias.
    """

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """Batch normalization configuration."""

        eps: float = 1e-5
        momentum: float | None = 0.1

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Choose channel-wise normalization from the input shape."""
        super().__init__(cfg, input_shape)
        shape = require_shape(
            "Node 'batch_norm' input",
            input_shape,
            (DenseTensorShape, SequenceTensorShape, ImageTensorShape),
        )
        features = shape.features if isinstance(shape, DenseTensorShape) else shape.channels
        layer_type = nn.BatchNorm2d if isinstance(shape, ImageTensorShape) else nn.BatchNorm1d
        self._layer = layer_type(features, eps=cfg.eps, momentum=cfg.momentum)
        self._layer.register_parameter("bias", None)
        self._statistics_accumulator: _ChannelStatisticsAccumulator | None = None

    @property
    def output_shape(self) -> TensorShape:
        """Return the unchanged input shape."""
        return self._input_shape

    def enable_batch_norm_statistics_collection(self) -> None:
        """Start accumulating input statistics without changing normalization."""
        self._statistics_accumulator = _ChannelStatisticsAccumulator(self._layer.running_mean)

    def apply_collected_batch_norm_statistics(self) -> None:
        """Apply one running-statistics update from all collected inputs."""
        if self._statistics_accumulator is None:
            return

        accumulator = self._statistics_accumulator
        self._statistics_accumulator = None

        if accumulator.count <= 1:
            return

        mean, variance = accumulator.statistics()
        with torch.no_grad():
            self._layer.num_batches_tracked.add_(1)
            momentum = (
                self._layer.momentum
                if self._layer.momentum is not None
                else 1.0 / self._layer.num_batches_tracked.item()
            )
            self._layer.running_mean.lerp_(mean, momentum)
            self._layer.running_var.lerp_(variance, momentum)

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Normalize without mutating buffers, regardless of the module mode."""
        if self._statistics_accumulator is not None:
            self._statistics_accumulator.update(inputs)
        return functional.batch_norm(
            inputs,
            self._layer.running_mean,
            self._layer.running_var,
            self._layer.weight,
            self._layer.bias,
            training=False,
            eps=self._layer.eps,
        ), None
