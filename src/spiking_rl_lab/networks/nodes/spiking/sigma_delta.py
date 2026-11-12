"""ReLU represented by thresholded corrections with a Gaussian surrogate."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn
from torch.nn import functional

from spiking_rl_lab.core.validation import require_positive
from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node
from spiking_rl_lab.networks.nodes.statistics import NodeStatistics
from spiking_rl_lab.networks.shape import DenseTensorShape, TensorShape, require_shape

if TYPE_CHECKING:
    from spiking_rl_lab.networks.nodes.statistics import NodeStatisticsAccumulator


class SigmaDeltaState(NamedTuple):
    """Represented output accumulated from zero."""

    value: torch.Tensor  # [batch, features], sum of corrections already transmitted.


@dataclasses.dataclass
class _TransmissionStatistics:
    """Count actual changes of the represented output without retaining its graph."""

    event_total: torch.Tensor | None = None
    count: int = 0

    def update(
        self,
        inputs: torch.Tensor,
        state: SigmaDeltaState | None,
        outputs: torch.Tensor,
        next_state: SigmaDeltaState,
    ) -> None:
        """Count signed corrections, including the first step from zero."""
        previous = 0 if state is None else state.value.detach()
        events = (outputs.detach() != previous).to(outputs.dtype).sum()
        self.event_total = events if self.event_total is None else self.event_total + events
        self.count += outputs.numel()

    def result(self) -> NodeStatistics | None:
        """Report the fraction and total number of transmitted corrections."""
        if not self.count:
            return None
        return NodeStatistics(
            loss=None,
            metrics={
                "activity_mean": self.event_total / self.count,
                "event_count": self.event_total,
            },
        )


@register_node("sigma_delta")
class SigmaDeltaNode(BaseNode):
    """Transmit changes above a learned threshold and return the represented value.

    Forward uses the same hard decision in train and eval. Backward differentiates
    the Gaussian-smoothed correction, including the previous value and threshold.
    State remains connected to the graph throughout a sequence.
    """

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """Initial per-coordinate threshold and absolute Gaussian smoothing width."""

        threshold: float = 0.01
        surrogate_width: float = 0.03

        def validate(self) -> None:
            """Require positive scales."""
            require_positive("threshold", self.threshold)
            require_positive("surrogate_width", self.surrogate_width)

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Allocate one unconstrained threshold parameter per input coordinate."""
        super().__init__(cfg, input_shape)
        shape = require_shape("Node 'sigma_delta' input", input_shape, DenseTensorShape)
        self.raw_threshold = nn.Parameter(torch.empty(shape.features))

    @property
    def output_shape(self) -> TensorShape:
        """Preserve the input shape."""
        return self._input_shape

    @property
    def threshold(self) -> torch.Tensor:
        """Return positive thresholds from unconstrained parameters."""
        return functional.softplus(self.raw_threshold)

    @torch.no_grad()
    def initialize_parameters(self) -> None:
        """Initialize softplus parameters to the configured threshold."""
        self.raw_threshold.fill_(self._cfg.threshold + math.log(-math.expm1(-self._cfg.threshold)))

    def initial_state(self, inputs: torch.Tensor) -> SigmaDeltaState:
        """Start each batch row with a zero represented value."""
        return SigmaDeltaState(torch.zeros_like(inputs))

    def reset_state(self, state: SigmaDeltaState, dones: torch.Tensor) -> SigmaDeltaState:
        """Reset completed rows without modifying the supplied state."""
        return SigmaDeltaState(state.value.masked_fill(dones.reshape(-1, 1), 0))

    def create_statistics(
        self, *, progress: float, collect_metrics: bool = False
    ) -> NodeStatisticsAccumulator | None:
        """Collect transmissions only when metrics are requested."""
        return _TransmissionStatistics() if collect_metrics else None

    def forward(
        self, inputs: torch.Tensor, state: SigmaDeltaState | None = None
    ) -> tuple[torch.Tensor, SigmaDeltaState]:
        """Hold the previous value or transmit the full signed ReLU correction."""
        if state is None:
            state = self.initial_state(inputs)
        value = inputs.relu()
        residual = value - state.value
        hard = state.value + torch.where(residual.abs() > self.threshold, residual, 0.0)
        if torch.is_grad_enabled():
            scale = self._cfg.surrogate_width * math.sqrt(2)
            probability = 0.5 * (1 + torch.erf((residual - self.threshold) / scale))
            probability = probability + 0.5 * (1 + torch.erf((-residual - self.threshold) / scale))
            smooth = state.value + residual * probability
            hard = hard.detach() + (smooth - smooth.detach())
        return hard, SigmaDeltaState(hard)
