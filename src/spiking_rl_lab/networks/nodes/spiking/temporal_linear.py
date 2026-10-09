"""Causal temporal connections for streaming dense network inputs."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional

from spiking_rl_lab.core.validation import require_minimum
from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node
from spiking_rl_lab.networks.shape import DenseTensorShape, TensorShape, require_shape

if TYPE_CHECKING:
    from spiking_rl_lab.networks.state import ListState


@register_node("temporal_linear")
class TemporalLinearNode(BaseNode):
    """Sum weighted current and past inputs without modifying neuron dynamics.

    Inputs have shape [batch, features]. State stores the previous window - 1
    input vectors, newest first. Training uses a dense linear operation. This
    node does not implement a sparse event scheduler.
    """

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """Number of output features, causal taps and optional output bias."""

        out_features: int
        window: int = 8
        bias: bool = False

        def validate(self) -> None:
            """Require positive feature and window dimensions."""
            require_minimum("out_features", self.out_features, minimum=1)
            require_minimum("window", self.window, minimum=1)

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Allocate output-by-lag-by-input weights."""
        super().__init__(cfg, input_shape)
        shape = require_shape("Node 'temporal_linear' input", input_shape, DenseTensorShape)
        self.window = cfg.window
        self.weight = nn.Parameter(torch.empty(cfg.out_features, cfg.window, shape.features))
        self.bias = nn.Parameter(torch.empty(cfg.out_features)) if cfg.bias else None
        self._output_shape = TensorShape.dense(cfg.out_features)

    @property
    def output_shape(self) -> TensorShape:
        """Return the dense output shape."""
        return self._output_shape

    def initialize_parameters(self) -> None:
        """Use LinearNode initialization with fan-in including all temporal taps."""
        nn.init.kaiming_normal_(self.weight.flatten(1), mode="fan_in", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def initial_state(self, inputs: torch.Tensor) -> ListState | None:
        """Create empty history. A one-tap connection is stateless."""
        if self.window == 1:
            return None
        return [inputs.new_zeros(inputs.shape[0], self.window - 1, inputs.shape[-1])]

    def reset_state(self, state: ListState | None, dones: torch.Tensor) -> ListState | None:
        """Clear history only for ended episodes without modifying the input state."""
        if state is None:
            return None
        return [state[0].masked_fill(dones.reshape(-1, 1, 1), 0)]

    def forward(
        self, inputs: torch.Tensor, state: ListState | None = None
    ) -> tuple[torch.Tensor, ListState | None]:
        """Compute a causal current and return the next explicit history."""
        if self.window == 1:
            return functional.linear(inputs, self.weight[:, 0], self.bias), None
        if state is None:
            state = self.initial_state(inputs)
        taps = torch.cat((inputs.unsqueeze(1), state[0]), dim=1)
        output = functional.linear(taps.flatten(1), self.weight.flatten(1), self.bias)
        return output, [taps[:, :-1]]
