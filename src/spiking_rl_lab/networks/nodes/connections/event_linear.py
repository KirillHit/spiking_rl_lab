"""Linear connections that retain source identity for receiver-side traces."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from spiking_rl_lab.core.validation import require_positive
from spiking_rl_lab.networks.events import (
    ConnectionContext,
    EventPacket,
    Events,
)
from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node
from spiking_rl_lab.networks.nodes.connections.event_linear_execution import execute_linear
from spiking_rl_lab.networks.shape import (
    DenseTensorShape,
    TensorShape,
    require_shape,
)


@register_node("event_linear")
class EventLinearNode(BaseNode):
    """Fan out scalar corrections using ordinary registered linear parameters."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """Event connection output size."""

        out_features: int
        scalar_input: bool = False
        """Convert tensor values to source corrections at the start of each window."""

        def validate(self) -> None:
            """Require a positive output size."""
            require_positive("out_features", self.out_features)

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Allocate weights without owning receiver dynamics or state."""
        super().__init__(cfg, input_shape)
        shape = require_shape(
            "Node 'event_linear' input",
            input_shape,
            DenseTensorShape,
            event_stream=not cfg.scalar_input,
        )
        self._layer = nn.Linear(shape.features, cfg.out_features, bias=False)
        self._output_shape = TensorShape.dense(cfg.out_features, event_stream=True)

    @property
    def output_shape(self) -> TensorShape:
        """Return the addressed output channel count."""
        return self._output_shape

    def initialize_parameters(self) -> None:
        """Use the same initialization as the standard linear node."""
        nn.init.kaiming_normal_(self._layer.weight, mode="fan_in", nonlinearity="relu")

    def initial_state(self, inputs: torch.Tensor | EventPacket) -> torch.Tensor | None:
        """Retain the previous scalar input for conversion to source corrections."""
        return torch.zeros_like(inputs) if self._cfg.scalar_input else None

    def reset_state(self, state: torch.Tensor | None, dones: torch.Tensor) -> torch.Tensor | None:
        """Clear completed scalar sources, matching receiver reset to zero."""
        return state.masked_fill(dones.reshape(-1, 1), 0) if state is not None else None

    def forward(
        self,
        inputs: torch.Tensor | EventPacket,
        state: torch.Tensor | None = None,
    ) -> tuple[EventPacket, torch.Tensor | None]:
        """Route source corrections and attach their weight indices and weight matrix."""
        if self._cfg.scalar_input:
            previous = self.initial_state(inputs) if state is None else state
            corrections = (inputs.detach() - previous).flatten()
            changed = corrections != 0
            increments = corrections[changed]
            incoming = Events(
                inputs.shape[0],
                torch.cat((changed.new_zeros(1, dtype=torch.int64), changed.cumsum(0))),
                torch.zeros_like(increments),
                increments,
            )
            autograd_proxy = inputs
            next_state = inputs.detach().clone()
        else:
            if not isinstance(inputs, EventPacket) or inputs.connection is not None:
                msg = "EventLinear requires a neuron event output"
                raise TypeError(msg)
            incoming, autograd_proxy = inputs.events, inputs.autograd_proxy
            next_state = None
        return self._connect(incoming, autograd_proxy), next_state

    def _connect(self, incoming: Events, autograd_proxy: torch.Tensor | None) -> EventPacket:
        """Route source corrections and retain the spatial gradient path."""
        weight = self._layer.weight
        if (
            incoming.times.device.type != "cpu"
            or incoming.times.dtype not in (torch.float32, torch.float64)
            or incoming.times.dtype != weight.dtype
        ):
            msg = "EventLinear requires CPU float32 or float64 inputs matching its weights"
            raise ValueError(msg)
        offsets, times, increments, weight_indices = (
            torch.from_numpy(value)
            for value in execute_linear(
                incoming.offsets.detach().numpy(),
                incoming.times.detach().numpy(),
                incoming.increments.detach().numpy(),
                incoming.batch_size,
                (self._layer.out_features, self._layer.in_features),
            )
        )
        events = Events(incoming.batch_size, offsets, times, increments)
        if autograd_proxy is not None and torch.is_grad_enabled() and autograd_proxy.requires_grad:
            autograd_proxy = autograd_proxy @ weight.detach().T
        else:
            autograd_proxy = None
        return EventPacket(
            events,
            autograd_proxy=autograd_proxy,
            connection=ConnectionContext(
                weight_indices=weight_indices,
                weights=weight,
            ),
        )
