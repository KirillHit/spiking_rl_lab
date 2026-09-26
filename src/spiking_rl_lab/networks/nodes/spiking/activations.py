"""Spiking activation node implementations."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, NamedTuple

import torch
from norse.torch.functional.threshold import threshold

from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node

if TYPE_CHECKING:
    from spiking_rl_lab.networks.shape import TensorShape
    from spiking_rl_lab.networks.state import ListState


class MembraneState(NamedTuple):
    """Membrane voltage carried between steps by LIF and LI nodes."""

    v: torch.Tensor


def _reset_state_rows[StateT: tuple[torch.Tensor, ...]](
    state: StateT,
    initial_state: StateT,
    dones: torch.Tensor,
) -> StateT:
    """Replace completed batch rows with the node's initial state."""
    values = []
    for value, initial_value in zip(state, initial_state, strict=True):
        mask = dones.to(device=value.device, dtype=torch.bool).reshape(
            -1, *([1] * (value.ndim - 1))
        )
        values.append(torch.where(mask, initial_value, value))
    return type(state)(*values)


@register_node("lif")
class LIFNode(BaseNode):
    """Leaky integrate-and-fire node."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """LIF node configuration."""

        decay: float = 0.6
        learnable_decay: bool = True
        shared_decay: bool = False
        normalize_input: bool = False
        v_th: float = 1.0
        learnable_v_th: bool = False
        v_reset: float = 0.0
        method: str = "super"
        alpha: float = 100.0

        def validate(self) -> None:
            """Validate the memory retention coefficient initialization."""
            if not 0.0 <= self.decay <= 1.0:
                msg = "decay must be between 0 and 1"
                raise ValueError(msg)
            if self.learnable_decay and not 0.0 < self.decay < 1.0:
                msg = "Learnable decay requires 0 < decay < 1"
                raise ValueError(msg)
            if self.learnable_v_th and not self.v_th > self.v_reset + 1e-6:
                msg = "Learnable v_th requires v_th > v_reset + 1e-6"
                raise ValueError(msg)

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        self.register_parameter("_decay_logit", None)
        if cfg.learnable_decay:
            decay_shape = (
                ()
                if cfg.shared_decay
                else (int(input_shape.dims[0]),) + (1,) * (len(input_shape.dims) - 1)
            )
            self._decay_logit = torch.nn.Parameter(torch.logit(torch.full(decay_shape, cfg.decay)))
        self.register_parameter("_v_th_raw", None)
        if cfg.learnable_v_th:
            threshold_shape = (int(input_shape.dims[0]),) + (1,) * (len(input_shape.dims) - 1)
            gap = torch.full(threshold_shape, cfg.v_th - cfg.v_reset - 1e-6)
            self._v_th_raw = torch.nn.Parameter(gap + torch.log(-torch.expm1(-gap)))

    @property
    def output_shape(self) -> TensorShape:
        """Return output shape."""
        return self._input_shape

    def initial_state(self, inputs: torch.Tensor) -> MembraneState:
        """Create the LIF cell's resting state for ``inputs``."""
        return MembraneState(v=torch.zeros_like(inputs))

    def reset_state(self, state: MembraneState | None, dones: torch.Tensor) -> MembraneState | None:
        """Restore completed environments to the LIF resting state."""
        if state is None:
            return None
        return _reset_state_rows(state, self.initial_state(state.v), dones)

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState]:
        """Run the LIF cell for one step."""
        if state is None:
            state = self.initial_state(inputs)
        decay = self._decay_logit.sigmoid() if self._decay_logit is not None else self._cfg.decay
        drive = (1 - decay) * inputs if self._cfg.normalize_input else inputs
        voltage = decay * state.v + drive
        v_th = self._cfg.v_th
        if self._v_th_raw is not None:
            v_th = self._cfg.v_reset + torch.nn.functional.softplus(self._v_th_raw) + 1e-6
        spikes = threshold(voltage - v_th, self._cfg.method, self._cfg.alpha)
        voltage = (1 - spikes) * voltage + spikes * self._cfg.v_reset
        return spikes, MembraneState(v=voltage)


@register_node("li")
class LINode(BaseNode):
    """Leaky integrator node."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """LI node configuration."""

        decay: float = 0.8
        learnable_decay: bool = True
        shared_decay: bool = False
        normalize_input: bool = False

        def validate(self) -> None:
            """Validate the memory retention coefficient initialization."""
            if not 0.0 <= self.decay <= 1.0:
                msg = "decay must be between 0 and 1"
                raise ValueError(msg)
            if self.learnable_decay and not 0.0 < self.decay < 1.0:
                msg = "Learnable decay requires 0 < decay < 1"
                raise ValueError(msg)

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        self.register_parameter("_decay_logit", None)
        if cfg.learnable_decay:
            decay_shape = (
                ()
                if cfg.shared_decay
                else (int(input_shape.dims[0]),) + (1,) * (len(input_shape.dims) - 1)
            )
            self._decay_logit = torch.nn.Parameter(torch.logit(torch.full(decay_shape, cfg.decay)))

    @property
    def output_shape(self) -> TensorShape:
        """Return output shape."""
        return self._input_shape

    def initial_state(self, inputs: torch.Tensor) -> MembraneState:
        """Create the leaky integrator's resting state for ``inputs``."""
        return MembraneState(v=torch.zeros_like(inputs))

    def reset_state(self, state: MembraneState | None, dones: torch.Tensor) -> MembraneState | None:
        """Restore completed environments to the integrator resting state."""
        if state is None:
            return None
        return _reset_state_rows(state, self.initial_state(state.v), dones)

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState]:
        """Run the LI cell for one step."""
        if state is None:
            state = self.initial_state(inputs)
        decay = self._decay_logit.sigmoid() if self._decay_logit is not None else self._cfg.decay
        drive = (1 - decay) * inputs if self._cfg.normalize_input else inputs
        voltage = decay * state.v + drive
        return voltage, MembraneState(v=voltage)
