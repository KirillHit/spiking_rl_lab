"""Spiking activation node implementations."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import norse.torch as snn
import torch
from norse.torch.functional.leaky_integrator_box import LIBoxParameters, LIBoxState
from norse.torch.functional.lif_box import LIFBoxFeedForwardState, LIFBoxParameters

from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node

if TYPE_CHECKING:
    from spiking_rl_lab.networks.shape import TensorShape
    from spiking_rl_lab.networks.state import ListState


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

        dt: float = 0.001
        tau_mem_inv: float = 400.0
        learnable_tau: bool = True
        shared_tau: bool = False
        v_leak: float = 0.0
        v_th: float = 1.0
        v_reset: float = 0.0
        method: str = "super"
        alpha: float = 100.0

        def validate(self) -> None:
            """Validate the time step and learnable tau initialization."""
            if not self.dt > 0.0:
                msg = "dt must be positive"
                raise ValueError(msg)
            if self.learnable_tau and not 0.0 < self.dt * self.tau_mem_inv < 1.0:
                msg = "Learnable tau requires 0 < dt * tau_mem_inv < 1"
                raise ValueError(msg)

        def parameters(self) -> LIFBoxParameters:
            """Build Norse LIF box parameters."""
            return LIFBoxParameters(
                tau_mem_inv=torch.as_tensor(self.tau_mem_inv),
                v_leak=torch.as_tensor(self.v_leak),
                v_th=torch.as_tensor(self.v_th),
                v_reset=torch.as_tensor(self.v_reset),
                method=self.method,
                alpha=torch.as_tensor(self.alpha),
            )

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        self._cell = snn.LIFBoxCell(p=cfg.parameters(), dt=cfg.dt)
        self.register_parameter("_tau_logit", None)
        if cfg.learnable_tau:
            k = cfg.dt * cfg.tau_mem_inv
            tau_shape = (
                ()
                if cfg.shared_tau
                else (int(input_shape.dims[0]),) + (1,) * (len(input_shape.dims) - 1)
            )
            self._tau_logit = torch.nn.Parameter(torch.logit(torch.full(tau_shape, k)))

    @property
    def output_shape(self) -> TensorShape:
        """Return output shape."""
        return self._input_shape

    def initial_state(self, inputs: torch.Tensor) -> LIFBoxFeedForwardState:
        """Create the LIF cell's resting state for ``inputs``."""
        return self._cell.initial_state(inputs)

    def reset_state(
        self, state: LIFBoxFeedForwardState | None, dones: torch.Tensor
    ) -> LIFBoxFeedForwardState | None:
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
        if self._tau_logit is not None:
            self._cell.p = self._cell.p._replace(
                tau_mem_inv=self._tau_logit.sigmoid() / self._cell.dt,
            )
        spikes, next_state = self._cell(inputs, state=state)
        return spikes, next_state


@register_node("li")
class LINode(BaseNode):
    """Leaky integrator node."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """LI node configuration."""

        dt: float = 0.001
        tau_mem_inv: float = 200.0
        learnable_tau: bool = True
        shared_tau: bool = False
        v_leak: float = 0.0

        def validate(self) -> None:
            """Validate the time step and learnable tau initialization."""
            if not self.dt > 0.0:
                msg = "dt must be positive"
                raise ValueError(msg)
            if self.learnable_tau and not 0.0 < self.dt * self.tau_mem_inv < 1.0:
                msg = "Learnable tau requires 0 < dt * tau_mem_inv < 1"
                raise ValueError(msg)

        def parameters(self) -> LIBoxParameters:
            """Build Norse LI box parameters."""
            return LIBoxParameters(
                tau_mem_inv=torch.as_tensor(self.tau_mem_inv),
                v_leak=torch.as_tensor(self.v_leak),
            )

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        self._cell = snn.LIBoxCell(p=cfg.parameters(), dt=cfg.dt)
        self.register_parameter("_tau_logit", None)
        if cfg.learnable_tau:
            k = cfg.dt * cfg.tau_mem_inv
            tau_shape = (
                ()
                if cfg.shared_tau
                else (int(input_shape.dims[0]),) + (1,) * (len(input_shape.dims) - 1)
            )
            self._tau_logit = torch.nn.Parameter(torch.logit(torch.full(tau_shape, k)))

    @property
    def output_shape(self) -> TensorShape:
        """Return output shape."""
        return self._input_shape

    def initial_state(self, inputs: torch.Tensor) -> LIBoxState:
        """Create the leaky integrator's resting state for ``inputs``."""
        return self._cell.initial_state(inputs)

    def reset_state(self, state: LIBoxState | None, dones: torch.Tensor) -> LIBoxState | None:
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
        if self._tau_logit is not None:
            self._cell.p = self._cell.p._replace(
                tau_mem_inv=self._tau_logit.sigmoid() / self._cell.dt,
            )
        outputs, next_state = self._cell(inputs, state=state)
        return outputs, next_state
