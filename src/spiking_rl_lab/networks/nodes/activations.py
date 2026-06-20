"""Activation node implementations."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import norse.torch as snn
import torch
from norse.torch.functional.leaky_integrator import LIParameters
from norse.torch.functional.lif import LIFParameters

from spiking_rl_lab.networks.nodes.base_node import BaseNode, BaseNodeCfg, ListState
from spiking_rl_lab.networks.nodes.register import register_node

if TYPE_CHECKING:
    from spiking_rl_lab.networks.shape import TensorShape


@dataclasses.dataclass(kw_only=True, slots=True)
class LIFNodeCfg(BaseNodeCfg):
    """LIF node configuration."""

    dt: float = 0.001
    tau_syn_inv: float = 200.0
    tau_mem_inv: float = 100.0
    v_leak: float = 0.0
    v_th: float = 1.0
    v_reset: float = 0.0
    method: str = "super"
    alpha: float = 100.0

    def parameters(self) -> LIFParameters:
        """Build Norse LIF parameters."""
        return LIFParameters(
            tau_syn_inv=torch.as_tensor(self.tau_syn_inv),
            tau_mem_inv=torch.as_tensor(self.tau_mem_inv),
            v_leak=torch.as_tensor(self.v_leak),
            v_th=torch.as_tensor(self.v_th),
            v_reset=torch.as_tensor(self.v_reset),
            method=self.method,
            alpha=torch.as_tensor(self.alpha),
        )


@register_node("lif")
class LIFNode(BaseNode[LIFNodeCfg]):
    """Leaky integrate-and-fire node."""

    def __init__(self, cfg: LIFNodeCfg, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        self._cell = snn.LIFCell(p=cfg.parameters(), dt=cfg.dt)

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState]:
        """Run the LIF cell for one step."""
        spikes, next_state = self._cell(inputs, state=state)
        return spikes, next_state


@dataclasses.dataclass(kw_only=True, slots=True)
class LINodeCfg(BaseNodeCfg):
    """LI node configuration."""

    dt: float = 0.001
    tau_syn_inv: float = 200.0
    tau_mem_inv: float = 100.0
    v_leak: float = 0.0

    def parameters(self) -> LIParameters:
        """Build Norse LI parameters."""
        return LIParameters(
            tau_syn_inv=torch.as_tensor(self.tau_syn_inv),
            tau_mem_inv=torch.as_tensor(self.tau_mem_inv),
            v_leak=torch.as_tensor(self.v_leak),
        )


@register_node("li")
class LINode(BaseNode[LINodeCfg]):
    """Leaky integrator node."""

    def __init__(self, cfg: LINodeCfg, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        self._cell = snn.LICell(p=cfg.parameters(), dt=cfg.dt)

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState]:
        """Run the LI cell for one step."""
        outputs, next_state = self._cell(inputs, state=state)
        return outputs, next_state


@dataclasses.dataclass(kw_only=True, slots=True)
class ReLUNodeCfg(BaseNodeCfg):
    """ReLU node configuration."""


@register_node("relu")
class ReLUNode(BaseNode[ReLUNodeCfg]):
    """ReLU activation node."""

    def __init__(self, cfg: ReLUNodeCfg, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        self._activation = torch.nn.ReLU()

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Run the ReLU activation."""
        return self._activation(inputs), state


@dataclasses.dataclass(kw_only=True, slots=True)
class SiLUNodeCfg(BaseNodeCfg):
    """SiLU node configuration."""


@register_node("silu")
class SiLUNode(BaseNode[SiLUNodeCfg]):
    """SiLU activation node."""

    def __init__(self, cfg: SiLUNodeCfg, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        self._activation = torch.nn.SiLU()

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Run the SiLU activation."""
        return self._activation(inputs), state
