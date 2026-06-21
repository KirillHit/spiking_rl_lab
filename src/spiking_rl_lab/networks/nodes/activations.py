"""Activation node implementations."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar, Literal

import norse.torch as snn
import torch
from norse.torch.functional.leaky_integrator import LIParameters
from norse.torch.functional.lif import LIFParameters

from spiking_rl_lab.core.exception import NetworkCreationError
from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node

if TYPE_CHECKING:
    from spiking_rl_lab.networks.base_network import ListState
    from spiking_rl_lab.networks.shape import TensorShape


@register_node("lif")
class LIFNode(BaseNode):
    """Leaky integrate-and-fire node."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
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

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        self._cell = snn.LIFCell(p=cfg.parameters(), dt=cfg.dt)

    @property
    def output_shape(self) -> TensorShape:
        """Return output shape."""
        return self._input_shape

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState]:
        """Run the LIF cell for one step."""
        spikes, next_state = self._cell(inputs, state=state)
        return spikes, next_state


@register_node("li")
class LINode(BaseNode):
    """Leaky integrator node."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
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

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        self._cell = snn.LICell(p=cfg.parameters(), dt=cfg.dt)

    @property
    def output_shape(self) -> TensorShape:
        """Return output shape."""
        return self._input_shape

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState]:
        """Run the LI cell for one step."""
        outputs, next_state = self._cell(inputs, state=state)
        return outputs, next_state


@register_node("torch_activation")
class TorchActivationNode(BaseNode):
    """Parameter-free torch activation node."""

    activations: ClassVar[dict[str, type[torch.nn.Module]]] = {
        "relu": torch.nn.ReLU,
        "silu": torch.nn.SiLU,
        "sigmoid": torch.nn.Sigmoid,
        "tanh": torch.nn.Tanh,
    }

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
        """Torch activation node configuration."""

        activation: Literal["relu", "silu", "sigmoid", "tanh"]

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize the node."""
        super().__init__(cfg, input_shape)
        activation_cls = self.activations.get(cfg.activation)
        if activation_cls is None:
            available = ", ".join(sorted(self.activations))
            msg = f"Unsupported torch activation '{cfg.activation}'. Available: {available}"
            raise NetworkCreationError(msg)
        self._activation = activation_cls()

    @property
    def output_shape(self) -> TensorShape:
        """Return output shape."""
        return self._input_shape

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Run the torch activation."""
        return self._activation(inputs), None
