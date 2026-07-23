"""Standard activation node implementations."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar, Literal

import torch

from spiking_rl_lab.core.exception import NetworkCreationError
from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node

if TYPE_CHECKING:
    from spiking_rl_lab.networks.shape import TensorShape
    from spiking_rl_lab.networks.state import ListState


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
    class Config(BaseNode.Config):
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
