"""Base network abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch import nn

from spiking_rl_lab.core.factory import ConfiguredBase

if TYPE_CHECKING:
    import torch

    from spiking_rl_lab.networks.nodes.base_node import ListState
    from spiking_rl_lab.networks.shape import TensorShape


class BaseNetwork(nn.Module, ConfiguredBase, ABC):
    """Base class for configured networks."""

    def __init__(self, cfg: object) -> None:
        """Validate and store network configuration."""
        nn.Module.__init__(self)
        ConfiguredBase.__init__(self, cfg)

    @property
    @abstractmethod
    def output_shape(self) -> TensorShape:
        """Return network output shape."""

    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Run the network and return output with optional state."""
