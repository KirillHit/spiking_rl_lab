"""Network implementations."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import nn

from spiking_rl_lab.core.factory import (
    ConfiguredBase,
)
from spiking_rl_lab.network.nodes.register import build_node

if TYPE_CHECKING:
    from spiking_rl_lab.network.nodes import ListState
    from spiking_rl_lab.network.nodes.register import NetworkNodeCfg
    from spiking_rl_lab.network.shape import TensorShape


@dataclasses.dataclass(kw_only=True, slots=True)
class SequentialNetworkCfg:
    """Sequential network configuration."""

    init_weights: bool = True
    nodes: list[NetworkNodeCfg] = dataclasses.field(default_factory=list)


NetworkCfg = SequentialNetworkCfg


class BaseNetwork(nn.Module, ConfiguredBase, ABC):
    """Base class for configured networks."""

    Config: ClassVar[type[object]]

    def __init__(self, cfg: object) -> None:
        """Validate and store network configuration."""
        nn.Module.__init__(self)
        ConfiguredBase.__init__(self, cfg)

    @property
    @abstractmethod
    def output_shape(self) -> TensorShape:
        """Return network output shape."""


class Network(BaseNetwork):
    """Sequential network built from registered node classes."""

    Config: ClassVar[type[SequentialNetworkCfg]] = SequentialNetworkCfg

    def __init__(self, cfg: SequentialNetworkCfg, input_shape: TensorShape) -> None:
        """Build network nodes from ``cfg.nodes``."""
        super().__init__(cfg)
        self._net = nn.ModuleList()
        self._output_shape = input_shape

        for node_cfg in self._cfg.nodes:
            node = build_node(node_cfg, self._output_shape)
            self._net.append(node)
            self._output_shape = node.output_shape

        if self._cfg.init_weights:
            self._init_weights()

    @property
    def output_shape(self) -> TensorShape:
        """Return network output shape."""
        return self._output_shape

    def _init_weights(self) -> None:
        """Initialize standard trainable layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState]:
        """Run the network and return output with per-layer state."""
        state = [None] * len(self._net) if state is None else state
        for idx, layer in enumerate(self._net):
            inputs, state[idx] = layer(inputs, state[idx])
        return inputs, state
