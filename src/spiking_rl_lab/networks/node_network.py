"""Node-based network implementation."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from torch import nn

from spiking_rl_lab.networks.base_network import BaseNetwork, ListState
from spiking_rl_lab.networks.builder import register_network
from spiking_rl_lab.networks.nodes.builder import NetworkNodeCfg, build_node

if TYPE_CHECKING:
    from spiking_rl_lab.networks.shape import TensorShape


@register_network("node_graph")
class NodeNetwork(BaseNetwork):
    """Network built from configured nodes."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNetwork.Config):
        """Node-based network configuration."""

        init_weights: bool = True
        nodes: list[NetworkNodeCfg] = dataclasses.field(default_factory=list)

        def __post_init__(self) -> None:
            """Convert YAML node mappings to typed node configs."""
            self.nodes = [
                node if isinstance(node, NetworkNodeCfg) else NetworkNodeCfg(**node)
                for node in self.nodes
            ]

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
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
