"""Node-based network implementation."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import nn

from spiking_rl_lab.core.factory import ConfiguredBase
from spiking_rl_lab.networks.nodes.builder import NodeConfig, build_node

if TYPE_CHECKING:
    from spiking_rl_lab.networks.shape import TensorShape
    from spiking_rl_lab.networks.state import ListState


@dataclasses.dataclass(kw_only=True, slots=True)
class NodeNetworkConfig:
    """Configuration for a node-based network."""

    init_weights: bool = True
    nodes: list[NodeConfig] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        """Convert YAML node mappings to typed node configs."""
        self.nodes = [
            node if isinstance(node, NodeConfig) else NodeConfig(**node) for node in self.nodes
        ]


class NodeNetwork(nn.Module, ConfiguredBase):
    """Network built from configured nodes."""

    Config: ClassVar[type[NodeNetworkConfig]] = NodeNetworkConfig

    def __init__(self, cfg: NodeNetworkConfig, *, input_shape: TensorShape) -> None:
        """Build network nodes from ``cfg.nodes``."""
        nn.Module.__init__(self)
        ConfiguredBase.__init__(self, cfg)
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

    def initial_state(self, inputs: torch.Tensor) -> ListState:
        """Create one initial state per node for a batch of inputs."""
        state = []
        for layer in self._net:
            layer_state = layer.initial_state(inputs)
            state.append(layer_state)
            inputs, _ = layer(inputs, layer_state)
        return state

    def reset_state(self, state: ListState | None, dones: torch.Tensor) -> ListState | None:
        """Reset state rows for completed environments through their owning nodes."""
        if state is None:
            return None
        return [
            layer.reset_state(layer_state, dones)
            for layer, layer_state in zip(self._net, state, strict=True)
        ]

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState]:
        """Run the network without mutating the provided per-layer state."""
        previous_state = [None] * len(self._net) if state is None else state
        next_state = [None] * len(self._net)
        for idx, layer in enumerate(self._net):
            inputs, next_state[idx] = layer(inputs, previous_state[idx])
        return inputs, next_state
