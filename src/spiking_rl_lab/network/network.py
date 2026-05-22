"""Network module assembled from registered nodes."""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

from spiking_rl_lab.network.nodes.register import build_node
from spiking_rl_lab.utils.exception import NetworkCreationError

if TYPE_CHECKING:
    from spiking_rl_lab.network.nodes import ListState
    from spiking_rl_lab.network.shape import TensorShape
    from spiking_rl_lab.utils.config import NetworkNodeCfg

log = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True)
class NetworkCfg:
    """Network configuration."""

    init_weights: bool = True
    net_arch: list[NetworkNodeCfg] = dataclasses.field(default_factory=list)


class Network(nn.Module):
    """Sequential network built from registered node classes."""

    def __init__(self, cfg: NetworkCfg, input_shape: TensorShape) -> None:
        """Build network nodes from ``cfg.net_arch``."""
        super().__init__()
        self._cfg = cfg
        self._net = nn.ModuleList()
        self._output_shape = input_shape

        for node_cfg in self._cfg.net_arch:
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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


def build_network(cfg: NetworkCfg) -> Network:
    """Build a network from configuration."""
    try:
        log.info("Creating network...")
        return Network(cfg=cfg)
    except NetworkCreationError:
        raise
    except Exception as exc:
        msg = "Failed to create network"
        raise NetworkCreationError(msg) from exc
