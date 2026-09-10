"""Coordinate normalization statistics across networks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spiking_rl_lab.networks.nodes.standard.normalization import BatchNormNode

if TYPE_CHECKING:
    from spiking_rl_lab.networks.node_network import NodeNetwork


def _batch_norm_nodes(networks: tuple[NodeNetwork, ...]) -> dict[BatchNormNode, None]:
    """Find unique batch-normalization nodes in the supplied networks."""
    return dict.fromkeys(
        node
        for network in networks
        for node in network.modules()
        if isinstance(node, BatchNormNode)
    )


def enable_batch_norm_statistics_collection(*networks: NodeNetwork) -> None:
    """Enable statistics collection for all batch-normalization nodes."""
    for node in _batch_norm_nodes(networks):
        node.enable_batch_norm_statistics_collection()


def apply_collected_batch_norm_statistics(*networks: NodeNetwork) -> None:
    """Apply collected statistics in all batch-normalization nodes."""
    for node in _batch_norm_nodes(networks):
        node.apply_collected_batch_norm_statistics()
