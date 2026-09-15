"""Coordinate normalization statistics across networks."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from spiking_rl_lab.networks.nodes.standard.normalization import BatchNormNode

if TYPE_CHECKING:
    from collections.abc import Generator

    from spiking_rl_lab.networks.node_network import NodeNetwork


def _batch_norm_nodes(networks: tuple[NodeNetwork, ...]) -> dict[BatchNormNode, None]:
    """Find unique batch-normalization nodes in the supplied networks."""
    return dict.fromkeys(
        node
        for network in networks
        for node in network.modules()
        if isinstance(node, BatchNormNode)
    )


def has_batch_norm(*networks: NodeNetwork) -> bool:
    """Return whether any supplied network contains batch normalization."""
    return bool(_batch_norm_nodes(networks))


@contextlib.contextmanager
def collect_batch_norm_statistics(*networks: NodeNetwork) -> Generator[None]:
    """Collect forward-pass statistics without applying them."""
    nodes = _batch_norm_nodes(networks)
    try:
        for node in nodes:
            node.start_batch_norm_statistics_collection()
        yield
    finally:
        for node in nodes:
            node.stop_batch_norm_statistics_collection()


def apply_collected_batch_norm_statistics(*networks: NodeNetwork) -> None:
    """Apply collected statistics in all batch-normalization nodes."""
    for node in _batch_norm_nodes(networks):
        node.apply_collected_batch_norm_statistics()
