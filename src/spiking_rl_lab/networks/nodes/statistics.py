"""Node-owned accumulators for regularization and metrics."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from spiking_rl_lab.networks.nodes.base_node import BaseNode

if TYPE_CHECKING:
    from collections.abc import Generator

    import torch

    from spiking_rl_lab.networks.node_network import NodeNetwork
    from spiking_rl_lab.networks.state import ListState


@dataclass
class NodeStatistics:
    """A differentiable node penalty and detached scalar metrics."""

    loss: torch.Tensor | None
    metrics: dict[str, torch.Tensor]


class NodeStatisticsAccumulator(Protocol):
    """Collect outputs for one sequence without modifying the node's state."""

    def update(self, outputs: torch.Tensor) -> None:
        """Accumulate a batch of outputs while preserving its gradient graph."""
        ...

    def result(self) -> NodeStatistics | None:
        """Return collected statistics, or None if no outputs were collected."""
        ...


@contextmanager
def collect_node_statistics(
    network: NodeNetwork, *, progress: float, collect_metrics: bool = False
) -> Generator[dict[str, NodeStatistics], None, None]:
    """Collect node penalties and optional metrics, filling the result on exit.

    ``progress`` is the completed fraction of training in [0, 1].
    """
    statistics: dict[str, NodeStatistics] = {}
    accumulators: dict[str, NodeStatisticsAccumulator] = {}
    handles = []
    try:
        for name, node in network.named_modules():
            if not isinstance(node, BaseNode):
                continue
            accumulator = node.create_statistics(progress=progress, collect_metrics=collect_metrics)
            if accumulator is None:
                continue
            accumulators[name] = accumulator

            def collect(
                _node: BaseNode,
                _inputs: tuple[object, ...],
                output: tuple[torch.Tensor, ListState | None],
                *,
                accumulator: NodeStatisticsAccumulator = accumulator,
            ) -> None:
                accumulator.update(output[0])

            handles.append(node.register_forward_hook(collect))
        yield statistics
    finally:
        for handle in handles:
            handle.remove()

    for name, accumulator in accumulators.items():
        result = accumulator.result()
        if result is not None:
            statistics[name] = result
