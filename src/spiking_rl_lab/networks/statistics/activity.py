"""Collect and aggregate spike activity from network nodes."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from spiking_rl_lab.networks.nodes.spiking.activations import LIFNode

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping

    import torch
    from torch import nn

    from spiking_rl_lab.networks.node_network import NodeNetwork
    from spiking_rl_lab.networks.state import ListState


DEFAULT_SPIKING_NODE_TYPES = (LIFNode,)


def mean_spike_activity(
    _module: nn.Module,
    _inputs: tuple[object, ...],
    output: tuple[torch.Tensor, ListState],
    *,
    dim: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Return spike activity averaged over the selected dimensions, or all by default."""
    return output[0].mean(dim=dim)


def spike_activity_moments(
    layer_activity: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the mean and mean square over all neuron firing rates."""
    neuron_count = sum(activity.numel() for activity in layer_activity.values())
    mean = sum(activity.sum() for activity in layer_activity.values()) / neuron_count
    mean_square = (
        sum(activity.square().sum() for activity in layer_activity.values()) / neuron_count
    )
    return mean, mean_square


@contextmanager
def collect_forward_outputs(
    network: NodeNetwork,
    transform: Callable[[nn.Module, tuple[object, ...], object], torch.Tensor],
    node_types: tuple[type[nn.Module], ...] = DEFAULT_SPIKING_NODE_TYPES,
    *,
    detach: bool = False,
) -> Generator[dict[str, list[torch.Tensor]], None, None]:
    """Collect transformed outputs from matching network nodes during a forward pass.

    Set ``detach`` when the outputs are used only as metrics.
    """
    outputs: dict[str, list[torch.Tensor]] = {}
    handles = []
    for name, module in network.named_modules():
        if not isinstance(module, node_types):
            continue
        values: list[torch.Tensor] = []
        outputs[name] = values

        def collect(
            hooked_module: nn.Module,
            inputs: tuple[object, ...],
            output: object,
            *,
            values: list[torch.Tensor] = values,
        ) -> None:
            value = transform(hooked_module, inputs, output)
            values.append(value.detach() if detach else value)

        handles.append(module.register_forward_hook(collect))

    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()
