"""Outgoing event counts and their quantum-dependent activity penalty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiking_rl_lab.networks.nodes.statistics import NodeStatistics

if TYPE_CHECKING:
    import torch

    from spiking_rl_lab.networks.events import EventPacket


@dataclass
class ActivityStatistics:
    """Average each neuron's outgoing event count over batch elements and windows."""

    neuron_count: int
    activity_scale: float | None
    collect_metrics: bool
    log_h: torch.Tensor
    event_total: torch.Tensor | None = None
    sample_count: int = 0

    def update(self, outputs: EventPacket) -> None:
        """Count output events directly from the batch-by-neuron slice boundaries."""
        events = outputs.events
        counts = events.offsets.diff().reshape(events.batch_size, self.neuron_count).sum(0)
        self.event_total = counts if self.event_total is None else self.event_total + counts
        self.sample_count += events.batch_size

    def result(self) -> NodeStatistics | None:
        """Report measured cost, approximating d(rate)/d(log quantum) as -rate."""
        if self.event_total is None or not self.sample_count:
            return None
        rates = self.event_total / self.sample_count
        loss = None
        if self.activity_scale is not None:
            values = rates * (self.log_h.detach() - self.log_h).exp()
            loss = self.activity_scale * values.sum()
        metrics = {}
        if self.collect_metrics:
            metrics = {
                "activity_mean": rates.mean(),
                "events_per_window": rates.sum(),
            }
        return NodeStatistics(loss=loss, metrics=metrics)
