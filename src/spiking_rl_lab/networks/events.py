"""Event streams and their connection context.

Tensor dimensions: B = batch size, N = neuron count, E = total event count,
K = number of incoming weights per neuron.
Connected events carry source corrections, applied with weights[n, weight_indices[e]].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class Events:
    """Corrections grouped by batch and neuron, ordered by time within each group."""

    batch_size: int
    """B, including batch elements without events."""
    offsets: torch.Tensor
    """[B*N+1], int64 boundaries of each neuron's event slice."""
    times: torch.Tensor
    """[E], times relative to step start in [0, duration], including this step's endpoint."""
    increments: torch.Tensor
    """[E], signed source signal corrections before applying connection weights."""

    @classmethod
    def empty(cls, prototype: torch.Tensor, *, batch_size: int, neuron_count: int) -> Events:
        """Create empty neuron slices using the signal dtype and device."""
        return cls(
            batch_size,
            torch.zeros(batch_size * neuron_count + 1, dtype=torch.int64, device=prototype.device),
            prototype.new_empty(0),
            prototype.new_empty(0),
        )


@dataclass(frozen=True, slots=True)
class ConnectionContext:
    """Connection data in the same event order as the accompanying Events object."""

    weight_indices: torch.Tensor
    """[E], int64 column indices in weights for each delivered event."""
    weights: torch.Tensor
    """[N,K], connection weights retaining the autograd path to parameters."""


@dataclass(frozen=True, slots=True)
class EventPacket:
    """Event stream passed between modules."""

    events: Events
    """Signal changes delivered to the next module."""
    autograd_proxy: torch.Tensor | None = None
    """[B,N], tensor carrying gradients between modules."""
    connection: ConnectionContext | None = None
    """Attached by a connection and consumed by the receiving neuron."""
