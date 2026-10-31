"""Explicit physical accumulators and separately named local-learning traces."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import torch


class NeuronState(NamedTuple):
    """Potential, transmitted level, input drive and accumulated source levels."""

    potential: torch.Tensor
    transmitted: torch.Tensor
    drive: torch.Tensor
    source_levels: torch.Tensor


class TraceState(NamedTuple):
    """Local sensitivities of potential to incoming weights and tau."""

    weight: torch.Tensor
    tau: torch.Tensor


class SigmaDeltaState(NamedTuple):
    """Physical and learning state, both preserved across execution windows."""

    neuron: NeuronState
    traces: TraceState


def initial_sigma_delta(
    shape: tuple[int, int], prototype: torch.Tensor, weight_count: int
) -> SigmaDeltaState:
    """Start from zero physical accumulators and learning traces."""
    zeros = prototype.new_zeros
    source_shape = (*shape, weight_count)
    return SigmaDeltaState(
        NeuronState(
            zeros(shape),
            zeros(shape),
            zeros(shape),
            zeros(source_shape),
        ),
        TraceState(zeros(source_shape), zeros(shape)),
    )


def reset_sigma_delta(state: SigmaDeltaState, dones: torch.Tensor) -> SigmaDeltaState:
    """Clear physical accumulators and learning traces for completed rows."""
    if not dones.any():
        return state
    dones = dones.reshape(-1)

    def clear(values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(
            value.masked_fill(dones.reshape(-1, *([1] * (value.ndim - 1))), 0) for value in values
        )

    return SigmaDeltaState(NeuronState(*clear(state.neuron)), TraceState(*clear(state.traces)))
