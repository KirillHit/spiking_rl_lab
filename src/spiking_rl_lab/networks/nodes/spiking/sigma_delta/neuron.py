"""Sequential event processing for one neuron and its lazily advanced traces."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numba import njit

from spiking_rl_lab.networks.nodes.spiking.sigma_delta.dynamics import (
    NeuronParameters,
    advance,
    exponential_decay,
    next_event,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ConnectionEvents(NamedTuple):
    """NumPy views of receiver slices, source corrections and connection weights."""

    offsets: NDArray
    times: NDArray
    increments: NDArray
    weight_indices: NDArray
    weights: NDArray


class NeuronBuffers(NamedTuple):
    """Writable next-state views: [B*N] scalars and [B*N,K] connection arrays."""

    potential: NDArray
    transmitted: NDArray
    drive: NDArray
    source_levels: NDArray
    weight_trace: NDArray
    tau_trace: NDArray


@njit(cache=True)
def append_event(
    times: NDArray, increments: NDArray, count: int, time: float, increment: float
) -> tuple:
    """Grow a private output buffer geometrically as needed."""
    if count == times.size:
        capacity = max(2, times.size * 2)
        grown_times = np.empty(capacity, dtype=times.dtype)
        grown_increments = np.empty(capacity, dtype=increments.dtype)
        grown_times[:count], grown_increments[:count] = times, increments
        times, increments = grown_times, grown_increments
    times[count], increments[count] = time, increment
    return times, increments


@njit(cache=True, inline="always")
def receive_event(  # noqa: PLR0917
    events: ConnectionEvents,
    state: NeuronBuffers,
    row: int,
    cursor: int,
    trace_times: NDArray,
    tau: float,
) -> float:
    """Advance the affected weight trace, update its source and return the drive correction."""
    source = events.weight_indices[cursor]
    time = events.times[cursor]
    state.weight_trace[row, source] = exponential_decay(
        state.weight_trace[row, source],
        state.source_levels[row, source],
        (time - trace_times[source]) / tau,
    )
    trace_times[source] = time
    increment = events.increments[cursor]
    state.source_levels[row, source] += increment
    return increment * events.weights[row % events.weights.shape[0], source]


@njit(cache=True, inline="always")
def finish_weight_traces(
    state: NeuronBuffers, row: int, trace_times: NDArray, parameters: NeuronParameters
) -> None:
    """Synchronize all weight traces to the end of the window."""
    for source in range(state.source_levels.shape[1]):
        state.weight_trace[row, source] = exponential_decay(
            state.weight_trace[row, source],
            state.source_levels[row, source],
            (parameters.duration - trace_times[source]) / parameters.tau,
        )


@njit
def execute_neuron(
    events: ConnectionEvents,
    row: int,
    state: NeuronBuffers,
    parameters: NeuronParameters,
    empty: NDArray,
) -> tuple:
    """Merge input deliveries and own corrections, then synchronize endpoint traces."""
    potentials, transmitted, drives, source_levels, _, tau_traces = state
    sources, drive = source_levels[row], drives[row]
    potential, level, tau_trace = potentials[row], transmitted[row], tau_traces[row]
    tau, quantization_step, end, _, scalar_output, relu = parameters
    now = events.times.dtype.type(0)
    trace_times = np.full(sources.size, now, dtype=events.times.dtype) if sources.size else empty
    cursor, stop = events.offsets[row], events.offsets[row + 1]
    infinity = np.float32(np.inf)
    prediction = infinity if scalar_output else next_event(potential, level, drive, now, parameters)
    times, increments = empty, empty
    count = 0
    while True:
        delivery = events.times[cursor] if cursor < stop else infinity
        target = min(delivery, prediction)
        if target > end:
            break
        potential, tau_trace = advance(potential, tau_trace, drive, tau, target - now)
        now = target
        if delivery <= prediction:
            drive += receive_event(events, state, row, cursor, trace_times, tau)
            cursor += 1
            if delivery == prediction:
                # Preserve the scheduled output event at the same time.
                continue
        else:
            previous = level
            signal = max(potential, np.float32(0)) if relu else potential
            level += np.copysign(quantization_step, signal - level)
            assert level != previous, "Quantization step is below signal resolution"  # noqa: S101
            times, increments = append_event(times, increments, count, now, level - previous)
            count += 1
        prediction = (
            infinity if scalar_output else next_event(potential, level, drive, now, parameters)
        )
    potential, tau_trace = advance(potential, tau_trace, drive, tau, end - now)
    finish_weight_traces(state, row, trace_times, parameters)
    potentials[row], transmitted[row], drives[row], tau_traces[row] = (
        potential,
        level,
        drive,
        tau_trace,
    )
    return (times[:count], increments[:count]) if count else (empty, empty)
