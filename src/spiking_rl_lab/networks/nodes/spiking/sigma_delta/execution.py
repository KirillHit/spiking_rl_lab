"""Torch/NumPy boundary and flat batch-by-neuron parallel execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numba
import numpy as np
import torch
from numba import njit, prange

from spiking_rl_lab.networks.events import Events
from spiking_rl_lab.networks.nodes.spiking.sigma_delta.dynamics import NeuronParameters
from spiking_rl_lab.networks.nodes.spiking.sigma_delta.neuron import (
    ConnectionEvents,
    NeuronBuffers,
    execute_neuron,
)
from spiking_rl_lab.networks.nodes.spiking.sigma_delta.state import (
    NeuronState,
    SigmaDeltaState,
    TraceState,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from spiking_rl_lab.networks.events import EventPacket


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Physical execution options, separate from trainable neuron parameters."""

    scalar_output: bool
    hysteresis: float
    threads: int | None
    chunk_size: int
    relu: bool


@torch.no_grad()
def execute(
    packet: EventPacket,
    state: SigmaDeltaState,
    log_tau: torch.Tensor,
    log_h: torch.Tensor,
    *,
    duration: float,
    config: ExecutionConfig,
) -> tuple[Events, SigmaDeltaState]:
    """Execute detached physical data and traces, without reading autograd context."""
    incoming = connection_events(packet)
    next_state, buffers = prepare_state(state)
    tau = log_tau.exp().numpy()
    scalar_type = tau.dtype.type
    numba.set_num_threads(
        numba.config.NUMBA_NUM_THREADS if config.threads is None else config.threads
    )
    with numba.parallel_chunksize(config.chunk_size):
        output = execute_batch(
            incoming,
            buffers,
            LayerParameters(
                tau,
                log_h.exp().numpy(),
                scalar_type(duration),
                scalar_type(config.hysteresis),
                config.scalar_output,
                config.relu,
            ),
        )
    offsets, times, increments = (torch.from_numpy(value) for value in output)
    return Events(packet.events.batch_size, offsets, times, increments), next_state


def connection_events(packet: EventPacket) -> ConnectionEvents:
    """Expose event and weight arrays as NumPy views, with empty scalar-input connections."""
    events, connection = packet.events, packet.connection
    times = events.times.detach().numpy()
    if connection is None:
        weight_indices = np.empty(0, dtype=np.int64)
        weights = np.empty((0, 0), dtype=times.dtype)
    else:
        weight_indices = connection.weight_indices.detach().numpy()
        weights = connection.weights.detach().numpy()
    return ConnectionEvents(
        events.offsets.detach().numpy(),
        times,
        events.increments.detach().numpy(),
        weight_indices,
        weights,
    )


def prepare_state(state: SigmaDeltaState) -> tuple[SigmaDeltaState, NeuronBuffers]:
    """Copy persistent state once and expose writable contiguous views to Numba."""
    next_state = SigmaDeltaState(
        NeuronState(
            *(value.detach().clone(memory_format=torch.contiguous_format) for value in state.neuron)
        ),
        TraceState(
            *(value.detach().clone(memory_format=torch.contiguous_format) for value in state.traces)
        ),
    )
    batch_size, neuron_count = next_state.neuron.potential.shape
    row_count = batch_size * neuron_count
    weight_count = next_state.neuron.source_levels.shape[-1]
    buffers = NeuronBuffers(
        next_state.neuron.potential.numpy().reshape(row_count),
        next_state.neuron.transmitted.numpy().reshape(row_count),
        next_state.neuron.drive.numpy().reshape(row_count),
        next_state.neuron.source_levels.numpy().reshape(row_count, weight_count),
        next_state.traces.weight.numpy().reshape(row_count, weight_count),
        next_state.traces.tau.numpy().reshape(row_count),
    )
    return next_state, buffers


class LayerParameters(NamedTuple):
    """Per-neuron arrays and common window options passed to Numba."""

    tau: NDArray
    quantization_step: NDArray
    duration: float
    hysteresis: float
    scalar_output: bool
    relu: bool


@njit(parallel=True)
def execute_batch(
    events: ConnectionEvents, state: NeuronBuffers, parameters: LayerParameters
) -> tuple:
    """Update private state buffers in one dynamic batch/neuron loop and pack emissions."""
    tau, quantization_step, duration, hysteresis, scalar_output, relu = parameters
    neuron_count = tau.size
    row_count = state.potential.size
    empty = np.empty(0, dtype=state.potential.dtype)
    outgoing = [(empty, empty) for _ in range(row_count)]
    for row in prange(row_count):
        index = np.int64(row)
        neuron = index % neuron_count
        outgoing[index] = execute_neuron(
            events,
            index,
            state,
            NeuronParameters(
                tau[neuron],
                quantization_step[neuron],
                duration,
                hysteresis,
                scalar_output,
                relu,
            ),
            empty,
        )
    return pack_events(outgoing, state.potential)


@njit(cache=True)
def pack_events(outgoing: list, prototype: NDArray) -> tuple:
    """Copy neuron streams to consecutive slices without merging their timelines."""
    offsets = np.zeros(len(outgoing) + 1, dtype=np.int64)
    for row in range(len(outgoing)):
        offsets[row + 1] = offsets[row] + outgoing[row][0].size
    times = np.empty(offsets[-1], dtype=prototype.dtype)
    increments = np.empty_like(times)
    for row in range(len(outgoing)):
        source_times, source_increments = outgoing[row]
        start = offsets[row]
        for index in range(source_times.size):
            times[start + index] = source_times[index]
            increments[start + index] = source_increments[index]
    return offsets, times, increments
