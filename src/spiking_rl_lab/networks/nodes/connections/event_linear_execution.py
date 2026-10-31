"""Dense event routing directly into chronological receiver slices."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

if TYPE_CHECKING:
    from numpy.typing import NDArray


@njit(cache=True)
def linear_offsets(
    source_offsets: NDArray, batch_size: int, sources: int, receivers: int
) -> NDArray:
    """Every dense receiver gets all events from the sources of its batch."""
    offsets = np.zeros(batch_size * receivers + 1, dtype=np.int64)
    for batch in range(batch_size):
        count = source_offsets[(batch + 1) * sources] - source_offsets[batch * sources]
        for receiver in range(receivers):
            row = batch * receivers + receiver
            offsets[row + 1] = offsets[row] + count
    return offsets


@njit(parallel=True, cache=True)
def execute_linear(
    source_offsets: NDArray,
    times: NDArray,
    increments: NDArray,
    batch_size: int,
    weight_shape: tuple[int, int],
) -> tuple:
    """Merge each batch once, then copy the stream to contiguous receiver slices."""
    receivers, sources = weight_shape
    offsets = linear_offsets(source_offsets, batch_size, sources, receivers)
    output_times = np.empty(offsets[-1], dtype=times.dtype)
    output_increments = np.empty_like(output_times)
    weight_indices = np.empty(offsets[-1], dtype=np.int64)
    for batch in prange(batch_size):
        start = offsets[batch * receivers]
        cursor = start
        queue = [
            (times[source_offsets[row]], np.int64(row), source_offsets[row])
            for row in range(batch * sources, (batch + 1) * sources)
            if source_offsets[row] < source_offsets[row + 1]
        ]
        heapq.heapify(queue)
        while queue:
            time, source_row, index = heapq.heappop(queue)
            source = source_row % sources
            increment = increments[index]
            output_times[cursor] = time
            output_increments[cursor] = increment
            weight_indices[cursor] = source
            cursor += 1
            index += 1
            if index < source_offsets[source_row + 1]:
                heapq.heappush(queue, (times[index], source_row, index))
        count = cursor - start
        for receiver in range(1, receivers):
            target = start + receiver * count
            for index in range(count):
                output_times[target + index] = output_times[start + index]
                output_increments[target + index] = output_increments[start + index]
                weight_indices[target + index] = weight_indices[start + index]
    return offsets, output_times, output_increments, weight_indices
