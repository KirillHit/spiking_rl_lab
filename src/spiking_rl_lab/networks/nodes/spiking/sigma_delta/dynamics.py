"""Scalar SigmaDelta flow and exact prediction between input corrections."""

from typing import NamedTuple

import numpy as np
from numba import njit


class NeuronParameters(NamedTuple):
    """Scalar flow, threshold and window parameters for one neuron."""

    tau: float
    quantization_step: float
    duration: float
    hysteresis: float
    scalar_output: bool
    relu: bool


@njit(cache=True, inline="always")
def exponential_decay(value: float, target: float, scaled_time: float) -> float:
    """Approach a constant target, preserving small changes and decaying residuals."""
    if scaled_time <= np.float32(1):
        return value + (target - value) * -np.expm1(-scaled_time)
    return target + (value - target) * np.exp(-scaled_time)


@njit(cache=True)
def advance(
    potential: float, tau_trace: float, drive: float, tau: float, elapsed: float
) -> tuple[float, float]:
    """Advance potential and its local tau sensitivity at constant drive."""
    scaled = elapsed / tau
    decay = np.exp(-scaled)
    correction = potential - drive
    term = decay * scaled / tau * correction if decay != 0 else decay
    return exponential_decay(potential, drive, scaled), decay * tau_trace + term


@njit(cache=True)
def next_event(
    potential: float,
    transmitted: float,
    drive: float,
    time: float,
    parameters: NeuronParameters,
) -> float:
    """Predict the next correction, or infinity for an unreachable threshold."""
    tau, quantization_step, _, hysteresis, _, relu = parameters
    zero = np.float32(0)
    infinity = np.float32(np.inf)
    half_quantization_step = quantization_step * np.float32(0.5)
    threshold = half_quantization_step + hysteresis
    if threshold == half_quantization_step:
        threshold = np.nextafter(half_quantization_step, infinity)
    signal = max(potential, zero) if relu else potential
    if abs(signal - transmitted) >= threshold:
        return time
    gap = drive - potential
    direction = np.float32(1) if gap > zero else np.float32(-1)
    if relu and direction < zero and transmitted < threshold:
        return infinity
    distance = max(zero, threshold - direction * (potential - transmitted))
    remaining = abs(gap) - distance
    if gap == zero or remaining <= zero:
        return infinity
    fraction = distance / abs(gap)
    delay = (
        -tau * np.log1p(-fraction)
        if fraction < np.float32(0.5)
        else tau * (np.log(abs(gap)) - np.log(abs(remaining)))
    )
    predicted = time + delay
    if not np.isfinite(predicted):
        return infinity
    return np.nextafter(time, infinity) if distance > zero and predicted == time else predicted
