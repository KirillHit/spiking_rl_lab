"""Endpoint local learning on Torch, without temporal backpropagation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.autograd.function import once_differentiable

if TYPE_CHECKING:
    from torch.autograd.function import FunctionCtx

    from spiking_rl_lab.networks.nodes.spiking.sigma_delta.execution import ExecutionConfig
    from spiking_rl_lab.networks.nodes.spiking.sigma_delta.state import SigmaDeltaState


class SigmaDeltaFunction(torch.autograd.Function):
    """Use local traces and activation derivatives, treating quantization as identity."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        input_duration: float | None,
        _autograd_proxy: torch.Tensor | None,
        _weights: torch.Tensor | None,
        log_tau: torch.Tensor,
        _log_h: torch.Tensor,
        state: SigmaDeltaState,
        config: ExecutionConfig,
    ) -> torch.Tensor:
        """Expose the physical readout and retain weight and neuron sensitivities."""
        tau = log_tau.exp()
        potential = state.neuron.potential
        ctx.save_for_backward(tau, potential, state.neuron.transmitted, *state.traces)
        ctx.input_duration = input_duration
        ctx.scalar_output = config.scalar_output
        ctx.relu = config.relu
        if not config.scalar_output:
            return torch.zeros_like(potential)
        return potential.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, error: torch.Tensor, *_unused: torch.Tensor) -> tuple:
        """Use local sensitivities and a prescribed quantum-accuracy signal."""
        tau, potential, transmitted, traces, tau_trace = ctx.saved_tensors
        grad_log_h = None
        if ctx.needs_input_grad[4] and not ctx.scalar_output:
            # Reduce the quantum when an untransmitted correction would lower the loss.
            signal = potential.relu() if ctx.relu else potential
            residual = signal - transmitted
            grad_log_h = (-error * residual).clamp_min(0).sum(0)
        if ctx.relu:
            error = error * (potential > 0)
        grad_tau = (error * tau_trace).sum(0) * tau if ctx.needs_input_grad[3] else None
        grad_weights = (error.unsqueeze(-1) * traces).sum(0) if ctx.needs_input_grad[2] else None
        grad_autograd_proxy = error if ctx.needs_input_grad[1] else None
        if grad_autograd_proxy is not None and ctx.input_duration is not None:
            grad_autograd_proxy = grad_autograd_proxy * -torch.expm1(-ctx.input_duration / tau)
        return (
            None,
            grad_autograd_proxy,
            grad_weights,
            grad_tau,
            grad_log_h,
            None,
            None,
        )
