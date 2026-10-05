"""Ysskar neuron model with analytic free flow and step-integrated charge."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, NamedTuple

import torch

from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node
from spiking_rl_lab.networks.state import reset_state_rows

if TYPE_CHECKING:
    from spiking_rl_lab.networks.shape import TensorShape


class YsskarState(NamedTuple):
    """Unwrapped phase and signed synaptic current carried between steps."""

    phi: torch.Tensor
    current: torch.Tensor


@register_node("ysskar")
class YsskarNeuron(BaseNode):
    """Inject weighted input charge, advance analytically, and return charge."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """Physical times and smooth output profile configuration."""

        dt: float = 0.01
        """Duration of one step."""

        tau: float = 0.03
        """Dynamics time scale, current decays with tau / 2."""

        learnable_tau: bool = False
        """Learn a separate time scale for each neuron."""

        width: float = 0.03
        """Half the charge lies within +/- 2 * atan(width) of the spike center."""

        learnable_width: bool = True
        """Enable learning of the charge width."""

        asymmetry: float = 0.8
        """Positive values favor charge before the spike center."""

        compile: bool = False
        """Compile this node's forward and backward with TorchInductor on first use."""

        def validate(self) -> None:
            """Require finite times and a normalized positive output profile."""
            for name in ("dt", "tau"):
                value = getattr(self, name)
                if not math.isfinite(value) or value <= 0:
                    msg = f"{name} must be finite and positive"
                    raise ValueError(msg)
            if not 0 < self.width <= 1:
                msg = "width must satisfy 0 < width <= 1"
                raise ValueError(msg)
            if self.learnable_width and self.width == 1:
                msg = "Learnable width requires 0 < width < 1"
                raise ValueError(msg)
            if not 0 <= self.asymmetry < 1:
                msg = "asymmetry must satisfy 0 <= asymmetry < 1"
                raise ValueError(msg)

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize neuron parameters and optionally compile the node in place."""
        super().__init__(cfg, input_shape)
        self.register_parameter("_log_tau", None)
        if cfg.learnable_tau:
            tau_shape = (int(input_shape.dims[0]),) + (1,) * (len(input_shape.dims) - 1)
            self._log_tau = torch.nn.Parameter(torch.full(tau_shape, math.log(cfg.tau)))
        self.register_parameter("_width_logit", None)
        if cfg.learnable_width:
            self._width_logit = torch.nn.Parameter(torch.logit(torch.tensor(cfg.width)))
        if cfg.compile:
            self.compile(backend="inductor", fullgraph=True, dynamic=False)

    @property
    def tau(self) -> torch.Tensor | float:
        """Return the positive dynamics time scale."""
        return self._log_tau.exp() if self._log_tau is not None else self._cfg.tau

    @property
    def width(self) -> torch.Tensor | float:
        """Return profile width constrained to the unit interval."""
        return self._width_logit.sigmoid() if self._width_logit is not None else self._cfg.width

    def width_loss(self, target: float) -> torch.Tensor:
        """Penalize learnable width above the target."""
        if self._width_logit is None:
            return torch.zeros(())
        return (self.width.log() - math.log(target)).clamp_min(0).square()

    @property
    def output_shape(self) -> TensorShape:
        """Preserve the input tensor shape."""
        return self._input_shape

    def initial_state(self, inputs: torch.Tensor) -> YsskarState:
        """Create resting phase and current on the input device and dtype."""
        return YsskarState(torch.zeros_like(inputs), torch.zeros_like(inputs))

    def reset_state(self, state: YsskarState | None, dones: torch.Tensor) -> YsskarState | None:
        """Reset only completed batch rows to rest."""
        if state is None:
            return None
        return reset_state_rows(state, self.initial_state(state.phi), dones)

    def normalize_state(self, state: YsskarState | None) -> YsskarState | None:
        """Wrap phase to the first circle while preserving current."""
        if state is None:
            return None
        return YsskarState(torch.remainder(state.phi, 2 * math.pi), state.current)

    def _primitive(self, phi: torch.Tensor) -> torch.Tensor:
        """Evaluate the continuous unwrapped unit-charge primitive."""
        theta = phi - 1.5 * math.pi
        turns = torch.floor((theta + math.pi) / (2 * math.pi))
        half = (theta - turns * (2 * math.pi)) / 2
        width = torch.as_tensor(self.width, device=phi.device, dtype=phi.dtype)
        base = turns + torch.atan2(half.sin(), width * half.cos()) / math.pi
        complement = 1 - width.square()
        ratio = (half.sin() / width).square()
        denominator = torch.where(complement == 0, torch.ones_like(complement), complement)
        skew = torch.where(complement == 0, ratio, torch.log1p(complement * ratio) / denominator)
        return base - self._cfg.asymmetry * width * skew / math.pi

    def _advance(self, phi: torch.Tensor, current: torch.Tensor) -> YsskarState:
        """Advance phase and current analytically, including spike crossings."""
        half = phi / 2 - math.pi / 4
        sine, cosine = half.sin(), half.cos()
        numerator = (sine + (1 + current) * cosine) / 2
        tau = torch.as_tensor(self.tau, device=phi.device, dtype=phi.dtype)
        elapsed = self._cfg.dt / tau
        decay = torch.exp(-elapsed)
        decay_delta = torch.expm1(-elapsed)
        next_denominator = cosine + decay_delta * numerator
        next_current = current * decay**2
        next_sine = 2 * decay * numerator - (1 + next_current) * next_denominator
        # Use the same chart signs for the phase increment and crossing count.
        initial_sign = torch.where(cosine >= 0, 1.0, -1.0)
        final_sign = torch.where(next_denominator >= 0, 1.0, -1.0)
        initial_angle = torch.atan2(sine * initial_sign, cosine * initial_sign)
        final_angle = torch.atan2(next_sine * final_sign, next_denominator * final_sign)
        crossing = (
            (numerator * cosine > 0)
            & (numerator.abs() > cosine.abs())
            & (initial_sign != final_sign)
        )
        next_phi = phi + 2 * (final_angle - initial_angle) + 2 * math.pi * crossing.to(phi.dtype)
        return YsskarState(next_phi, next_current)

    def forward(
        self, inputs: torch.Tensor, state: YsskarState | None = None
    ) -> tuple[torch.Tensor, YsskarState]:
        """Return integrated signed charge and the next explicit state."""
        if state is None:
            state = self.initial_state(inputs)
        next_state = self._advance(state.phi, state.current + inputs)
        charge = self._primitive(next_state.phi) - self._primitive(state.phi)
        return charge, next_state
