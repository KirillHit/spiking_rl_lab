"""Leaky integrator with synaptic current and exact continuous-time dynamics."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, NamedTuple

import torch

from spiking_rl_lab.core.validation import require_positive
from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node
from spiking_rl_lab.networks.state import reset_state_rows

if TYPE_CHECKING:
    from spiking_rl_lab.networks.shape import TensorShape


class LICurrentState(NamedTuple):
    """Membrane potential v and synaptic current i carried between steps."""

    v: torch.Tensor
    i: torch.Tensor


@register_node("li_current")
class LICurrentNode(BaseNode):
    """Inject charge into i and evolve exactly, with tau_v / tau_i bounded away from 1."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """Trainable current time scale and bounded potential-to-current time-scale ratio."""

        dt: float = 0.05
        tau_i: float = 0.1
        learnable_tau: bool = True
        tau_ratio: float = 3.0
        tau_ratio_min: float = 1.1
        learnable_tau_ratio: bool = True
        compile: bool = False

        def validate(self) -> None:
            """Require positive durations and a ratio lower bound strictly greater than 1."""
            for name in ("dt", "tau_i", "tau_ratio_min", "tau_ratio"):
                value = getattr(self, name)
                if not math.isfinite(value):
                    msg = f"{name} must be finite"
                    raise ValueError(msg)
            require_positive("dt", self.dt)
            require_positive("tau_i", self.tau_i)
            require_positive("tau_ratio_min - 1", self.tau_ratio_min - 1)
            require_positive("tau_ratio - tau_ratio_min", self.tau_ratio - self.tau_ratio_min)

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize a current time scale and ratio per feature or channel."""
        super().__init__(cfg, input_shape)
        tau_shape = (int(input_shape.dims[0]),) + (1,) * (len(input_shape.dims) - 1)
        parameter = (
            torch.nn.Parameter(torch.full(tau_shape, math.log(cfg.tau_i)))
            if cfg.learnable_tau
            else None
        )
        self.register_parameter("_log_tau_i", parameter)
        ratio_offset = cfg.tau_ratio - cfg.tau_ratio_min
        ratio_raw = ratio_offset + math.log(-math.expm1(-ratio_offset))
        self.register_parameter(
            "_tau_ratio_raw",
            torch.nn.Parameter(torch.full(tau_shape, ratio_raw))
            if cfg.learnable_tau_ratio
            else None,
        )
        if cfg.compile:
            self.compile(backend="inductor", fullgraph=True, dynamic=False)

    @property
    def tau_i(self) -> torch.Tensor | float:
        """Return the positive input-filter time scale."""
        parameter = self._log_tau_i
        return parameter.exp() if parameter is not None else self._cfg.tau_i

    @property
    def tau_ratio(self) -> torch.Tensor | float:
        """Keep the ratio separated from 1 with a smooth positive offset."""
        parameter = self._tau_ratio_raw
        return (
            self._cfg.tau_ratio_min + torch.nn.functional.softplus(parameter)
            if parameter is not None
            else self._cfg.tau_ratio
        )

    @property
    def tau_v(self) -> torch.Tensor | float:
        """Return the slower potential time scale derived from the learned ratio."""
        return self.tau_ratio * self.tau_i

    @property
    def output_shape(self) -> TensorShape:
        """Preserve feature and batch layout."""
        return self._input_shape

    def initial_state(self, inputs: torch.Tensor) -> LICurrentState:
        """Start both signed states at zero."""
        return LICurrentState(torch.zeros_like(inputs), torch.zeros_like(inputs))

    def reset_state(
        self, state: LICurrentState | None, dones: torch.Tensor
    ) -> LICurrentState | None:
        """Reset only completed environments at episode boundaries."""
        if state is None:
            return None
        return reset_state_rows(state, self.initial_state(state.v), dones)

    def forward(
        self,
        inputs: torch.Tensor,
        state: LICurrentState | None = None,
    ) -> tuple[torch.Tensor, LICurrentState]:
        """Inject charge and advance over the fixed training step."""
        if state is None:
            state = self.initial_state(inputs)
        tau_i = torch.as_tensor(self.tau_i, device=inputs.device, dtype=inputs.dtype)
        ratio = torch.as_tensor(self.tau_ratio, device=inputs.device, dtype=inputs.dtype)
        rate_i = self._cfg.dt / tau_i
        rate_v = rate_i / ratio
        decay_v = torch.exp(-rate_v)
        ratio_gap = ratio - 1
        coupling = -decay_v * torch.expm1(-rate_v * ratio_gap) / ratio_gap
        i_jump = state.i + inputs
        i = torch.exp(-rate_i) * i_jump
        v = decay_v * state.v + coupling * i_jump
        next_state = LICurrentState(v, i)
        return v, next_state
