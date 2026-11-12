"""One trainable weight and discrete causal delay per dense connection."""

from __future__ import annotations

import dataclasses
import math
from itertools import pairwise
from typing import Literal, NamedTuple

import torch
from torch import nn
from torch.nn import functional

from spiking_rl_lab.core.validation import require_minimum, require_positive
from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node
from spiking_rl_lab.networks.shape import DenseTensorShape, TensorShape, require_shape


class DelayState(NamedTuple):
    """Newest-first input history and per-row bootstrap flag."""

    history: torch.Tensor  # [batch, max_delay, in_features], past inputs, newest first.
    initialized: torch.Tensor  # bool [batch, 1], false rows repeat the next input as history.


@dataclasses.dataclass(kw_only=True, slots=True)
class DelayStage:
    """Delay operator applied from a given fraction of training onwards."""

    start: float
    mode: Literal["soft", "ste", "fixed"]
    temperature: float = 1.0

    def __post_init__(self) -> None:
        """Validate a training fraction and operator parameters."""
        if not 0 <= self.start <= 1:
            msg = "delay stage start must be in [0, 1]"
            raise ValueError(msg)
        _validate_operator(self.mode, self.temperature)


def _validate_operator(mode: str, temperature: float) -> None:
    require_positive("temperature", temperature)
    if mode not in {"soft", "ste", "fixed"} or not math.isfinite(temperature):
        msg = "delay_linear requires soft, ste or fixed mode and finite temperature"
        raise ValueError(msg)


@register_node("delay_linear")
class DelayLinearNode(BaseNode):
    """Mix causal taps or select one delay per edge, returning explicit state.

    Inputs are [batch, features]. Delays count internal network ticks. Bootstrap
    repeats the first input throughout history. This is a dense training node,
    not an event scheduler. Call set_mode("fixed") to capture a fixed operator.
    """

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """Output size, maximum delay, bias and temporal training operator."""

        out_features: int
        max_delay: int = 8
        bias: bool = True
        mode: Literal["soft", "ste", "fixed"] = "ste"
        temperature: float = 1.0
        schedule: list[DelayStage] = dataclasses.field(default_factory=list)

        def __post_init__(self) -> None:
            """Convert YAML stages to validated configurations."""
            self.schedule = [
                stage if isinstance(stage, DelayStage) else DelayStage(**stage)
                for stage in self.schedule
            ]

        def validate(self) -> None:
            """Require valid dimensions, a supported mode and finite temperature."""
            require_minimum("out_features", self.out_features, minimum=1)
            require_minimum("max_delay", self.max_delay, minimum=0)
            _validate_operator(self.mode, self.temperature)
            starts = [stage.start for stage in self.schedule]
            if starts and (starts[0] != 0 or any(a >= b for a, b in pairwise(starts))):
                msg = "delay schedule must start at 0 and have strictly increasing starts"
                raise ValueError(msg)

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Allocate spatial weights, delay logits and deployment buffers."""
        super().__init__(cfg, input_shape)
        shape = require_shape("Node 'delay_linear' input", input_shape, DenseTensorShape)
        first = cfg.schedule[0] if cfg.schedule else cfg
        self._mode = first.mode
        self.temperature = first.temperature
        self.weight = nn.Parameter(torch.empty(cfg.out_features, shape.features))
        self.bias = nn.Parameter(torch.empty(cfg.out_features)) if cfg.bias else None
        self.delay_logits = nn.Parameter(
            torch.empty(cfg.out_features, cfg.max_delay + 1, shape.features)
        )
        self.register_buffer("fixed_delays", torch.zeros_like(self.weight, dtype=torch.long))
        self._output_shape = TensorShape.dense(cfg.out_features)

    @property
    def mode(self) -> str:
        """Return the active operator, independent of train/eval status."""
        return self._mode

    def set_mode(self, mode: str, *, temperature: float | None = None) -> None:
        """Change the operator in place, capturing argmax when entering fixed."""
        temperature = self.temperature if temperature is None else temperature
        _validate_operator(mode, temperature)
        if mode == "fixed" and self.mode != "fixed":
            with torch.no_grad():
                self.fixed_delays.copy_(self.delay_logits.argmax(dim=1))
        self._mode, self.temperature = mode, temperature

    def update_training_progress(self, progress: float) -> None:
        """Apply a configured stage for the supplied training progress."""
        if not 0 <= progress <= 1:
            msg = "training progress must be in [0, 1]"
            raise ValueError(msg)
        stages = self._cfg.schedule
        if not stages:
            return
        stage = next(stage for stage in reversed(stages) if progress >= stage.start)
        self.set_mode(stage.mode, temperature=stage.temperature)

    def get_extra_state(self) -> dict[str, str | float]:
        """Persist runtime mode and temperature alongside tensors."""
        return {"mode": self.mode, "temperature": self.temperature}

    def set_extra_state(self, state: dict[str, str | float]) -> None:
        """Restore runtime settings without recapturing fixed delays."""
        _validate_operator(state["mode"], state["temperature"])
        self._mode, self.temperature = state["mode"], state["temperature"]

    @property
    def output_shape(self) -> TensorShape:
        """Return the flat output shape."""
        return self._output_shape

    @torch.no_grad()
    def initialize_parameters(self) -> None:
        """Match the reference's uniform weights and short-delay prior."""
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        lag = torch.arange(
            self._cfg.max_delay + 1, device=self.weight.device, dtype=self.weight.dtype
        )
        self.delay_logits.copy_(-math.log(2) * lag[None, :, None])
        self.delay_logits.add_(0.01 * torch.randn_like(self.delay_logits))
        self.fixed_delays.copy_(self.delay_logits.argmax(dim=1))

    def initial_state(self, inputs: torch.Tensor) -> DelayState | None:
        """Return unused history, or no state for an ordinary linear operator."""
        if self._cfg.max_delay == 0:
            return None
        return DelayState(
            inputs.new_zeros(inputs.shape[0], self._cfg.max_delay, inputs.shape[-1]),
            torch.zeros(inputs.shape[0], 1, device=inputs.device, dtype=torch.bool),
        )

    def reset_state(self, state: DelayState | None, dones: torch.Tensor) -> DelayState | None:
        """Clear completed rows without changing the supplied state."""
        if state is None:
            return None
        return DelayState(
            state.history.masked_fill(dones.reshape(-1, 1, 1), 0),
            state.initialized.masked_fill(dones.reshape(-1, 1), value=False),
        )

    def forward(
        self, inputs: torch.Tensor, state: DelayState | None = None
    ) -> tuple[torch.Tensor, DelayState | None]:
        """Apply the chosen temporal operator and advance input history."""
        if self._cfg.max_delay == 0:
            return functional.linear(inputs, self.weight, self.bias), None
        if state is None:
            state = self.initial_state(inputs)
        past = torch.where(state.initialized.unsqueeze(1), state.history, inputs.unsqueeze(1))
        taps = torch.cat((inputs.unsqueeze(1), past), dim=1)
        if self.mode == "fixed":
            channel = torch.arange(inputs.shape[-1], device=inputs.device)
            output = (taps[:, self.fixed_delays, channel] * self.weight).sum(dim=-1)
            if self.bias is not None:
                output = output + self.bias
        else:
            probabilities = (self.delay_logits / self.temperature).softmax(dim=1)
            gate = probabilities
            if self.mode == "ste":
                hard = functional.one_hot(
                    self.delay_logits.argmax(dim=1), self._cfg.max_delay + 1
                ).permute(0, 2, 1)
                gate = hard.to(probabilities) - probabilities.detach() + probabilities
            kernel = gate * self.weight.unsqueeze(1)
            output = functional.linear(taps.flatten(1), kernel.flatten(1), self.bias)
        return output, DelayState(taps[:, :-1], torch.ones_like(state.initialized))
