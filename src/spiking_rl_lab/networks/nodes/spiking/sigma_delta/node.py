"""Configurable SigmaDelta node with explicit physical and learning state."""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn

from spiking_rl_lab.core.validation import require_minimum, require_positive
from spiking_rl_lab.networks.events import EventPacket, Events
from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node
from spiking_rl_lab.networks.nodes.spiking.sigma_delta.autograd import SigmaDeltaFunction
from spiking_rl_lab.networks.nodes.spiking.sigma_delta.execution import ExecutionConfig, execute
from spiking_rl_lab.networks.nodes.spiking.sigma_delta.state import (
    SigmaDeltaState,
    initial_sigma_delta,
    reset_sigma_delta,
)
from spiking_rl_lab.networks.nodes.spiking.sigma_delta.statistics import ActivityStatistics
from spiking_rl_lab.networks.shape import DenseTensorShape, TensorShape, require_shape


@register_node("sigma_delta")
class SigmaDeltaNode(BaseNode):
    """Analytic neuron with ReLU in hidden layers and signed scalar boundaries."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """Neuron dynamics and CPU execution options."""

        duration: float = 1.0
        tau: float = 1.0
        tau_log_std: float = 0.0
        h: float = 0.03
        learnable_tau: bool = True
        learnable_h: bool = False
        activity_scale: float | None = None
        """Price per event in task-loss units per observation."""
        scalar_input: bool = False
        """Encode scalar values as signed events without ReLU."""
        scalar_output: bool = False
        """Decode connected events to signed potentials without ReLU."""
        hysteresis: float = 1e-4
        chunk_size: int = 64
        threads: int | None = None  # None means using all available threads

        def validate(self) -> None:
            """Require positive dynamics and a valid worker configuration."""
            require_positive("duration", self.duration)
            require_positive("tau", self.tau)
            require_minimum("tau_log_std", self.tau_log_std, minimum=0)
            require_positive("h", self.h)
            if self.activity_scale is not None:
                require_minimum("activity_scale", self.activity_scale, minimum=0)
            require_minimum("hysteresis", self.hysteresis, minimum=0)
            if self.threads is not None:
                require_minimum("threads", self.threads, minimum=1)
            require_minimum("chunk_size", self.chunk_size, minimum=1)

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Register per-neuron log parameters, leaving all state outside the module."""
        super().__init__(cfg, input_shape)
        shape = require_shape(
            "Node 'sigma_delta' input",
            input_shape,
            DenseTensorShape,
            event_stream=not cfg.scalar_input,
        )
        self._neuron_count = shape.features
        self._log_tau = nn.Parameter(
            torch.full((self.neuron_count,), math.log(cfg.tau)),
            requires_grad=cfg.learnable_tau,
        )
        self._log_h = nn.Parameter(
            torch.full((self.neuron_count,), math.log(cfg.h)),
            requires_grad=cfg.learnable_h and not cfg.scalar_output,
        )
        self._output_shape = TensorShape.dense(
            self.neuron_count, event_stream=not cfg.scalar_output
        )
        self._execution_cfg = ExecutionConfig(
            cfg.scalar_output,
            cfg.hysteresis,
            cfg.threads,
            cfg.chunk_size,
            relu=not (cfg.scalar_input or cfg.scalar_output),
        )

    @property
    def neuron_count(self) -> int:
        """Return the number of neurons in the layer."""
        return self._neuron_count

    @property
    def output_shape(self) -> TensorShape:
        """Return event stream geometry or dense endpoint geometry."""
        return self._output_shape

    def initialize_parameters(self) -> None:
        """Initialize neuron timescales and quantization steps."""
        nn.init.normal_(self._log_tau, mean=math.log(self._cfg.tau), std=self._cfg.tau_log_std)
        nn.init.constant_(self._log_h, math.log(self._cfg.h))

    def initial_state(self, inputs: torch.Tensor | EventPacket) -> SigmaDeltaState:
        """Allocate neuron state and traces for scalar or connected event input."""
        weight_count = 0
        batch_size = inputs.shape[0] if self._cfg.scalar_input else inputs.events.batch_size
        if not self._cfg.scalar_input:
            if inputs.connection is None:
                msg = "SigmaDelta requires connection context"
                raise TypeError(msg)
            weight_count = inputs.connection.weights.shape[1]
        return initial_sigma_delta((batch_size, self.neuron_count), self._log_tau, weight_count)

    def create_statistics(
        self, *, progress: float, collect_metrics: bool = False
    ) -> ActivityStatistics | None:
        """Collect outgoing activity and its optional event-cost penalty."""
        if self._cfg.scalar_output or (self._cfg.activity_scale is None and not collect_metrics):
            return None
        return ActivityStatistics(
            self.neuron_count, self._cfg.activity_scale, collect_metrics, self._log_h
        )

    def reset_state(
        self,
        state: SigmaDeltaState | None,
        dones: torch.Tensor,
    ) -> SigmaDeltaState | None:
        """Clear completed rows through the same physical and trace state contract."""
        return reset_sigma_delta(state, dones) if state is not None else None

    def forward(
        self,
        inputs: torch.Tensor | EventPacket,
        state: SigmaDeltaState | None = None,
    ) -> tuple[torch.Tensor | EventPacket, SigmaDeltaState]:
        """Advance persistent accumulators and attach the separate local backward."""
        state = self.initial_state(inputs) if state is None else state
        if self._cfg.scalar_input:
            if inputs.device.type != "cpu" or inputs.dtype not in (torch.float32, torch.float64):
                msg = "Event execution requires CPU float32 or float64 tensors"
                raise ValueError(msg)
            state = state._replace(neuron=state.neuron._replace(drive=inputs.detach()))
            packet = EventPacket(
                Events.empty(inputs, batch_size=inputs.shape[0], neuron_count=self.neuron_count)
            )
            autograd_proxy, weights = inputs, None
        else:
            connection = inputs.connection
            if connection is None:
                msg = "SigmaDelta requires connection context"
                raise TypeError(msg)
            weights = connection.weights
            packet = inputs
            autograd_proxy = inputs.autograd_proxy
        events, next_state = execute(
            packet,
            state,
            self._log_tau,
            self._log_h,
            duration=self._cfg.duration,
            config=self._execution_cfg,
        )
        readout = SigmaDeltaFunction.apply(
            self._cfg.duration if self._cfg.scalar_input else None,
            autograd_proxy,
            weights,
            self._log_tau,
            self._log_h,
            next_state,
            self._execution_cfg,
        )
        output = readout if self._cfg.scalar_output else EventPacket(events, autograd_proxy=readout)
        return output, next_state
