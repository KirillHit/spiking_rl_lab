"""Continuous activations with optional binary output for checkpoint transfer."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar, Literal

import torch

from spiking_rl_lab.core.validation import require_minimum, require_range
from spiking_rl_lab.networks.nodes.builder import register_node
from spiking_rl_lab.networks.nodes.standard.activations import TorchActivationNode
from spiking_rl_lab.networks.nodes.statistics import NodeStatistics

if TYPE_CHECKING:
    from spiking_rl_lab.networks.nodes.statistics import NodeStatisticsAccumulator
    from spiking_rl_lab.networks.state import ListState


@dataclasses.dataclass
class _BinaryStatistics:
    """Accumulate binary and activity penalties with thresholded activity metrics."""

    binary_scale: float | None
    activity_scale: float | None
    collect_metrics: bool
    threshold: float
    binary_total: torch.Tensor | None = None
    activity_penalty_total: torch.Tensor | None = None
    event_total: torch.Tensor | None = None
    count: int = 0

    def update(self, outputs: torch.Tensor) -> None:
        """Accumulate differentiable penalties and detached event counts."""
        if self.binary_scale is not None:
            penalty = (outputs.square() * (outputs - 1).square()).sum()
            self.binary_total = (
                penalty if self.binary_total is None else self.binary_total + penalty
            )
        if self.activity_scale is not None:
            penalty = outputs.abs().sum()
            self.activity_penalty_total = (
                penalty
                if self.activity_penalty_total is None
                else self.activity_penalty_total + penalty
            )
        self.count += outputs.numel()
        if self.collect_metrics:
            activity = (outputs.detach() > self.threshold).to(outputs.dtype).sum()
            self.event_total = activity if self.event_total is None else self.event_total + activity

    def result(self) -> NodeStatistics | None:
        """Sum penalties and report event frequency and effective coefficients."""
        if not self.count:
            return None
        loss = None
        metrics = {}
        for name, scale, total in (
            ("binary", self.binary_scale, self.binary_total),
            ("activity", self.activity_scale, self.activity_penalty_total),
        ):
            if scale is not None:
                term = scale * total / self.count
                loss = term if loss is None else loss + term
                if self.collect_metrics:
                    metrics[f"{name}_loss"] = term.detach()
        if self.collect_metrics:
            metrics.update(
                {
                    "activity_mean": self.event_total / self.count,
                    "binary_scale": self.event_total.new_tensor(self.binary_scale or 0.0),
                    "activity_scale": self.event_total.new_tensor(self.activity_scale or 0.0),
                }
            )
        return NodeStatistics(loss=loss, metrics=metrics)


@register_node("binary_activation")
class BinaryActivationNode(TorchActivationNode):
    """Run a continuous activation or threshold its output into binary events."""

    activations: ClassVar[dict[str, type[torch.nn.Module]]] = {
        "relu": torch.nn.ReLU,
        "silu": torch.nn.SiLU,
    }

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(TorchActivationNode.Config):
        """Choose the activation and an output threshold for optional hard mode."""

        activation: Literal["relu", "silu"] = "relu"
        threshold: float = 0.5
        hard: bool = False
        binary_scale: float = 0.0
        activity_scale: float = 0.0
        regularization_ramp_fraction: float = 0.0

        def validate(self) -> None:
            """Require a nonnegative penalty coefficient and a bounded ramp duration."""
            require_minimum("binary_scale", self.binary_scale, minimum=0.0)
            require_minimum("activity_scale", self.activity_scale, minimum=0.0)
            require_range(
                "regularization_ramp_fraction",
                self.regularization_ramp_fraction,
                minimum=0.0,
                maximum=1.0,
            )

    def create_statistics(
        self, *, progress: float, collect_metrics: bool = False
    ) -> NodeStatisticsAccumulator | None:
        """Create an accumulator for binary and activity penalties and optional metrics."""
        binary_enabled = bool(self._cfg.binary_scale) and not self._cfg.hard
        activity_enabled = bool(self._cfg.activity_scale) and not self._cfg.hard
        if not binary_enabled and not activity_enabled and not collect_metrics:
            return None
        ramp = self._cfg.regularization_ramp_fraction
        fraction = min(1.0, progress / ramp) if ramp else 1.0
        binary_scale = self._cfg.binary_scale * fraction if binary_enabled else None
        activity_scale = self._cfg.activity_scale * fraction if activity_enabled else None
        return _BinaryStatistics(binary_scale, activity_scale, collect_metrics, self._cfg.threshold)

    def forward(
        self, inputs: torch.Tensor, state: ListState | None = None
    ) -> tuple[torch.Tensor, None]:
        """Apply the activation and optionally binarize it with a strict threshold."""
        outputs = self._activation(inputs)
        if self._cfg.hard:
            outputs = (outputs > self._cfg.threshold).to(outputs.dtype)
        return outputs, None
