"""Population coding network node implementation."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

from spiking_rl_lab.core.validation import require_minimum, require_positive
from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node
from spiking_rl_lab.networks.shape import DenseTensorShape, TensorShape, require_shape

if TYPE_CHECKING:
    from spiking_rl_lab.networks.state import ListState


@register_node("population_code")
class PopulationCodeNode(BaseNode):
    """Encode dense scalar features with Gaussian population tuning curves.

    Inputs must be normalized to approximately zero mean and unit variance,
    typically by a running standardization preprocessor.
    """

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """Population coding configuration."""

        neurons_per_feature: int = 8
        sigma: float = 0.3

        def __post_init__(self) -> None:
            """Validate population coding parameters."""
            require_minimum("neurons_per_feature", self.neurons_per_feature, minimum=2)
            require_positive("sigma", self.sigma)

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize the population encoder."""
        super().__init__(cfg, input_shape)
        dense_shape = require_shape("Node 'population_code' input", input_shape, DenseTensorShape)
        self._output_shape = TensorShape.dense(dense_shape.features * cfg.neurons_per_feature)
        self.register_buffer(
            "_centers",
            torch.linspace(-1.0, 1.0, cfg.neurons_per_feature).view(1, 1, -1),
        )

    @property
    def output_shape(self) -> TensorShape:
        """Return output shape."""
        return self._output_shape

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Encode inputs as flattened Gaussian population activities."""
        encoded_inputs = torch.tanh(inputs).unsqueeze(-1)
        activities = torch.exp(-0.5 * ((encoded_inputs - self._centers) / self._cfg.sigma).square())
        return activities.flatten(start_dim=1), None
