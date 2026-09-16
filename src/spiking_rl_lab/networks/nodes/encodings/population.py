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
    """Encode dense scalar features with PopSAN-style Gaussian tuning curves.

    Inputs must be normalized to approximately zero mean and unit variance,
    typically by a running standardization preprocessor.

    See Tang et al., 2021: https://proceedings.mlr.press/v155/tang21a.html
    """

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """Population coding configuration."""

        neurons_per_feature: int = 8
        sigma: float = 0.3
        learnable_mu: bool = False
        learnable_sigma: bool = False

        def __post_init__(self) -> None:
            """Validate population coding parameters."""
            require_minimum("neurons_per_feature", self.neurons_per_feature, minimum=2)
            require_positive("sigma", self.sigma)

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize the population encoder."""
        super().__init__(cfg, input_shape)
        dense_shape = require_shape("Node 'population_code' input", input_shape, DenseTensorShape)
        self._output_shape = TensorShape.dense(dense_shape.features * cfg.neurons_per_feature)
        parameter_shape = (dense_shape.features, cfg.neurons_per_feature)
        boundary_margin = 1.0 / cfg.neurons_per_feature
        mu_centers = (
            torch.linspace(
                -1.0 + boundary_margin,
                1.0 - boundary_margin,
                cfg.neurons_per_feature,
            )
            .expand(parameter_shape)
            .clone()
        )
        self._mu_raw = torch.nn.Parameter(torch.atanh(mu_centers), requires_grad=cfg.learnable_mu)

        sigma = torch.full(parameter_shape, cfg.sigma)
        sigma_raw = sigma + torch.log(-torch.expm1(-sigma))
        self._sigma_raw = torch.nn.Parameter(sigma_raw, requires_grad=cfg.learnable_sigma)

    @property
    def mu(self) -> torch.Tensor:
        """Return the population centers."""
        return torch.tanh(self._mu_raw)

    @property
    def sigma(self) -> torch.Tensor:
        """Return positive population widths."""
        return torch.nn.functional.softplus(self._sigma_raw)

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
        activities = torch.exp(-0.5 * ((encoded_inputs - self.mu) / self.sigma).square())
        return activities.flatten(start_dim=1), None
