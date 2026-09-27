"""Learned independent readouts for dense output populations."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

from spiking_rl_lab.core.validation import require_positive, require_shape_fields
from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node
from spiking_rl_lab.networks.shape import DenseTensorShape, TensorShape

if TYPE_CHECKING:
    from spiking_rl_lab.networks.state import ListState


@register_node("population_decode")
class PopulationDecodeNode(BaseNode):
    """Decode contiguous populations with separate trainable weights per output.

    Inputs are ordered by output, then neuron within its population. They are
    decoded by a grouped Conv1d on [batch, outputs, population_size], with
    kernel_size=population_size and groups=outputs. No output bias is added.
    """

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """Population sizes for the independent linear readouts."""

        num_populations: int
        neurons_per_population: int

        def validate(self) -> None:
            """Require positive population dimensions."""
            require_positive("num_populations", self.num_populations)
            require_positive("neurons_per_population", self.neurons_per_population)

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Validate the dense layout and construct independent convolutional readouts."""
        super().__init__(cfg, input_shape)
        require_shape_fields(
            "Node 'population_decode' input",
            input_shape,
            shape_type=DenseTensorShape,
            fields={"features": cfg.num_populations * cfg.neurons_per_population},
        )
        self._output_shape = TensorShape.dense(cfg.num_populations)
        self._layer = torch.nn.Conv1d(
            in_channels=cfg.num_populations,
            out_channels=cfg.num_populations,
            kernel_size=cfg.neurons_per_population,
            groups=cfg.num_populations,
            bias=False,
        )

    @property
    def output_shape(self) -> TensorShape:
        """Return one dense feature per population."""
        return self._output_shape

    def initialize_parameters(self) -> None:
        """Initialize independent linear readouts with Kaiming-normal weights."""
        torch.nn.init.kaiming_normal_(self._layer.weight, mode="fan_in", nonlinearity="linear")

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Sum weighted neuron values independently for every output."""
        populations = inputs.reshape(
            -1, self._cfg.num_populations, self._cfg.neurons_per_population
        )
        outputs = self._layer(populations).squeeze(-1)
        return outputs.reshape(*inputs.shape[:-1], self._cfg.num_populations), None
