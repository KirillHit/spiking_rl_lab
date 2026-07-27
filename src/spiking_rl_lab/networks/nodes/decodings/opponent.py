"""Opponent-population decoding network node implementation."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from spiking_rl_lab.core.validation import require_positive, require_shape_fields
from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import register_node
from spiking_rl_lab.networks.shape import DenseTensorShape, TensorShape

if TYPE_CHECKING:
    import torch

    from spiking_rl_lab.networks.state import ListState


@register_node("opponent_decode")
class OpponentDecodeNode(BaseNode):
    """Decode positive and negative spike populations into normalized balances.

    Input features are ordered by population, then positive and negative group,
    then neuron within the group. Inputs are expected to be output spikes.

    This is inspired by opponent activity in striatal D1/D2 populations, which
    functionally promote and suppress an action respectively. It preserves each
    population's relative balance while largely discarding its total activity,
    so a later readout can combine independent signed population votes.

    See González-Redondo et al., 2023:
    https://doi.org/10.1016/j.neucom.2023.126377
    """

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """Opponent-population decoder configuration."""

        num_populations: int
        neurons_per_group: int
        epsilon: float = 1e-6

        def __post_init__(self) -> None:
            """Validate decoder parameters."""
            require_positive("num_populations", self.num_populations)
            require_positive("neurons_per_group", self.neurons_per_group)
            require_positive("epsilon", self.epsilon)

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Initialize the decoder and validate its input population layout."""
        super().__init__(cfg, input_shape)
        expected_features = 2 * cfg.num_populations * cfg.neurons_per_group
        require_shape_fields(
            "Node 'opponent_decode' input",
            input_shape,
            shape_type=DenseTensorShape,
            fields={"features": expected_features},
        )
        self._output_shape = TensorShape.dense(cfg.num_populations)

    @property
    def output_shape(self) -> TensorShape:
        """Return one normalized balance for every opponent population."""
        return self._output_shape

    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState | None]:
        """Return per-population balances in the range from -1 to 1."""
        populations = inputs.reshape(
            -1,
            self._cfg.num_populations,
            2,
            self._cfg.neurons_per_group,
        )
        activities = populations.sum(dim=-1)
        positive, negative = activities.unbind(dim=2)
        balances = (positive - negative) / (positive + negative + self._cfg.epsilon)
        return balances, None
