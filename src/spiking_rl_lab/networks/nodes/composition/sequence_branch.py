"""Apply a sequence network to the leading population-coded features."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

from spiking_rl_lab.core.validation import require_positive
from spiking_rl_lab.networks.node_network import NodeNetwork, NodeNetworkConfig
from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import NodeConfig, register_node
from spiking_rl_lab.networks.shape import (
    DenseTensorShape,
    SequenceTensorShape,
    TensorShape,
    require_shape,
)

if TYPE_CHECKING:
    from spiking_rl_lab.networks.state import ListState


@register_node("sequence_branch")
class SequenceBranchNode(BaseNode):
    """Process an ordered feature prefix and concatenate the untouched suffix.

    Each position's population channels are contiguous in the dense input.
    """

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """Feature counts before coding and channels per feature after coding."""

        sequence_features: int
        tail_features: int
        channels_per_feature: int
        nodes: list[NodeConfig]

        def validate(self) -> None:
            """Require a nonempty sequence and valid channel counts."""
            require_positive("sequence_features", self.sequence_features)
            require_positive("tail_features", self.tail_features)
            require_positive("channels_per_feature", self.channels_per_feature)
            if not self.nodes:
                msg = "sequence_branch requires at least one branch node"
                raise ValueError(msg)

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Build the sequence network and validate the dense input layout."""
        super().__init__(cfg, input_shape)
        dense_shape = require_shape("Node 'sequence_branch' input", input_shape, DenseTensorShape)
        self._encoded_sequence_size = cfg.sequence_features * cfg.channels_per_feature
        encoded_tail_size = cfg.tail_features * cfg.channels_per_feature
        expected_features = self._encoded_sequence_size + encoded_tail_size
        if dense_shape.features != expected_features:
            msg = (
                "sequence_branch expects "
                f"{expected_features} encoded features, got {dense_shape.features}"
            )
            raise ValueError(msg)

        self._network = NodeNetwork(
            NodeNetworkConfig(nodes=cfg.nodes),
            input_shape=TensorShape.sequence(cfg.channels_per_feature, cfg.sequence_features),
        )
        branch_shape = require_shape(
            "Node 'sequence_branch' branch output",
            self._network.output_shape,
            SequenceTensorShape,
        )
        self._output_shape = TensorShape.dense(
            branch_shape.channels * branch_shape.length + encoded_tail_size
        )

    @property
    def output_shape(self) -> TensorShape:
        """Return the flattened branch and suffix size."""
        return self._output_shape

    def _sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        """Move population channels ahead of the ordered positions."""
        sequence = inputs[:, : self._encoded_sequence_size].reshape(
            inputs.shape[0],
            self._cfg.sequence_features,
            self._cfg.channels_per_feature,
        )
        return sequence.transpose(1, 2)

    def initial_state(self, inputs: torch.Tensor) -> ListState:
        """Initialize the state of the sequence branch."""
        return self._network.initial_state(self._sequence(inputs))

    def reset_state(self, state: ListState | None, dones: torch.Tensor) -> ListState | None:
        """Reset the sequence branch for finished environments."""
        return self._network.reset_state(state, dones)

    def forward(
        self, inputs: torch.Tensor, state: ListState | None = None
    ) -> tuple[torch.Tensor, ListState]:
        """Extract local features and join them with the remaining populations."""
        features, next_state = self._network(self._sequence(inputs), state)
        output = torch.cat(
            (features.flatten(start_dim=1), inputs[:, self._encoded_sequence_size :]),
            dim=1,
        )
        return output, next_state
