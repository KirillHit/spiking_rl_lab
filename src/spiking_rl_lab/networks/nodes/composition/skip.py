"""Skip connections around nested node networks."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

import torch

from spiking_rl_lab.networks.node_network import NodeNetwork, NodeNetworkConfig
from spiking_rl_lab.networks.nodes.base_node import BaseNode
from spiking_rl_lab.networks.nodes.builder import NodeConfig, register_node
from spiking_rl_lab.networks.shape import (
    DenseTensorShape,
    ImageTensorShape,
    SequenceTensorShape,
    TensorShape,
    require_shape,
)

if TYPE_CHECKING:
    from spiking_rl_lab.networks.state import ListState


@register_node("skip")
class SkipNode(BaseNode):
    """Concatenate or add the input and a nested network's output."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BaseNode.Config):
        """Branch nodes and merge operation; concat joins features or channels."""

        nodes: list[NodeConfig]
        merge: Literal["concat", "add"] = "concat"

        def validate(self) -> None:
            """Validate the merge operation."""
            if self.merge not in ("concat", "add"):
                msg = f"Unsupported skip merge: {self.merge!r}"
                raise ValueError(msg)

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Build the branch and validate matching dimensions."""
        super().__init__(cfg, input_shape)
        require_shape(
            "Skip input", input_shape, (DenseTensorShape, SequenceTensorShape, ImageTensorShape)
        )
        self._network = NodeNetwork(NodeNetworkConfig(nodes=cfg.nodes), input_shape=input_shape)
        branch_shape = self._network.output_shape
        merge_field = "features" if isinstance(input_shape, DenseTensorShape) else "channels"
        self._concat = cfg.merge == "concat"
        self._validate_shapes(input_shape, branch_shape, merge_field)
        fields = input_shape.fields
        self._output_shape = input_shape
        if self._concat:
            self._output_shape = dataclasses.replace(
                input_shape, **{merge_field: fields[merge_field] + branch_shape.fields[merge_field]}
            )

    def _validate_shapes(
        self, input_shape: TensorShape, branch_shape: TensorShape, merge_field: str
    ) -> None:
        """Require compatible input and branch shapes for the selected merge."""
        if type(branch_shape) is not type(input_shape):
            msg = "Skip input and branch output must have the same shape type"
            raise ValueError(msg)
        for name, size in input_shape.fields.items():
            if self._concat and name == merge_field:
                continue
            if branch_shape.fields[name] != size:
                msg = f"Skip input and branch output must have matching {name}"
                raise ValueError(msg)

    @property
    def output_shape(self) -> TensorShape:
        """Return the merged output shape."""
        return self._output_shape

    def initial_state(self, inputs: torch.Tensor) -> ListState:
        """Create the branch's initial state."""
        return self._network.initial_state(inputs)

    def reset_state(self, state: ListState | None, dones: torch.Tensor) -> ListState | None:
        """Reset completed rows in the branch state."""
        return self._network.reset_state(state, dones)

    def forward(
        self, inputs: torch.Tensor, state: ListState | None = None
    ) -> tuple[torch.Tensor, ListState]:
        """Run the branch and merge its output with the current input."""
        output, next_state = self._network(inputs, state)
        if self._concat:
            return torch.cat((inputs, output), dim=1), next_state
        return inputs + output, next_state
