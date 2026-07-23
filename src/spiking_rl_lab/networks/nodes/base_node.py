"""Shared abstractions for network nodes."""

from __future__ import annotations

import copy
import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

from spiking_rl_lab.core.factory import ConfiguredBase

if TYPE_CHECKING:
    from spiking_rl_lab.networks.shape import TensorShape
    from spiking_rl_lab.networks.state import ListState


class BaseNode(nn.Module, ConfiguredBase, ABC):
    """Base class for network nodes.

    With identical parameters, inputs, and state, ``forward`` must return identical
    outputs and next state. It must not mutate its state or module buffers.
    """

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
        """Base node configuration."""

    def __init__(self, cfg: Config, input_shape: TensorShape) -> None:
        """Store node configuration."""
        super().__init__()
        ConfiguredBase.__init__(self, cfg)
        self._input_shape = input_shape

    @property
    def cfg(self) -> Config:
        """Return a detached configuration copy."""
        return copy.deepcopy(self._cfg)

    @property
    def input_shape(self) -> TensorShape:
        """Return input shape."""
        return self._input_shape

    @property
    @abstractmethod
    def output_shape(self) -> TensorShape:
        """Return output shape."""

    def initial_state(self, inputs: torch.Tensor) -> ListState | None:
        """Create the initial state for a batch of ``inputs``.

        Stateless nodes keep the default ``None`` state. Nodes that retain
        state must override this method and return their own initial value.
        """
        return None

    def reset_state(self, state: ListState | None, dones: torch.Tensor) -> ListState | None:
        """Reset the state rows selected by ``dones``.

        Stateless nodes keep the default implementation. Stateful nodes must
        override it so their reset value matches :meth:`initial_state`.
        """
        if state is not None:
            msg = f"{type(self).__name__} must implement reset_state for its state"
            raise NotImplementedError(msg)
        return None

    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState]:
        """Compute a deterministic next output and explicit state."""
