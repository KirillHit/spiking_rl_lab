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
    from spiking_rl_lab.networks.types import ListState, TensorShape


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

    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState]:
        """Compute a deterministic next output and explicit state."""
