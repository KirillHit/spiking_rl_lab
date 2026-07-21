"""Policy interfaces independent from neural-network implementations."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from gymnasium.spaces.utils import flatdim

from spiking_rl_lab.core.factory import ConfiguredBase

if TYPE_CHECKING:
    from collections.abc import Mapping

    import gymnasium
    import torch

    from spiking_rl_lab.policies.distributions import ActionDistribution


class BasePolicy(ConfiguredBase, ABC):
    """Build action distributions from network-produced parameters."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
        """Base policy configuration."""

    def __init__(self, cfg: Config, *, action_space: gymnasium.Space) -> None:
        """Store policy metadata without creating trainable modules."""
        super().__init__(cfg)
        self.action_space = action_space
        self.num_actions = flatdim(action_space)

    @abstractmethod
    def distribution(self, parameters: Mapping[str, torch.Tensor]) -> ActionDistribution:
        """Build an action distribution from network-produced parameters."""
