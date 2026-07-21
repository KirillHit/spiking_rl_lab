"""Policy interfaces independent from neural-network implementations."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from gymnasium.spaces.utils import flatdim

from spiking_rl_lab.core.factory import ConfiguredBase

if TYPE_CHECKING:
    import gymnasium

    from spiking_rl_lab.policies.distributions import ActionDistribution


class BasePolicy(torch.nn.Module, ConfiguredBase, ABC):
    """Build action distributions from a policy-network output tensor."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
        """Base policy configuration."""

    def __init__(self, cfg: Config, *, action_space: gymnasium.Space) -> None:
        """Initialize policy module and action-space metadata."""
        torch.nn.Module.__init__(self)
        ConfiguredBase.__init__(self, cfg)
        self.action_space = action_space
        self.num_actions = flatdim(action_space)

    @abstractmethod
    def distribution(self, features: torch.Tensor) -> ActionDistribution:
        """Build an action distribution from a policy-network output tensor."""
