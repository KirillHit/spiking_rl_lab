"""Shared base class for configured skrl models."""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from typing import TYPE_CHECKING

from skrl.models.torch import Model

from spiking_rl_lab.core.factory import ConfiguredBase

if TYPE_CHECKING:
    import gymnasium as gym
    import torch

    from spiking_rl_lab.networks.types import TensorShape


class BaseModel(Model, ConfiguredBase):
    """Common base class for spiking RL models."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
        """Base model configuration."""

    def __init__(
        self,
        cfg: object,
        *,
        observation_space: gym.Space | None = None,
        state_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize model base state.

        Args:
            cfg: Model configuration.
            observation_space: Observation space.
            state_space: State space.
            action_space: Action space.
            device: Device for tensors and modules.

        """
        super().__init__(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        ConfiguredBase.__init__(self, cfg)

    @property
    @abstractmethod
    def output_shape(self) -> TensorShape:
        """Return model output shape."""


class PolicyModel(BaseModel):
    """Base class for action-producing models."""


class StochasticPolicyModel(PolicyModel):
    """Base class for stochastic policy models."""


class DeterministicPolicyModel(PolicyModel):
    """Base class for deterministic policy models."""


class ValueModel(BaseModel):
    """Base class for value-function models."""
