"""Base agent helpers for MLflow tracking."""

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Literal

import gymnasium as gym
import torch
from skrl.models.torch import Model


@dataclasses.dataclass(kw_only=True)
class BaseModelCfg:
    """Base class for the model's configuration."""

    # Policy distribution parameters for skrl mixins (Categorical / Gaussian / Deterministic).
    unnormalized_log_prob: bool = True
    clip_actions: bool = False
    clip_mean_actions: bool = False
    clip_log_std: bool = True
    min_log_std: float = -20
    max_log_std: float = 2
    reduction: Literal["mean", "sum", "prod", "none"] = "sum"


class BaseModel(Model, ABC):
    """Base class for spiking neural network models."""

    def __init__(
        self,
        *,
        cfg: BaseModelCfg,
        observation_space: gym.Space | None = None,
        state_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Construct the base model.

        Args:
            cfg (BaseModelCfg): The model's configuration.
            observation_space (gym.Space | None, optional): The ``num_observations`` property
                will contain the size of the space. Defaults to None.
            state_space (gym.Space | None, optional): The ``num_states`` property will contain
                the size of the space. Defaults to None.
            action_space (gym.Space | None, optional): The ``num_actions`` property will
                contain the size of the space. Defaults to None.
            device (str | torch.device | None, optional): Data allocation and computation device.
                If not specified, the default device will be used. Defaults to None.

        """
        super().__init__(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        self.cfg = cfg

    @classmethod
    def requires_policy_mixin(cls) -> bool:
        """Return True if the builder must wrap this core model with a policy mixin.

        Subclasses may override this method to disable policy wrapping.
        """
        return True

    @abstractmethod
    def compute_categorical(
        self,
        inputs: dict[str, Any],
        role: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return categorical policy logits: (logits, extras)."""

    @abstractmethod
    def compute_gaussian(
        self,
        inputs: dict[str, Any],
        role: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return Gaussian params: (mean, log_std, extras)."""

    @abstractmethod
    def compute_deterministic(
        self,
        inputs: dict[str, Any],
        role: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return deterministic actions: (actions, extras)."""
