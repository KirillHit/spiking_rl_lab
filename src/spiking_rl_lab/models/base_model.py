"""Shared base classes for models."""

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

import gymnasium as gym
import torch
from skrl.models.torch import Model


@dataclasses.dataclass(kw_only=True)
class BaseModelCfg:
    """Common configuration shared by all models."""

    # Policy distribution parameters for skrl mixins
    # (Categorical / Gaussian / Deterministic).
    unnormalized_log_prob: bool = True
    clip_actions: bool = False
    clip_mean_actions: bool = False
    clip_log_std: bool = True
    min_log_std: float = -20
    max_log_std: float = 2
    reduction: Literal["mean", "sum", "prod", "none"] = "sum"


class BaseModel(Model, ABC):
    """Abstract base class for spiking RL models."""

    cfg_cls: ClassVar[type[BaseModelCfg]] = BaseModelCfg

    def __init__(
        self,
        *,
        cfg: BaseModelCfg,
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
        self.cfg = cfg

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        """Run default action path.

        Non-policy models can rely on this implementation. Policy models override
        it through skrl mixins.

        Args:
            inputs: Model inputs.
            role: Model role.

        Returns:
            Model output from ``compute``.

        """
        return self.compute(inputs, role=role)


class PolicyModel(BaseModel, ABC):
    """Base class for policy models.

    Subclasses are expected to implement the policy hook that matches the
    action space selected by the builder:

    - ``compute_categorical`` for discrete action spaces
    - ``compute_gaussian`` for continuous stochastic action spaces
    - ``compute_deterministic`` for continuous deterministic action spaces

    The builder wraps this class with the corresponding skrl mixin.
    """

    def compute_categorical(
        self,
        inputs: dict[str, Any],
        role: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute categorical policy logits.

        Args:
            inputs: Model inputs.
            role: Model role.

        Returns:
            Policy logits and extra outputs.

        Raises:
            NotImplementedError: If the hook is not implemented.

        """
        msg = f"{type(self).__name__} does not implement compute_categorical()"
        raise NotImplementedError(msg)

    def compute_gaussian(
        self,
        inputs: dict[str, Any],
        role: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute Gaussian policy parameters.

        Args:
            inputs: Model inputs.
            role: Model role.

        Returns:
            Mean actions and extra outputs with ``log_std``.

        Raises:
            NotImplementedError: If the hook is not implemented.

        """
        msg = f"{type(self).__name__} does not implement compute_gaussian()"
        raise NotImplementedError(msg)

    def compute_deterministic(
        self,
        inputs: dict[str, Any],
        role: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute deterministic policy actions.

        Args:
            inputs: Model inputs.
            role: Model role.

        Returns:
            Actions and extra outputs.

        Raises:
            NotImplementedError: If the hook is not implemented.

        """
        msg = f"{type(self).__name__} does not implement compute_deterministic()"
        raise NotImplementedError(msg)


class ValueModel(BaseModel, ABC):
    """Base class for value models.

    Subclasses implement ``compute_value``. The default ``compute`` forwards to
    that method, so the builder can instantiate value models directly.
    """

    def compute(
        self,
        inputs: dict[str, Any],
        *,
        role: str = "",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute value output.

        Args:
            inputs: Model inputs.
            role: Model role.

        Returns:
            Value tensor and extra outputs.

        """
        return self.compute_value(inputs, role)

    @abstractmethod
    def compute_value(
        self,
        inputs: dict[str, Any],
        role: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute value estimates.

        Args:
            inputs: Model inputs.
            role: Model role.

        Returns:
            Value tensor and extra outputs.

        Raises:
            NotImplementedError: If the hook is not implemented.

        """
        msg = f"{type(self).__name__} does not implement compute_value()"
        raise NotImplementedError(msg)
