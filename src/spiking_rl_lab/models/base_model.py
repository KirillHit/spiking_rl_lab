"""Shared model configuration and concrete policy/value models."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal

import torch
from skrl.models.torch import CategoricalMixin, DeterministicMixin, GaussianMixin, Model
from torch import nn

if TYPE_CHECKING:
    import gymnasium
    import gymnasium as gym


@dataclasses.dataclass(kw_only=True, slots=True)
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

    # Network architecture parameters
    net_arch: dict[str, Any] = dataclasses.field(default_factory=dict)
    log_std_init: float = 0.0


def _get_observations(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Get flattened observations from model inputs.

    Args:
        inputs: Model inputs.

    Returns:
        Observation tensor.

    Raises:
        KeyError: If observations are missing.

    """
    observations = inputs.get("observations")
    if observations is None:
        msg = "Model inputs must contain 'observations'"
        raise KeyError(msg)
    return observations.view(observations.shape[0], -1)


class BaseModel(Model):
    """Common base class for spiking RL models."""

    def __init__(
        self,
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
        self._cfg = cfg
        self._net = None  # use generator

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


class CategoricalPolicyModel(CategoricalMixin, BaseModel):
    """Categorical policy model for discrete action spaces."""

    def __init__(
        self,
        cfg: BaseModelCfg,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize categorical policy model.

        Args:
            cfg: Model configuration.
            observation_space: Observation space.
            state_space: State space.
            action_space: Action space.
            device: Device for tensors and modules.

        """
        BaseModel.__init__(
            self,
            cfg=cfg,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        CategoricalMixin.__init__(
            self,
            unnormalized_log_prob=self._cfg.unnormalized_log_prob,
        )

    def compute(
        self,
        inputs: dict[str, Any],
        role: str = "",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute categorical policy output.

        Args:
            inputs: Model inputs.
            role: Model role.

        Returns:
            Policy logits and extra outputs.

        """
        return self._net(_get_observations(inputs)), {}


class GaussianPolicyModel(GaussianMixin, BaseModel):
    """Gaussian policy model for continuous stochastic action spaces."""

    def __init__(
        self,
        cfg: BaseModelCfg,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize Gaussian policy model.

        Args:
            cfg: Model configuration.
            observation_space: Observation space.
            state_space: State space.
            action_space: Action space.
            device: Device for tensors and modules.

        """
        BaseModel.__init__(
            self,
            cfg=cfg,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        self._log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), self._cfg.log_std_init, device=self.device),
        )
        GaussianMixin.__init__(
            self,
            clip_actions=self._cfg.clip_actions,
            clip_mean_actions=self._cfg.clip_mean_actions,
            clip_log_std=self._cfg.clip_log_std,
            min_log_std=self._cfg.min_log_std,
            max_log_std=self._cfg.max_log_std,
            reduction=self._cfg.reduction,
        )

    def compute(
        self,
        inputs: dict[str, Any],
        role: str = "",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute Gaussian policy output.

        Args:
            inputs: Model inputs.
            role: Model role.

        Returns:
            Mean actions and extra outputs with ``log_std``.

        """
        mean_actions = self._net(_get_observations(inputs))
        log_std = self._log_std_parameter.expand_as(mean_actions)
        return mean_actions, {"log_std": log_std}


class DeterministicPolicyModel(DeterministicMixin, BaseModel):
    """Deterministic policy model for continuous action spaces."""

    def __init__(
        self,
        cfg: BaseModelCfg,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize deterministic policy model.

        Args:
            cfg: Model configuration.
            observation_space: Observation space.
            state_space: State space.
            action_space: Action space.
            device: Device for tensors and modules.

        """
        BaseModel.__init__(
            self,
            cfg=cfg,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(
            self,
            clip_actions=self._cfg.clip_actions,
        )

    def compute(
        self,
        inputs: dict[str, Any],
        role: str = "",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute deterministic policy output.

        Args:
            inputs: Model inputs.
            role: Model role.

        Returns:
            Actions and extra outputs.

        """
        return self._net(_get_observations(inputs)), {}


class ValueModel(BaseModel):
    """Dense value model with spiking hidden activations."""

    def __init__(
        self,
        cfg: BaseModelCfg,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize model.

        Args:
            observation_space: Observation space.
            state_space: State space.
            action_space: Action space.
            device: Device for tensors and modules.
            cfg: Model configuration.

        """
        super().__init__(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )

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
        return self._net(_get_observations(inputs)), {}
