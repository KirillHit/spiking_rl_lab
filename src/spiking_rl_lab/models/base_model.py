"""Shared model configuration and concrete policy/value models."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from skrl.models.torch import CategoricalMixin, DeterministicMixin, GaussianMixin, Model
from torch import nn

from spiking_rl_lab.core.factory import ConfiguredBase

if TYPE_CHECKING:
    import gymnasium
    import gymnasium as gym

    from spiking_rl_lab.network.shape import TensorShape


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


class BaseModel(Model, ConfiguredBase):
    """Common base class for spiking RL models."""

    Config: ClassVar[type[BaseModelCfg]] = BaseModelCfg

    def __init__(
        self,
        cfg: BaseModelCfg,
        *,
        observation_space: gym.Space | None = None,
        state_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
        device: str | torch.device | None = None,
        network: nn.Module | None = None,
    ) -> None:
        """Initialize model base state.

        Args:
            cfg: Model configuration.
            observation_space: Observation space.
            state_space: State space.
            action_space: Action space.
            device: Device for tensors and modules.
            network: Network module used by ``compute``.

        """
        super().__init__(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        ConfiguredBase.__init__(self, cfg)
        self._net = network

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

    def _forward_network(self, observations: torch.Tensor) -> torch.Tensor:
        """Run the configured network and return only its tensor output."""
        if self._net is None:
            msg = f"{self.__class__.__name__} requires a configured network"
            raise RuntimeError(msg)
        outputs = self._net(observations)
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    @property
    def network_output_shape(self) -> TensorShape:
        """Return the configured network output shape."""
        if self._net is None:
            msg = f"{self.__class__.__name__} requires a configured network"
            raise RuntimeError(msg)
        return self._net.output_shape


class CategoricalPolicyModel(CategoricalMixin, BaseModel):
    """Categorical policy model for discrete action spaces."""

    def __init__(
        self,
        cfg: BaseModelCfg,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        network: nn.Module | None = None,
    ) -> None:
        """Initialize categorical policy model.

        Args:
            cfg: Model configuration.
            observation_space: Observation space.
            state_space: State space.
            action_space: Action space.
            device: Device for tensors and modules.
            network: Network module used for forward passes.

        """
        BaseModel.__init__(
            self,
            cfg=cfg,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            network=network,
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
        return self._forward_network(_get_observations(inputs)), {}


class GaussianPolicyModel(GaussianMixin, BaseModel):
    """Gaussian policy model for continuous stochastic action spaces."""

    def __init__(
        self,
        cfg: BaseModelCfg,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        network: nn.Module | None = None,
    ) -> None:
        """Initialize Gaussian policy model.

        Args:
            cfg: Model configuration.
            observation_space: Observation space.
            state_space: State space.
            action_space: Action space.
            device: Device for tensors and modules.
            network: Network module used for forward passes.

        """
        BaseModel.__init__(
            self,
            cfg=cfg,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            network=network,
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
        mean_actions = self._forward_network(_get_observations(inputs))
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
        network: nn.Module | None = None,
    ) -> None:
        """Initialize deterministic policy model.

        Args:
            cfg: Model configuration.
            observation_space: Observation space.
            state_space: State space.
            action_space: Action space.
            device: Device for tensors and modules.
            network: Network module used for forward passes.

        """
        BaseModel.__init__(
            self,
            cfg=cfg,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            network=network,
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
        return self._forward_network(_get_observations(inputs)), {}


class ValueModel(BaseModel):
    """Dense value model with spiking hidden activations."""

    def __init__(
        self,
        cfg: BaseModelCfg,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        network: nn.Module | None = None,
    ) -> None:
        """Initialize model.

        Args:
            observation_space: Observation space.
            state_space: State space.
            action_space: Action space.
            device: Device for tensors and modules.
            cfg: Model configuration.
            network: Network module used for forward passes.

        """
        super().__init__(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
            network=network,
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
        return self._forward_network(_get_observations(inputs)), {}
