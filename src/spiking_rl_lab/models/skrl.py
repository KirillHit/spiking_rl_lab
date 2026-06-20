"""Default skrl model implementations."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal

import torch
from skrl.models.torch import CategoricalMixin, DeterministicMixin, GaussianMixin
from torch import nn

from spiking_rl_lab.models.base_model import (
    BaseModel,
    DeterministicPolicyModel,
    StochasticPolicyModel,
    ValueModel,
)
from spiking_rl_lab.models.builder import register_model
from spiking_rl_lab.networks.shape import DenseTensorShape

if TYPE_CHECKING:
    import gymnasium

    from spiking_rl_lab.networks.base_network import BaseNetwork
    from spiking_rl_lab.networks.builder import NetworkBuildContext
    from spiking_rl_lab.networks.shape import TensorShape


def _get_observations(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Get flattened observations from model inputs."""
    observations = inputs.get("observations")
    if observations is None:
        msg = "Model inputs must contain 'observations'"
        raise KeyError(msg)
    return observations.view(observations.shape[0], -1)


def _get_network_output(outputs: tuple[torch.Tensor, object | None] | torch.Tensor) -> torch.Tensor:
    """Return tensor output from a network call."""
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def _require_dense_network_output(
    model_name: str,
    network_name: str,
    network: BaseNetwork,
    features: int,
) -> None:
    """Raise if a skrl model network output is not dense with the expected width."""
    output_shape = network.output_shape
    if not isinstance(output_shape, DenseTensorShape):
        msg = (
            f"{model_name} requires dense output for network '{network_name}', "
            f"got {output_shape.kind}"
        )
        raise TypeError(msg)
    if output_shape.features != features:
        msg = (
            f"{model_name} requires network '{network_name}' output width "
            f"{features}, got {output_shape.features}"
        )
        raise ValueError(msg)


@register_model("categorical_policy")
class CategoricalPolicyModel(CategoricalMixin, StochasticPolicyModel):
    """Categorical policy model for discrete action spaces."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
        """Categorical policy model configuration."""

        network: str
        unnormalized_log_prob: bool = True

    def __init__(
        self,
        cfg: Config,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        network_builder: NetworkBuildContext | None = None,
    ) -> None:
        """Initialize categorical policy model."""
        BaseModel.__init__(
            self,
            cfg=cfg,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            network_builder=network_builder,
        )
        network = self.register_network(self._cfg.network)
        _require_dense_network_output(
            self.__class__.__name__,
            self._cfg.network,
            network,
            self.num_actions,
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
        """Compute categorical policy output."""
        outputs = self.get_network(self._cfg.network)(_get_observations(inputs))
        return _get_network_output(outputs), {}

    @property
    def output_shape(self) -> TensorShape:
        """Return model output shape."""
        return self.get_network(self._cfg.network).output_shape


@register_model("gaussian_policy")
class GaussianPolicyModel(GaussianMixin, StochasticPolicyModel):
    """Gaussian policy model for continuous stochastic action spaces."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
        """Gaussian policy model configuration."""

        network: str
        clip_actions: bool = False
        clip_mean_actions: bool = False
        clip_log_std: bool = True
        min_log_std: float = -20
        max_log_std: float = 2
        reduction: Literal["mean", "sum", "prod", "none"] = "sum"
        log_std_init: float = 0.0

    def __init__(
        self,
        cfg: Config,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        network_builder: NetworkBuildContext | None = None,
    ) -> None:
        """Initialize Gaussian policy model."""
        BaseModel.__init__(
            self,
            cfg=cfg,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            network_builder=network_builder,
        )
        network = self.register_network(self._cfg.network)
        _require_dense_network_output(
            self.__class__.__name__,
            self._cfg.network,
            network,
            self.num_actions,
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
        """Compute Gaussian policy output."""
        outputs = self.get_network(self._cfg.network)(_get_observations(inputs))
        mean_actions = _get_network_output(outputs)
        log_std = self._log_std_parameter.expand_as(mean_actions)
        return mean_actions, {"log_std": log_std}

    @property
    def output_shape(self) -> TensorShape:
        """Return model output shape."""
        return self.get_network(self._cfg.network).output_shape


@register_model("deterministic_policy")
class SkrlDeterministicPolicyModel(DeterministicMixin, DeterministicPolicyModel):
    """Deterministic policy model for continuous action spaces."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
        """Deterministic policy model configuration."""

        network: str
        clip_actions: bool = False

    def __init__(
        self,
        cfg: Config,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        network_builder: NetworkBuildContext | None = None,
    ) -> None:
        """Initialize deterministic policy model."""
        BaseModel.__init__(
            self,
            cfg=cfg,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            network_builder=network_builder,
        )
        network = self.register_network(self._cfg.network)
        _require_dense_network_output(
            self.__class__.__name__,
            self._cfg.network,
            network,
            self.num_actions,
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
        """Compute deterministic policy output."""
        outputs = self.get_network(self._cfg.network)(_get_observations(inputs))
        return _get_network_output(outputs), {}

    @property
    def output_shape(self) -> TensorShape:
        """Return model output shape."""
        return self.get_network(self._cfg.network).output_shape


@register_model("value")
class SkrlValueModel(ValueModel):
    """Dense value model with spiking hidden activations."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
        """Value model configuration."""

        network: str

    def __init__(
        self,
        cfg: Config,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        network_builder: NetworkBuildContext | None = None,
    ) -> None:
        """Initialize value model."""
        super().__init__(
            cfg=cfg,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            network_builder=network_builder,
        )
        network = self.register_network(self._cfg.network)
        _require_dense_network_output(self.__class__.__name__, self._cfg.network, network, 1)

    def compute(
        self,
        inputs: dict[str, Any],
        *,
        role: str = "",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute value output."""
        outputs = self.get_network(self._cfg.network)(_get_observations(inputs))
        return _get_network_output(outputs), {}

    @property
    def output_shape(self) -> TensorShape:
        """Return model output shape."""
        return self.get_network(self._cfg.network).output_shape
