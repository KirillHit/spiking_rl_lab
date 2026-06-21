"""Default skrl model implementations."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal

import torch
from skrl.models.torch import CategoricalMixin, DeterministicMixin, GaussianMixin
from torch import nn

from spiking_rl_lab.core.validation import require_shape_fields
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


def _compute_network(
    network: BaseNetwork,
    inputs: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run a network with optional hidden state and return skrl outputs."""
    outputs = network(_get_observations(inputs), inputs.get("hidden_states"))
    if not isinstance(outputs, tuple):
        return outputs, {}

    network_output, hidden_states = outputs
    return network_output, {"hidden_states": hidden_states}


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
        CategoricalMixin.__init__(
            self,
            unnormalized_log_prob=self._cfg.unnormalized_log_prob,
        )
        network = self.register_network(self._cfg.network)
        require_shape_fields(
            f"{self.__class__.__name__} network '{self._cfg.network}' output",
            network.output_shape,
            shape_type=DenseTensorShape,
            kind="dense output",
            fields={"features": self.num_actions},
        )

    def compute(
        self,
        inputs: dict[str, Any],
        role: str = "",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute categorical policy output."""
        return _compute_network(self.get_network(self._cfg.network), inputs)

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
        GaussianMixin.__init__(
            self,
            clip_actions=self._cfg.clip_actions,
            clip_mean_actions=self._cfg.clip_mean_actions,
            clip_log_std=self._cfg.clip_log_std,
            min_log_std=self._cfg.min_log_std,
            max_log_std=self._cfg.max_log_std,
            reduction=self._cfg.reduction,
        )
        network = self.register_network(self._cfg.network)
        require_shape_fields(
            f"{self.__class__.__name__} network '{self._cfg.network}' output",
            network.output_shape,
            shape_type=DenseTensorShape,
            kind="dense output",
            fields={"features": self.num_actions},
        )
        self._log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), self._cfg.log_std_init, device=self.device),
        )

    def compute(
        self,
        inputs: dict[str, Any],
        role: str = "",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute Gaussian policy output."""
        mean_actions, outputs = _compute_network(self.get_network(self._cfg.network), inputs)
        log_std = self._log_std_parameter.expand_as(mean_actions)
        outputs["log_std"] = log_std
        return mean_actions, outputs

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
        DeterministicMixin.__init__(
            self,
            clip_actions=self._cfg.clip_actions,
        )
        network = self.register_network(self._cfg.network)
        require_shape_fields(
            f"{self.__class__.__name__} network '{self._cfg.network}' output",
            network.output_shape,
            shape_type=DenseTensorShape,
            kind="dense output",
            fields={"features": self.num_actions},
        )

    def compute(
        self,
        inputs: dict[str, Any],
        role: str = "",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute deterministic policy output."""
        return _compute_network(self.get_network(self._cfg.network), inputs)

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
        require_shape_fields(
            f"{self.__class__.__name__} network '{self._cfg.network}' output",
            network.output_shape,
            shape_type=DenseTensorShape,
            kind="dense output",
            fields={"features": 1},
        )

    def compute(
        self,
        inputs: dict[str, Any],
        *,
        role: str = "",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute value output."""
        return _compute_network(self.get_network(self._cfg.network), inputs)

    def act(
        self,
        inputs: dict[str, Any],
        *,
        role: str = "",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return value estimates for the given inputs."""
        return self.compute(inputs, role=role)

    @property
    def output_shape(self) -> TensorShape:
        """Return model output shape."""
        return self.get_network(self._cfg.network).output_shape
