"""Simple MLP policy and value models."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar, Literal

import torch
from torch import nn

from spiking_rl_lab.models.base_model import BaseModelCfg, PolicyModel, ValueModel
from spiking_rl_lab.models.builder import register_model

if TYPE_CHECKING:
    import gymnasium


@dataclasses.dataclass(kw_only=True)
class MLPCfg(BaseModelCfg):
    """Configuration for MLP models."""

    net_arch: list[int] = dataclasses.field(default_factory=lambda: [64, 64])
    activation: Literal["relu", "tanh", "elu", "silu"] = "tanh"
    log_std_init: float = 0.0


def _get_activation(name: str) -> type[nn.Module]:
    """Get activation module class.

    Args:
        name: Activation name.

    Returns:
        Activation module class.

    Raises:
        ValueError: If the activation is unsupported.

    """
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "silu": nn.SiLU,
    }
    activation = activations.get(name)
    if activation is None:
        msg = f"Unsupported activation: {name}"
        raise ValueError(msg)
    return activation


def _build_mlp(
    input_size: int,
    output_size: int,
    hidden_sizes: list[int],
    activation_name: str,
) -> nn.Sequential:
    """Build MLP network.

    Args:
        input_size: Input feature count.
        output_size: Output feature count.
        hidden_sizes: Hidden layer sizes.
        activation_name: Hidden layer activation name.

    Returns:
        Sequential MLP module.

    """
    activation_cls = _get_activation(activation_name)
    layers: list[nn.Module] = []
    in_features = input_size

    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(in_features, hidden_size))
        layers.append(activation_cls())
        in_features = hidden_size

    layers.append(nn.Linear(in_features, output_size))
    return nn.Sequential(*layers)


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


@register_model("mlp_policy")
class MLPPolicy(PolicyModel):
    """MLP policy model."""

    cfg_cls: ClassVar[type[MLPCfg]] = MLPCfg

    def __init__(
        self,
        *,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: MLPCfg,
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
        self.cfg: MLPCfg
        self.policy_net = _build_mlp(
            input_size=self.num_observations,
            output_size=self.num_actions,
            hidden_sizes=self.cfg.net_arch,
            activation_name=self.cfg.activation,
        )
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), self.cfg.log_std_init, device=self.device),
        )

    def compute_categorical(
        self,
        inputs: dict[str, torch.Tensor],
        _role: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute categorical policy logits.

        Args:
            inputs: Model inputs.
            _role: Model role.

        Returns:
            Policy logits and extra outputs.

        """
        return self.policy_net(_get_observations(inputs)), {}

    def compute_gaussian(
        self,
        inputs: dict[str, torch.Tensor],
        _role: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute Gaussian policy parameters.

        Args:
            inputs: Model inputs.
            _role: Model role.

        Returns:
            Mean actions and extra outputs with ``log_std``.

        """
        mean_actions = self.policy_net(_get_observations(inputs))
        log_std = self.log_std_parameter.expand_as(mean_actions)
        return mean_actions, {"log_std": log_std}

    def compute_deterministic(
        self,
        inputs: dict[str, torch.Tensor],
        _role: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute deterministic policy actions.

        Args:
            inputs: Model inputs.
            _role: Model role.

        Returns:
            Actions and extra outputs.

        """
        return self.policy_net(_get_observations(inputs)), {}


@register_model("mlp_value")
class MLPValue(ValueModel):
    """MLP value model."""

    cfg_cls: ClassVar[type[MLPCfg]] = MLPCfg

    def __init__(
        self,
        *,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: MLPCfg,
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
        self.cfg: MLPCfg
        self.value_net = _build_mlp(
            input_size=self.num_observations,
            output_size=1,
            hidden_sizes=self.cfg.net_arch,
            activation_name=self.cfg.activation,
        )

    def compute_value(
        self,
        inputs: dict[str, torch.Tensor],
        _role: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute value estimates.

        Args:
            inputs: Model inputs.
            _role: Model role.

        Returns:
            Value tensor and extra outputs.

        """
        return self.value_net(_get_observations(inputs)), {}
