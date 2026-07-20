"""Standard action-distribution policy adapters."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

import torch
from gymnasium.spaces import Box, Discrete

from spiking_rl_lab.policies.base_policy import BasePolicy
from spiking_rl_lab.policies.builder import register_policy
from spiking_rl_lab.policies.distributions import ActionDistribution

if TYPE_CHECKING:
    from collections.abc import Mapping

    import gymnasium


def _parameter(parameters: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    """Get a required distribution parameter with a clear error."""
    try:
        return parameters[name]
    except KeyError as exc:
        msg = f"Policy parameters must contain '{name}'"
        raise KeyError(msg) from exc


def _action_bounds(
    action_space: Box,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Box bounds as tensors on the specified device and dtype."""
    return (
        torch.as_tensor(
            action_space.low,
            device=device,
            dtype=dtype,
        ),
        torch.as_tensor(
            action_space.high,
            device=device,
            dtype=dtype,
        ),
    )


def _require_discrete_action_space(action_space: gymnasium.Space) -> Discrete:
    """Raise a clear error when a categorical policy gets an incompatible space."""
    if not isinstance(action_space, Discrete):
        msg = "Categorical policy requires a Discrete action space"
        raise TypeError(msg)
    return action_space


def _require_box_action_space(action_space: gymnasium.Space) -> Box:
    """Raise a clear error when a continuous policy gets an incompatible space."""
    if not isinstance(action_space, Box):
        msg = "Continuous policy requires a Box action space"
        raise TypeError(msg)
    return action_space


@dataclasses.dataclass(frozen=True, slots=True)
class _CategoricalDistribution(ActionDistribution):
    """Categorical distribution adapted to Gym's discrete action shape."""

    distribution: torch.distributions.Categorical

    def sample(self) -> torch.Tensor:
        """Sample actions as ``[batch, 1]`` tensors."""
        return self.distribution.sample().unsqueeze(-1)

    def mode(self) -> torch.Tensor:
        """Return the most likely discrete action."""
        return self.distribution.probs.argmax(dim=-1, keepdim=True)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Evaluate discrete actions."""
        actions = actions.squeeze(-1) if actions.ndim > 1 else actions
        return self.distribution.log_prob(actions).unsqueeze(-1)

    def entropy(self) -> torch.Tensor:
        """Return categorical entropy."""
        return self.distribution.entropy().unsqueeze(-1)


@dataclasses.dataclass(frozen=True, slots=True)
class _GaussianDistribution(ActionDistribution):
    """Normal distribution with a scalar log-probability per action."""

    distribution: torch.distributions.Normal
    reduction: Literal["mean", "sum"]

    def _reduce(self, values: torch.Tensor) -> torch.Tensor:
        if self.reduction == "sum":
            return values.sum(dim=-1, keepdim=True)
        if self.reduction == "mean":
            return values.mean(dim=-1, keepdim=True)
        msg = f"Unsupported Gaussian log-probability reduction: {self.reduction!r}"
        raise ValueError(msg)

    def sample(self) -> torch.Tensor:
        """Sample a continuous action."""
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        """Return the mean action."""
        return self.distribution.mean

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Evaluate continuous actions."""
        return self._reduce(self.distribution.log_prob(actions))

    def entropy(self) -> torch.Tensor:
        """Return reduced normal entropy."""
        return self._reduce(self.distribution.entropy())


@dataclasses.dataclass(frozen=True, slots=True)
class _DeterministicDistribution(ActionDistribution):
    """Degenerate action distribution for deterministic policies."""

    actions: torch.Tensor

    def sample(self) -> torch.Tensor:
        """Return network actions."""
        return self.actions

    def mode(self) -> torch.Tensor:
        """Return network actions."""
        return self.actions

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Return zero log-probabilities for deterministic actions."""
        del actions
        return self._zeros()

    def entropy(self) -> torch.Tensor:
        """Return zero entropy for deterministic actions."""
        return self._zeros()

    def _zeros(self) -> torch.Tensor:
        return torch.zeros(
            (*self.actions.shape[:-1], 1),
            device=self.actions.device,
            dtype=self.actions.dtype,
        )


@register_policy("categorical")
class CategoricalPolicy(BasePolicy):
    """Categorical policy consuming network-produced action logits."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BasePolicy.Config):
        """Categorical policy configuration."""

    def __init__(self, cfg: Config, *, action_space: gymnasium.Space) -> None:
        """Initialize a categorical policy for a discrete action space."""
        _require_discrete_action_space(action_space)
        super().__init__(cfg, action_space=action_space)

    def distribution(
        self,
        parameters: Mapping[str, torch.Tensor],
    ) -> ActionDistribution:
        """Build a categorical action distribution from unnormalised logits."""
        logits = _parameter(parameters, "logits")
        if logits.shape[-1] != self.num_actions:
            msg = (
                f"Categorical logits must have {self.num_actions} features, got {logits.shape[-1]}"
            )
            raise ValueError(msg)
        return _CategoricalDistribution(torch.distributions.Categorical(logits=logits))


@register_policy("gaussian")
class GaussianPolicy(BasePolicy):
    """Gaussian policy consuming explicit mean and log-standard-deviation tensors."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BasePolicy.Config):
        """Gaussian policy configuration."""

        clip_mean_actions: bool = False
        clip_log_std: bool = True
        min_log_std: float = -20
        max_log_std: float = 2
        reduction: Literal["mean", "sum"] = "sum"

    def __init__(self, cfg: Config, *, action_space: gymnasium.Space) -> None:
        """Initialize a Gaussian policy for a continuous action space."""
        self._action_space = _require_box_action_space(action_space)
        super().__init__(cfg, action_space=action_space)

    def distribution(self, parameters: Mapping[str, torch.Tensor]) -> ActionDistribution:
        """Build a normal action distribution from network outputs."""
        mean_actions = _parameter(parameters, "mean_actions")
        log_std = _parameter(parameters, "log_std")
        if mean_actions.shape[-1] != self.num_actions:
            msg = (
                f"Gaussian means must have {self.num_actions} features, "
                f"got {mean_actions.shape[-1]}"
            )
            raise ValueError(msg)
        if log_std.shape != mean_actions.shape:
            msg = "Gaussian log_std must have the same shape as mean_actions"
            raise ValueError(msg)
        if self._cfg.clip_log_std:
            log_std = log_std.clamp(self._cfg.min_log_std, self._cfg.max_log_std)
        if self._cfg.clip_mean_actions:
            bounds = _action_bounds(
                self._action_space,
                device=mean_actions.device,
                dtype=mean_actions.dtype,
            )
            mean_actions = mean_actions.clamp(*bounds)
        return _GaussianDistribution(
            distribution=torch.distributions.Normal(mean_actions, log_std.exp()),
            reduction=self._cfg.reduction,
        )


@register_policy("deterministic")
class DeterministicPolicy(BasePolicy):
    """Deterministic policy consuming network-produced actions."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BasePolicy.Config):
        """Deterministic policy configuration."""

        clip_actions: bool = False

    def __init__(self, cfg: Config, *, action_space: gymnasium.Space) -> None:
        """Initialize a deterministic policy for a continuous action space."""
        self._action_space = _require_box_action_space(action_space)
        super().__init__(cfg, action_space=action_space)

    def _actions(
        self,
        parameters: Mapping[str, torch.Tensor],
        action_space: Box,
    ) -> torch.Tensor:
        actions = _parameter(parameters, "actions")
        if actions.shape[-1] != self.num_actions:
            msg = (
                f"Deterministic actions must have {self.num_actions} features, "
                f"got {actions.shape[-1]}"
            )
            raise ValueError(msg)
        if self._cfg.clip_actions:
            bounds = _action_bounds(
                action_space,
                device=actions.device,
                dtype=actions.dtype,
            )
            actions = actions.clamp(*bounds)
        return actions

    def distribution(self, parameters: Mapping[str, torch.Tensor]) -> ActionDistribution:
        """Build a deterministic action distribution from network outputs."""
        return _DeterministicDistribution(self._actions(parameters, self._action_space))
