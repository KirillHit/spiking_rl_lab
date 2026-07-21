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
    import gymnasium


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


def _clip_to_action_bounds(
    actions: torch.Tensor,
    *,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Clip actions using bounds already stored by the policy module."""
    return actions.clamp(low.to(dtype=actions.dtype), high.to(dtype=actions.dtype))


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

    def distribution(self, features: torch.Tensor) -> ActionDistribution:
        """Build a categorical action distribution from unnormalised logits."""
        logits = features
        if logits.shape[-1] != self.num_actions:
            msg = (
                f"Categorical logits must have {self.num_actions} features, got {logits.shape[-1]}"
            )
            raise ValueError(msg)
        return _CategoricalDistribution(torch.distributions.Categorical(logits=logits))


@register_policy("gaussian")
class GaussianPolicy(BasePolicy):
    """Gaussian policy with a learned, state-independent log standard deviation."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BasePolicy.Config):
        """Gaussian policy configuration."""

        clip_mean_actions: bool = False
        clip_log_std: bool = True
        min_log_std: float = -20
        max_log_std: float = 2
        initial_log_std: float = 0.0
        reduction: Literal["mean", "sum"] = "sum"

    def __init__(self, cfg: Config, *, action_space: gymnasium.Space) -> None:
        """Initialize a Gaussian policy for a continuous action space."""
        self._action_space = _require_box_action_space(action_space)
        super().__init__(cfg, action_space=action_space)
        self.log_std = torch.nn.Parameter(torch.full((self.num_actions,), cfg.initial_log_std))
        self.register_buffer(
            "_action_low", torch.as_tensor(self._action_space.low), persistent=False
        )
        self.register_buffer(
            "_action_high", torch.as_tensor(self._action_space.high), persistent=False
        )

    def distribution(self, features: torch.Tensor) -> ActionDistribution:
        """Build a normal action distribution from network outputs."""
        mean_actions = features
        if mean_actions.shape[-1] != self.num_actions:
            msg = (
                f"Gaussian means must have {self.num_actions} features, "
                f"got {mean_actions.shape[-1]}"
            )
            raise ValueError(msg)
        log_std = self.log_std.expand_as(mean_actions)
        if self._cfg.clip_log_std:
            log_std = log_std.clamp(self._cfg.min_log_std, self._cfg.max_log_std)
        if self._cfg.clip_mean_actions:
            mean_actions = _clip_to_action_bounds(
                mean_actions,
                low=self._action_low,
                high=self._action_high,
            )
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
        self.register_buffer(
            "_action_low", torch.as_tensor(self._action_space.low), persistent=False
        )
        self.register_buffer(
            "_action_high", torch.as_tensor(self._action_space.high), persistent=False
        )

    def _actions(self, features: torch.Tensor) -> torch.Tensor:
        actions = features
        if actions.shape[-1] != self.num_actions:
            msg = (
                f"Deterministic actions must have {self.num_actions} features, "
                f"got {actions.shape[-1]}"
            )
            raise ValueError(msg)
        if self._cfg.clip_actions:
            actions = _clip_to_action_bounds(
                actions,
                low=self._action_low,
                high=self._action_high,
            )
        return actions

    def distribution(self, features: torch.Tensor) -> ActionDistribution:
        """Build a deterministic action distribution from network outputs."""
        return _DeterministicDistribution(self._actions(features))
