"""Beta action-distribution policy adapter."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

import torch
from gymnasium.spaces import Box

from spiking_rl_lab.core.validation import require_positive
from spiking_rl_lab.policies.base_policy import BasePolicy
from spiking_rl_lab.policies.builder import register_policy
from spiking_rl_lab.policies.distributions import ActionDistribution

if TYPE_CHECKING:
    import gymnasium


@dataclasses.dataclass(frozen=True, slots=True)
class _BetaDistribution(ActionDistribution):
    """Beta distribution affinely mapped to finite action bounds."""

    distribution: torch.distributions.Beta
    low: torch.Tensor
    scale: torch.Tensor
    reduction: Literal["mean", "sum"]

    def _reduce(self, values: torch.Tensor) -> torch.Tensor:
        if self.reduction == "sum":
            return values.sum(dim=-1, keepdim=True)
        if self.reduction == "mean":
            return values.mean(dim=-1, keepdim=True)
        msg = f"Unsupported Beta log-probability reduction: {self.reduction!r}"
        raise ValueError(msg)

    def sample(self) -> torch.Tensor:
        """Sample an action inside the environment bounds."""
        return self.low + self.scale * self.distribution.sample()

    def mode(self) -> torch.Tensor:
        """Return the distribution mode for deterministic evaluation."""
        normalized = (self.distribution.concentration1 - 1) / (
            self.distribution.concentration1 + self.distribution.concentration0 - 2
        )
        return self.low + self.scale * normalized

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Evaluate bounded actions including the affine Jacobian."""
        normalized = (actions - self.low) / self.scale
        epsilon = torch.finfo(normalized.dtype).eps
        normalized = normalized.clamp(epsilon, 1 - epsilon)
        return self._reduce(self.distribution.log_prob(normalized) - self.scale.log())

    def entropy(self) -> torch.Tensor:
        """Return the exact entropy after affine action scaling."""
        return self._reduce(self.distribution.entropy() + self.scale.log())


@register_policy("beta")
class BetaPolicy(BasePolicy):
    """Bounded Beta policy consuming two concentration features per action."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(BasePolicy.Config):
        """Beta policy configuration."""

        reduction: Literal["mean", "sum"] = "sum"
        min_shape_offset: float = 1e-3

        def __post_init__(self) -> None:
            """Validate the minimum offset from the uniform Beta shape."""
            require_positive("min_shape_offset", self.min_shape_offset)

    def __init__(self, cfg: Config, *, action_space: gymnasium.Space) -> None:
        """Initialize a Beta policy for a finite continuous action space."""
        if not isinstance(action_space, Box):
            msg = "Beta policy requires a Box action space"
            raise TypeError(msg)
        low = torch.as_tensor(action_space.low)
        high = torch.as_tensor(action_space.high)
        if not torch.isfinite(low).all() or not torch.isfinite(high).all():
            msg = "Beta policy requires finite action bounds"
            raise ValueError(msg)
        if not torch.all(high > low):
            msg = "Beta policy requires every upper action bound to exceed its lower bound"
            raise ValueError(msg)

        super().__init__(cfg, action_space=action_space)
        self.register_buffer("_action_low", low, persistent=False)
        self.register_buffer("_action_scale", high - low, persistent=False)

    @property
    def required_output_features(self) -> int:
        """Return two concentration features per action dimension."""
        return 2 * self.num_actions

    def distribution(self, features: torch.Tensor) -> ActionDistribution:
        """Build a bounded Beta distribution from network outputs."""
        if features.shape[-1] != self.required_output_features:
            msg = (
                f"Beta parameters must have {self.required_output_features} features, "
                f"got {features.shape[-1]}"
            )
            raise ValueError(msg)

        raw_alpha, raw_beta = features.chunk(2, dim=-1)
        alpha = 1 + self._cfg.min_shape_offset + torch.nn.functional.softplus(raw_alpha)
        beta = 1 + self._cfg.min_shape_offset + torch.nn.functional.softplus(raw_beta)
        return _BetaDistribution(
            distribution=torch.distributions.Beta(alpha, beta),
            low=self._action_low.to(dtype=features.dtype),
            scale=self._action_scale.to(dtype=features.dtype),
            reduction=self._cfg.reduction,
        )
