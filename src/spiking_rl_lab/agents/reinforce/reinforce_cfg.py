"""REINFORCE agent configuration."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from omegaconf import MISSING

from spiking_rl_lab.agents.base_agent import BaseAgent
from spiking_rl_lab.core.validation import (
    require_minimum,
    require_optional_callable,
    require_optional_class,
    require_positive,
    require_range,
)
from spiking_rl_lab.networks.node_network import NodeNetworkConfig
from spiking_rl_lab.policies.builder import PolicyConfig

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclasses.dataclass(kw_only=True, slots=True)
class ReinforceConfig(BaseAgent.Config):
    """Configuration for the REINFORCE agent."""

    policy_network: NodeNetworkConfig = MISSING
    """Network that produces policy distribution parameters."""

    policy: PolicyConfig = MISSING
    """Policy adapter that interprets network outputs."""

    rollouts: int = 16
    """Number of policy transitions collected before each update."""

    sequence_length: int = 16
    """Maximum number of transitions in one truncated-BPTT window."""

    discount_factor: float = 0.99
    """Reward discount factor used to compute Monte Carlo returns."""

    learning_rate: float = 1e-3
    """Adamax optimizer learning rate."""

    learning_rate_scheduler: str | type[Any] | None = None
    """Optional learning rate scheduler class or dotted import path."""

    learning_rate_scheduler_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Keyword arguments passed to ``learning_rate_scheduler`` during construction."""

    observation_preprocessor: str | type[Any] | None = None
    """Optional observation preprocessor class or dotted import path."""

    observation_preprocessor_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Keyword arguments passed to ``observation_preprocessor`` during construction."""

    grad_norm_clip: float = 0.5
    """Maximum gradient norm. Set to ``0`` to disable clipping."""

    entropy_loss_scale: float = 0.0
    """Entropy regularization coefficient added to the policy loss."""

    rewards_shaper: str | Callable[..., Any] | None = None
    """Optional reward-shaping callable or dotted import path."""

    normalize_returns: bool = True
    """Whether to normalize returns across the collected rollout."""

    def __post_init__(self) -> None:
        """Validate REINFORCE hyperparameters after dataclass initialization."""
        if not isinstance(self.policy_network, NodeNetworkConfig):
            self.policy_network = NodeNetworkConfig(**self.policy_network)
        if not isinstance(self.policy, PolicyConfig):
            self.policy = PolicyConfig(**self.policy)
        require_minimum("rollouts", self.rollouts, minimum=1)
        require_minimum("sequence_length", self.sequence_length, minimum=1)
        require_range("discount_factor", self.discount_factor, minimum=0.0, maximum=1.0)
        require_positive("learning_rate", self.learning_rate)
        require_minimum("grad_norm_clip", self.grad_norm_clip, minimum=0.0)
        require_minimum("entropy_loss_scale", self.entropy_loss_scale, minimum=0.0)
        self.learning_rate_scheduler = require_optional_class(
            "learning_rate_scheduler",
            self.learning_rate_scheduler,
        )
        self.observation_preprocessor = require_optional_class(
            "observation_preprocessor",
            self.observation_preprocessor,
        )
        self.rewards_shaper = require_optional_callable("rewards_shaper", self.rewards_shaper)
