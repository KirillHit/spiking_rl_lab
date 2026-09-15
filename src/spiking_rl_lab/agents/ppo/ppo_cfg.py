"""PPO agent configuration."""

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
class PPOConfig(BaseAgent.Config):
    """Configuration for the PPO agent."""

    policy_network: NodeNetworkConfig = MISSING
    """Network that produces policy distribution parameters."""

    value_network: NodeNetworkConfig = MISSING
    """Network that estimates state values."""

    policy: PolicyConfig = MISSING
    """Policy adapter that interprets policy network outputs."""

    rollouts: int = 16
    """Number of transitions stored in each rollout."""

    sequence_length: int = 16
    """Transitions per independently replayed recurrent sequence."""

    discount_factor: float = 0.99
    """Reward discount factor used to compute returns."""

    gae_lambda: float = 0.95
    """Lambda coefficient used by generalized advantage estimation."""

    policy_learning_rate: float = 1e-3
    """Learning rate for the policy."""

    value_learning_rate: float = 1e-3
    """Learning rate for the value network."""

    policy_learning_rate_scheduler: str | type[Any] | None = None
    """Optional policy learning rate scheduler class or dotted import path."""

    policy_learning_rate_scheduler_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Keyword arguments passed to the policy learning rate scheduler."""

    value_learning_rate_scheduler: str | type[Any] | None = None
    """Optional value learning rate scheduler class or dotted import path."""

    value_learning_rate_scheduler_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Keyword arguments passed to the value learning rate scheduler."""

    observation_preprocessor: str | type[Any] | None = None
    """Optional observation preprocessor class or dotted import path."""

    observation_preprocessor_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Keyword arguments passed to ``observation_preprocessor`` during construction."""

    policy_grad_norm_clip: float = 0.5
    """Maximum policy gradient norm. Set to ``0`` to disable clipping."""

    value_grad_norm_clip: float = 0.5
    """Maximum value gradient norm. Set to ``0`` to disable clipping."""

    entropy_loss_scale: float = 0.0
    """Entropy regularization coefficient added to the policy loss."""

    spike_activity_loss_scale: float = 0.0
    """Mean squared neuron firing-rate penalty coefficient. Set to ``0`` to disable."""

    time_limit_bootstrap: bool = False
    """Whether to bootstrap returns at time-limit truncations."""

    rewards_shaper: str | Callable[..., Any] | None = None
    """Optional reward-shaping callable or dotted import path."""

    learning_epochs: int = 8
    """Number of optimization passes over each rollout."""

    mini_batches: int = 2
    """Number of groups formed from complete recurrent sequences."""

    ratio_clip: float = 0.2
    """Clipping radius for the policy probability ratio."""

    value_clip: float = 0.2
    """Clipping radius for value predictions; zero disables clipping."""

    value_loss_scale: float = 1.0
    """Value loss coefficient."""

    kl_threshold: float = 0.0
    """Stop later epochs when full-rollout KL exceeds this value; zero disables."""

    def __post_init__(self) -> None:
        """Validate PPO hyperparameters."""
        if not isinstance(self.policy_network, NodeNetworkConfig):
            self.policy_network = NodeNetworkConfig(**self.policy_network)
        if not isinstance(self.value_network, NodeNetworkConfig):
            self.value_network = NodeNetworkConfig(**self.value_network)
        if not isinstance(self.policy, PolicyConfig):
            self.policy = PolicyConfig(**self.policy)

        require_minimum("learning_epochs", self.learning_epochs, minimum=1)
        require_minimum("mini_batches", self.mini_batches, minimum=1)
        require_range("ratio_clip", self.ratio_clip, minimum=0.0, maximum=1.0)
        require_minimum("value_clip", self.value_clip, minimum=0.0)
        require_minimum("value_loss_scale", self.value_loss_scale, minimum=0.0)
        require_minimum("kl_threshold", self.kl_threshold, minimum=0.0)
        require_minimum("rollouts", self.rollouts, minimum=1)
        require_minimum("sequence_length", self.sequence_length, minimum=1)
        if self.rollouts % self.sequence_length:
            msg = "rollouts must be divisible by sequence_length"
            raise ValueError(msg)
        require_range("discount_factor", self.discount_factor, minimum=0.0, maximum=1.0)
        require_range("gae_lambda", self.gae_lambda, minimum=0.0, maximum=1.0)
        require_positive("policy_learning_rate", self.policy_learning_rate)
        require_positive("value_learning_rate", self.value_learning_rate)
        require_minimum("policy_grad_norm_clip", self.policy_grad_norm_clip, minimum=0.0)
        require_minimum("value_grad_norm_clip", self.value_grad_norm_clip, minimum=0.0)
        require_minimum("entropy_loss_scale", self.entropy_loss_scale, minimum=0.0)
        require_minimum("spike_activity_loss_scale", self.spike_activity_loss_scale, minimum=0.0)
        self.policy_learning_rate_scheduler = require_optional_class(
            "policy_learning_rate_scheduler", self.policy_learning_rate_scheduler
        )
        self.value_learning_rate_scheduler = require_optional_class(
            "value_learning_rate_scheduler", self.value_learning_rate_scheduler
        )
        self.observation_preprocessor = require_optional_class(
            "observation_preprocessor", self.observation_preprocessor
        )
        self.rewards_shaper = require_optional_callable("rewards_shaper", self.rewards_shaper)
