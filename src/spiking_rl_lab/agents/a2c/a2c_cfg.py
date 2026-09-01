"""A2C agent configuration."""

from __future__ import annotations

import dataclasses
from typing import Any

from omegaconf import MISSING

from spiking_rl_lab.agents.base_agent import BaseAgent
from spiking_rl_lab.core.validation import require_minimum, require_optional_class, require_positive
from spiking_rl_lab.networks.node_network import NodeNetworkConfig
from spiking_rl_lab.policies.builder import PolicyConfig


@dataclasses.dataclass(kw_only=True, slots=True)
class A2CConfig(BaseAgent.Config):
    """Configuration for the A2C agent."""

    policy_network: NodeNetworkConfig = MISSING
    """Network that produces policy distribution parameters."""

    value_network: NodeNetworkConfig = MISSING
    """Network that estimates state values."""

    policy: PolicyConfig = MISSING
    """Policy adapter that interprets policy network outputs."""

    rollouts: int = 16
    """Number of transitions stored in each rollout."""

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

    def __post_init__(self) -> None:
        """Validate A2C hyperparameters."""
        if not isinstance(self.policy_network, NodeNetworkConfig):
            self.policy_network = NodeNetworkConfig(**self.policy_network)
        if not isinstance(self.value_network, NodeNetworkConfig):
            self.value_network = NodeNetworkConfig(**self.value_network)
        if not isinstance(self.policy, PolicyConfig):
            self.policy = PolicyConfig(**self.policy)

        require_minimum("rollouts", self.rollouts, minimum=1)
        require_positive("learning_rate", self.learning_rate)
        self.learning_rate_scheduler = require_optional_class(
            "learning_rate_scheduler", self.learning_rate_scheduler
        )
        self.observation_preprocessor = require_optional_class(
            "observation_preprocessor", self.observation_preprocessor
        )
