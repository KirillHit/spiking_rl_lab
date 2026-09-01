"""A2C agent implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from gymnasium.spaces.utils import flatdim
from skrl.memories.torch import RandomMemory

from spiking_rl_lab.agents.a2c.a2c_cfg import A2CConfig
from spiking_rl_lab.agents.base_agent import BaseAgent
from spiking_rl_lab.agents.builder import register_agent
from spiking_rl_lab.core.exception import AgentCreationError
from spiking_rl_lab.core.validation import require_shape_fields
from spiking_rl_lab.networks.node_network import NodeNetwork
from spiking_rl_lab.networks.shape import DenseTensorShape, TensorShape
from spiking_rl_lab.policies.builder import build_policy

if TYPE_CHECKING:
    from skrl.envs.wrappers.torch import Wrapper
    from skrl.memories.torch import Memory

log = logging.getLogger(__name__)


@register_agent("a2c")
class A2C(BaseAgent):
    """Synchronous advantage actor-critic agent."""

    Config: ClassVar[type[A2CConfig]] = A2CConfig

    def __init__(self, cfg: A2CConfig, *, env: Wrapper) -> None:
        """Build networks, policy adapter, optimizer, and optional utilities."""
        super().__init__(cfg, env=env)

        try:
            input_shape = TensorShape.dense(flatdim(env.observation_space))
            self.policy_network = NodeNetwork(cfg.policy_network, input_shape=input_shape).to(
                cfg.device
            )
            self.value_network = NodeNetwork(cfg.value_network, input_shape=input_shape).to(
                cfg.device
            )
            require_shape_fields(
                "A2C policy network output",
                self.policy_network.output_shape,
                shape_type=DenseTensorShape,
                fields={"features": flatdim(env.action_space)},
            )
            require_shape_fields(
                "A2C value network output",
                self.value_network.output_shape,
                shape_type=DenseTensorShape,
                fields={"features": 1},
            )
            self.policy = build_policy(cfg.policy, action_space=env.action_space).to(cfg.device)
        except Exception as exc:
            msg = "Failed to create A2C components"
            raise AgentCreationError(msg) from exc

        self._parameters = tuple(
            parameter
            for module in (self.policy_network, self.value_network, self.policy)
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        self.optimizer = torch.optim.Adamax(self._parameters, lr=cfg.learning_rate)
        self.checkpoint_modules.update(
            policy_network=self.policy_network,
            value_network=self.value_network,
            policy=self.policy,
            optimizer=self.optimizer,
        )

        self.scheduler = None
        if cfg.learning_rate_scheduler is not None:
            self.scheduler = cfg.learning_rate_scheduler(
                self.optimizer, **cfg.learning_rate_scheduler_kwargs
            )
            self.checkpoint_modules["scheduler"] = self.scheduler

        if cfg.observation_preprocessor is None:
            self._observation_preprocessor = self._empty_preprocessor
        elif env.num_envs == 1:
            log.warning(
                "Disabling observation preprocessor: a single environment does not provide "
                "a batch for updating preprocessing statistics"
            )
            self._observation_preprocessor = self._empty_preprocessor
        else:
            kwargs = dict(cfg.observation_preprocessor_kwargs)
            kwargs.setdefault("size", self.observation_space)
            kwargs.setdefault("device", self.device)
            self._observation_preprocessor = cfg.observation_preprocessor(**kwargs).to(self.device)
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor

    def build_memory(self, *, env: Wrapper) -> Memory:
        """Build storage for one rollout."""
        return RandomMemory(
            memory_size=self.cfg.rollouts,
            num_envs=env.num_envs,
            device=self.device,
        )

    def act(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Select actions from the current policy."""
        raise NotImplementedError

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Prepare the agent before an environment interaction."""
        raise NotImplementedError

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Process the completed environment interaction."""
        raise NotImplementedError

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Update the policy and value networks from the collected rollout."""
        raise NotImplementedError
