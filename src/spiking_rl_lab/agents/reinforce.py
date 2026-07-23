"""REINFORCE agent implementation."""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from gymnasium.spaces import Discrete
from gymnasium.spaces.utils import flatdim
from omegaconf import MISSING
from skrl.memories.torch import RandomMemory

from spiking_rl_lab.agents.base_agent import BaseAgent
from spiking_rl_lab.agents.builder import register_agent
from spiking_rl_lab.core.exception import AgentCreationError
from spiking_rl_lab.core.validation import (
    require_minimum,
    require_optional_callable,
    require_optional_class,
    require_positive,
    require_range,
    require_shape_fields,
)
from spiking_rl_lab.networks.node_network import NodeNetwork, NodeNetworkConfig
from spiking_rl_lab.networks.shape import DenseTensorShape, TensorShape
from spiking_rl_lab.networks.state import ListState, detach_state
from spiking_rl_lab.policies.builder import PolicyConfig, build_policy

if TYPE_CHECKING:
    from collections.abc import Callable

    from skrl.envs.wrappers.torch import Wrapper
    from skrl.memories.torch import Memory

log = logging.getLogger(__name__)


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


@register_agent("reinforce")
class Reinforce(BaseAgent):
    """Monte Carlo policy-gradient agent with truncated BPTT for stateful networks."""

    Config: ClassVar[type[ReinforceConfig]] = ReinforceConfig

    def __init__(self, cfg: ReinforceConfig, *, env: Wrapper) -> None:
        """Build the policy network, policy adapter, and optimizer."""
        super().__init__(cfg, env=env)

        try:
            self.policy_network = NodeNetwork(
                cfg.policy_network,
                input_shape=TensorShape.dense(flatdim(env.observation_space)),
            ).to(cfg.device)
            require_shape_fields(
                "REINFORCE policy network output",
                self.policy_network.output_shape,
                shape_type=DenseTensorShape,
                fields={"features": flatdim(env.action_space)},
            )
            self.policy = build_policy(cfg.policy, action_space=env.action_space).to(cfg.device)
        except Exception as exc:
            msg = "Failed to create REINFORCE policy components"
            raise AgentCreationError(msg) from exc

        self._parameters = tuple(
            parameter
            for module in (self.policy_network, self.policy)
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        self.optimizer = torch.optim.Adamax(self._parameters, lr=cfg.learning_rate)

        self.checkpoint_modules.update(
            policy_network=self.policy_network, policy=self.policy, optimizer=self.optimizer
        )

        self.scheduler = None
        if cfg.learning_rate_scheduler is not None:
            self.scheduler = cfg.learning_rate_scheduler(
                self.optimizer,
                **cfg.learning_rate_scheduler_kwargs,
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

        self._hidden_states: ListState | None = None
        self._rollout_initial_state: ListState | None = None
        self._processed_observation: torch.Tensor | None = None

    def build_memory(self, *, env: Wrapper) -> Memory:
        """Build storage for one policy rollout."""
        return RandomMemory(
            memory_size=self.cfg.rollouts,
            num_envs=env.num_envs,
            device=self.device,
        )

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize rollout storage and keep policy modules deterministic for replay."""
        super().init(trainer_cfg=trainer_cfg)
        self.policy_network.eval()
        self.policy.eval()

        self.memory.create_tensor(
            name="observations", size=flatdim(self.observation_space), dtype=torch.float32
        )
        action_dtype = torch.int64 if isinstance(self.action_space, Discrete) else torch.float32
        self.memory.create_tensor(name="actions", size=self.action_space, dtype=action_dtype)
        self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="dones", size=1, dtype=torch.bool)
        self._hidden_states = None
        self._reset_rollout()

    def _reset_rollout(self) -> None:
        """Clear rollout storage without resetting the live network state."""
        self.memory.reset()
        self._rollout_initial_state = None
        self._processed_observation = None

    def act(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample an action and advance the policy-network state."""
        processed = self._observation_preprocessor(observations, train=self.training)
        inputs = torch.flatten(processed, start_dim=1)
        if self.training:
            self._processed_observation = inputs

        with torch.no_grad():
            if self._hidden_states is None:
                self._hidden_states = self.policy_network.initial_state(inputs)
            if self.training and self._rollout_initial_state is None:
                self._rollout_initial_state = self._hidden_states
            features, self._hidden_states = self.policy_network(inputs, self._hidden_states)
            distribution = self.policy.distribution(features)
            actions = distribution.sample() if self.training else distribution.mode()

        return actions, {}

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: object,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Track the interaction, reset done states, and store policy transitions."""
        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        dones = torch.logical_or(terminated, truncated)
        self._hidden_states = self.policy_network.reset_state(self._hidden_states, dones)
        if not self.training:
            return

        if self.cfg.rewards_shaper is not None:
            rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)
        self.memory.add_samples(
            observations=self._processed_observation,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )
        self._processed_observation = None

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Run the hook before environment interaction."""

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Update the policy as soon as the rollout storage is full."""
        if self.training and self.memory.filled:
            started_at = time.perf_counter()
            self.update(timestep=timestep, timesteps=timesteps)
            self.track_data(
                "Stats / Algorithm update time (ms)",
                (time.perf_counter() - started_at) * 1_000,
            )
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def _discounted_returns(self, rollout_steps: int) -> torch.Tensor:
        """Compute normalized Monte Carlo returns for the stored rollout."""
        rewards = self.memory.get_tensor_by_name("rewards")[:rollout_steps]
        dones = self.memory.get_tensor_by_name("dones")[:rollout_steps]
        returns = torch.zeros_like(rewards)
        future_return = torch.zeros((self.memory.num_envs, 1), device=self.device)

        for step in range(rollout_steps - 1, -1, -1):
            future_return = rewards[step] + (
                self.cfg.discount_factor * future_return * (~dones[step]).float()
            )
            returns[step] = future_return

        if self.cfg.normalize_returns:
            returns = (returns - returns.mean()) / returns.std(correction=0).clamp_min(1e-8)
        return returns

    def _loss(self, rollout_steps: int, returns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay the rollout sequentially and compute its truncated-BPTT loss."""
        observations = self.memory.get_tensor_by_name("observations")
        actions = self.memory.get_tensor_by_name("actions")
        dones = self.memory.get_tensor_by_name("dones")
        policy_terms = []
        entropy_terms = []

        hidden_states = self._rollout_initial_state
        for step in range(rollout_steps):
            features, hidden_states = self.policy_network(observations[step], hidden_states)
            distribution = self.policy.distribution(features)
            log_prob = distribution.log_prob(actions[step])
            policy_terms.append(-(returns[step] * log_prob).mean())
            if self.cfg.entropy_loss_scale:
                entropy_terms.append(-self.cfg.entropy_loss_scale * distribution.entropy().mean())
            hidden_states = self.policy_network.reset_state(hidden_states, dones[step])
            if (step + 1) % self.cfg.sequence_length == 0:
                hidden_states = detach_state(hidden_states)

        policy_loss = torch.stack(policy_terms).mean()
        entropy_loss = (
            torch.stack(entropy_terms).mean()
            if entropy_terms
            else torch.zeros((), device=self.device)
        )
        return policy_loss, entropy_loss

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Run one Monte Carlo policy-gradient update."""
        rollout_steps = self.memory.memory_size if self.memory.filled else self.memory.memory_index
        if not rollout_steps:
            return

        policy_loss, entropy_loss = self._loss(
            rollout_steps,
            self._discounted_returns(rollout_steps),
        )
        self.optimizer.zero_grad(set_to_none=True)
        (policy_loss + entropy_loss).backward()

        if self.cfg.grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(self._parameters, self.cfg.grad_norm_clip)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.track_data("Loss / Policy loss", policy_loss.item())
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", entropy_loss.item())
        if self.scheduler is not None:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

        self._reset_rollout()
