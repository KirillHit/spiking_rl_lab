"""A2C agent implementation."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from gymnasium.spaces import Discrete
from gymnasium.spaces.utils import flatdim
from skrl.memories.torch import RandomMemory

from spiking_rl_lab.agents.a2c.a2c_cfg import A2CConfig
from spiking_rl_lab.agents.base_agent import BaseAgent
from spiking_rl_lab.agents.builder import register_agent
from spiking_rl_lab.core.exception import AgentCreationError
from spiking_rl_lab.core.validation import require_shape_fields
from spiking_rl_lab.networks.node_network import NodeNetwork
from spiking_rl_lab.networks.shape import DenseTensorShape, TensorShape
from spiking_rl_lab.networks.state import ListState, detach_state
from spiking_rl_lab.policies.builder import build_policy

if TYPE_CHECKING:
    from skrl.envs.wrappers.torch import Wrapper
    from skrl.memories.torch import Memory

log = logging.getLogger(__name__)


def compute_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    discount_factor: float,
    lambda_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute returns and normalized generalized advantages."""
    advantages = torch.zeros_like(rewards)
    advantage = torch.zeros_like(last_values)
    not_done = ~(terminated | truncated)

    for step in range(rewards.shape[0] - 1, -1, -1):
        next_values = values[step + 1] if step < rewards.shape[0] - 1 else last_values
        advantage = (
            rewards[step]
            - values[step]
            + discount_factor * not_done[step] * (next_values + lambda_coefficient * advantage)
        )
        advantages[step] = advantage

    returns = advantages + values
    advantages = (advantages - advantages.mean()) / advantages.std(correction=0).clamp_min(1e-8)
    return returns, advantages


@register_agent("a2c")
class A2C(BaseAgent):
    """Synchronous advantage actor-critic agent."""

    Config: ClassVar[type[A2CConfig]] = A2CConfig

    def __init__(self, cfg: A2CConfig, *, env: Wrapper) -> None:
        """Build networks, policy adapter, optimizer, and optional utilities."""
        super().__init__(cfg, env=env)

        try:
            self.policy = build_policy(cfg.policy, action_space=env.action_space).to(cfg.device)
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
                fields={"features": self.policy.required_output_features},
            )
            require_shape_fields(
                "A2C value network output",
                self.value_network.output_shape,
                shape_type=DenseTensorShape,
                fields={"features": 1},
            )
        except Exception as exc:
            msg = "Failed to create A2C components"
            raise AgentCreationError(msg) from exc

        self._policy_parameters = tuple(
            parameter
            for module in (self.policy_network, self.policy)
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        self._value_parameters = tuple(
            parameter for parameter in self.value_network.parameters() if parameter.requires_grad
        )
        self.policy_optimizer = torch.optim.Adamax(
            self._policy_parameters, lr=cfg.policy_learning_rate
        )
        self.value_optimizer = torch.optim.Adamax(
            self._value_parameters, lr=cfg.value_learning_rate
        )
        self.checkpoint_modules.update(
            policy_network=self.policy_network,
            value_network=self.value_network,
            policy=self.policy,
            policy_optimizer=self.policy_optimizer,
            value_optimizer=self.value_optimizer,
        )

        self.policy_scheduler = None
        if cfg.policy_learning_rate_scheduler is not None:
            self.policy_scheduler = cfg.policy_learning_rate_scheduler(
                self.policy_optimizer, **cfg.policy_learning_rate_scheduler_kwargs
            )
            self.checkpoint_modules["policy_scheduler"] = self.policy_scheduler

        self.value_scheduler = None
        if cfg.value_learning_rate_scheduler is not None:
            self.value_scheduler = cfg.value_learning_rate_scheduler(
                self.value_optimizer, **cfg.value_learning_rate_scheduler_kwargs
            )
            self.checkpoint_modules["value_scheduler"] = self.value_scheduler

        if cfg.observation_preprocessor is None:
            self._observation_preprocessor = self._empty_preprocessor
        else:
            kwargs = dict(cfg.observation_preprocessor_kwargs)
            kwargs.setdefault("size", self.observation_space)
            kwargs.setdefault("device", self.device)
            self._observation_preprocessor = cfg.observation_preprocessor(**kwargs).to(self.device)
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor

        self._policy_state: ListState | None = None
        self._value_state: ListState | None = None
        self._rollout_policy_state: ListState | None = None
        self._rollout_value_state: ListState | None = None
        self._processed_observation: torch.Tensor | None = None
        self._next_observation: torch.Tensor | None = None
        self._current_value: torch.Tensor | None = None

    def build_memory(self, *, env: Wrapper) -> Memory:
        """Build storage for one rollout."""
        return RandomMemory(
            memory_size=self.cfg.rollouts,
            num_envs=env.num_envs,
            device=self.device,
        )

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize rollout storage and network states."""
        super().init(trainer_cfg=trainer_cfg)
        self.policy_network.eval()
        self.value_network.eval()
        self.policy.eval()

        self.memory.create_tensor(
            name="observations", size=flatdim(self.observation_space), dtype=torch.float32
        )
        action_dtype = torch.int64 if isinstance(self.action_space, Discrete) else torch.float32
        self.memory.create_tensor(name="actions", size=self.action_space, dtype=action_dtype)
        self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
        self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
        self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
        self._policy_state = None
        self._value_state = None
        self._reset_rollout()

    def _reset_rollout(self) -> None:
        """Clear rollout storage without resetting live network states."""
        self.memory.reset()
        self._rollout_policy_state = None
        self._rollout_value_state = None
        self._processed_observation = None
        self._next_observation = None
        self._current_value = None

    def act(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample an action and advance the policy and value network states."""
        processed = self._observation_preprocessor(observations, train=self.training)
        inputs = torch.flatten(processed, start_dim=1)
        if self.training:
            self._processed_observation = inputs

        with torch.no_grad():
            if self._policy_state is None:
                self._policy_state = self.policy_network.initial_state(inputs)
            if self._value_state is None:
                self._value_state = self.value_network.initial_state(inputs)
            if self.training and self._rollout_policy_state is None:
                self._rollout_policy_state = self._policy_state
                self._rollout_value_state = self._value_state

            policy_features, self._policy_state = self.policy_network(inputs, self._policy_state)
            values, self._value_state = self.value_network(inputs, self._value_state)
            distribution = self.policy.distribution(policy_features)
            actions = distribution.sample() if self.training else distribution.mode()
            if self.training:
                self._current_value = values

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
        """Track an interaction, store it, and reset completed network states."""
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
        if self.training:
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)
            if self.cfg.time_limit_bootstrap and truncated.any():
                next_inputs = torch.flatten(
                    self._observation_preprocessor(next_observations, train=False), start_dim=1
                )
                with torch.no_grad():
                    next_values, _ = self.value_network(next_inputs, self._value_state)
                rewards = rewards + self.cfg.discount_factor * next_values * truncated

            self.memory.add_samples(
                observations=self._processed_observation,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                values=self._current_value,
            )
            self._next_observation = next_observations

        self._policy_state = self.policy_network.reset_state(self._policy_state, dones)
        self._value_state = self.value_network.reset_state(self._value_state, dones)
        self._processed_observation = None
        self._current_value = None

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Prepare the agent before an environment interaction."""

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Process the completed environment interaction."""
        if self.training and self.memory.filled:
            started_at = time.perf_counter()
            self.update(timestep=timestep, timesteps=timesteps)
            self.track_data(
                "Stats / Algorithm update time (ms)",
                (time.perf_counter() - started_at) * 1_000,
            )
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def _loss(
        self,
        rollout_steps: int,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay a rollout and compute actor-critic losses with truncated BPTT."""
        observations = self.memory.get_tensor_by_name("observations")
        actions = self.memory.get_tensor_by_name("actions")
        dones = torch.logical_or(
            self.memory.get_tensor_by_name("terminated"),
            self.memory.get_tensor_by_name("truncated"),
        )
        policy_terms = []
        value_terms = []
        entropy_terms = []
        policy_state = self._rollout_policy_state
        value_state = self._rollout_value_state

        for step in range(rollout_steps):
            policy_features, policy_state = self.policy_network(observations[step], policy_state)
            predicted_values, value_state = self.value_network(observations[step], value_state)
            distribution = self.policy.distribution(policy_features)
            policy_terms.append(-(advantages[step] * distribution.log_prob(actions[step])).mean())
            value_terms.append(torch.nn.functional.mse_loss(predicted_values, returns[step]))
            if self.cfg.entropy_loss_scale:
                entropy_terms.append(-self.cfg.entropy_loss_scale * distribution.entropy().mean())

            policy_state = self.policy_network.reset_state(policy_state, dones[step])
            value_state = self.value_network.reset_state(value_state, dones[step])
            if (step + 1) % self.cfg.sequence_length == 0:
                policy_state = detach_state(policy_state)
                value_state = detach_state(value_state)

        policy_loss = torch.stack(policy_terms).mean()
        value_loss = torch.stack(value_terms).mean()
        entropy_loss = (
            torch.stack(entropy_terms).mean()
            if entropy_terms
            else torch.zeros((), device=self.device)
        )
        return policy_loss, value_loss, entropy_loss

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Update the policy and value networks from the collected rollout."""
        rollout_steps = self.memory.memory_size if self.memory.filled else self.memory.memory_index
        if not rollout_steps:
            return

        next_inputs = torch.flatten(
            self._observation_preprocessor(self._next_observation, train=False), start_dim=1
        )
        with torch.no_grad():
            last_values, _ = self.value_network(next_inputs, self._value_state)
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards")[:rollout_steps],
            terminated=self.memory.get_tensor_by_name("terminated")[:rollout_steps],
            truncated=self.memory.get_tensor_by_name("truncated")[:rollout_steps],
            values=self.memory.get_tensor_by_name("values")[:rollout_steps],
            last_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
        )
        policy_loss, value_loss, entropy_loss = self._loss(rollout_steps, returns, advantages)
        self.policy_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)
        (policy_loss + value_loss + entropy_loss).backward()

        if self.cfg.policy_grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(self._policy_parameters, self.cfg.policy_grad_norm_clip)
        if self.cfg.value_grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(self._value_parameters, self.cfg.value_grad_norm_clip)

        self.policy_optimizer.step()
        self.value_optimizer.step()
        if self.policy_scheduler is not None:
            self.policy_scheduler.step()
        if self.value_scheduler is not None:
            self.value_scheduler.step()

        self.track_data("Loss / Policy loss", policy_loss.item())
        self.track_data("Loss / Value loss", value_loss.item())
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", entropy_loss.item())
        self.track_data(
            "Learning / Policy learning rate", self.policy_optimizer.param_groups[0]["lr"]
        )
        self.track_data(
            "Learning / Value learning rate", self.value_optimizer.param_groups[0]["lr"]
        )

        self._reset_rollout()
