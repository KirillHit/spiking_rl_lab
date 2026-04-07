"""REINFORCE agent implementation."""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from skrl import config
from skrl.memories.torch import RandomMemory

from spiking_rl_lab.agents.base_agent import BaseAgent, BaseAgentCfg
from spiking_rl_lab.agents.builder import register_agent
from spiking_rl_lab.utils.validation import (
    resolve_optional_callable,
    resolve_optional_class,
    validate_min,
    validate_positive,
    validate_range,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import gymnasium
    from skrl.envs.wrappers.torch import Wrapper
    from skrl.memories.torch import Memory
    from skrl.models.torch import Model


@dataclasses.dataclass(kw_only=True)
class ReinforceCfg(BaseAgentCfg):
    """Configuration for the REINFORCE agent."""

    rollouts: int = 16
    """Number of environment steps collected before each policy update."""

    mini_batches: int = 1
    """Number of mini-batches used to split one rollout during optimization."""

    discount_factor: float = 0.99
    """Reward discount factor used to compute Monte Carlo returns."""

    learning_rate: float = 1e-3
    """Adam optimizer learning rate."""

    learning_rate_scheduler: str | type[Any] | None = None
    """Optional learning rate scheduler class or dotted import path."""

    learning_rate_scheduler_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Keyword arguments passed to ``learning_rate_scheduler`` during construction."""

    observation_preprocessor: str | type[Any] | None = None
    """Optional observation preprocessor class or dotted import path."""

    observation_preprocessor_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Keyword arguments passed to ``observation_preprocessor`` during construction."""

    state_preprocessor: str | type[Any] | None = None
    """Optional state preprocessor class or dotted import path."""

    state_preprocessor_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Keyword arguments passed to ``state_preprocessor`` during construction."""

    random_timesteps: int = 0
    """Number of initial timesteps that use random actions instead of the policy."""

    grad_norm_clip: float = 0.5
    """Maximum gradient norm. Set to ``0`` to disable clipping."""

    entropy_loss_scale: float = 0.0
    """Entropy regularization coefficient added to the policy loss."""

    rewards_shaper: str | Callable[..., Any] | None = None
    """Optional reward-shaping callable or dotted import path."""

    normalize_returns: bool = True
    """Whether to normalize returns across the collected rollout before optimization."""

    mixed_precision: bool = False
    """Whether to enable automatic mixed precision during forward and backward passes."""

    def __post_init__(self) -> None:
        """Validate REINFORCE hyperparameters after dataclass initialization."""
        validate_min("rollouts", self.rollouts, minimum=1)
        validate_min("mini_batches", self.mini_batches, minimum=1)
        validate_range("discount_factor", self.discount_factor, minimum=0.0, maximum=1.0)
        validate_positive("learning_rate", self.learning_rate)
        validate_min("random_timesteps", self.random_timesteps, minimum=0)
        validate_min("grad_norm_clip", self.grad_norm_clip, minimum=0.0)
        validate_min("entropy_loss_scale", self.entropy_loss_scale, minimum=0.0)
        self.learning_rate_scheduler = resolve_optional_class(
            "learning_rate_scheduler",
            self.learning_rate_scheduler,
        )
        self.observation_preprocessor = resolve_optional_class(
            "observation_preprocessor",
            self.observation_preprocessor,
        )
        self.state_preprocessor = resolve_optional_class(
            "state_preprocessor",
            self.state_preprocessor,
        )
        self.rewards_shaper = resolve_optional_callable(
            "rewards_shaper",
            self.rewards_shaper,
        )


@register_agent("reinforce")
class Reinforce(BaseAgent):
    """REINFORCE agent implementation."""

    cfg_cls: ClassVar[type[ReinforceCfg]] = ReinforceCfg

    def build_memory(self, *, env: Wrapper) -> Memory | None:
        """Build rollout memory sized for at least one REINFORCE update window."""
        rollout_memory_size = self.cfg.rollouts
        return RandomMemory(
            memory_size=rollout_memory_size,
            num_envs=env.num_envs,
            device=self.device,
        )

    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None,
        observation_space: gymnasium.Space | None,
        state_space: gymnasium.Space | None,
        action_space: gymnasium.Space | None,
        device: str | torch.device | None,
        cfg: ReinforceCfg,
    ) -> None:
        """REINFORCE agent implementation."""
        self.cfg: ReinforceCfg
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )

        self.policy = self.models["policy"]
        self.checkpoint_modules["policy"] = self.policy

        if config.torch.is_distributed:
            self.policy.broadcast_parameters()

        self._device_type = torch.device(self.device).type
        self.scaler = torch.amp.GradScaler(
            device=self._device_type,
            enabled=self.cfg.mixed_precision,
        )

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate)
        self.checkpoint_modules["optimizer"] = self.optimizer

        self.scheduler = None
        scheduler_cls = self.cfg.learning_rate_scheduler
        if scheduler_cls is not None:
            self.scheduler = scheduler_cls(
                self.optimizer,
                **self.cfg.learning_rate_scheduler_kwargs,
            )
            self.checkpoint_modules["scheduler"] = self.scheduler

        if self.cfg.observation_preprocessor:
            self._observation_preprocessor = self.cfg.observation_preprocessor(
                **self.cfg.observation_preprocessor_kwargs,
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor

        if self.cfg.state_preprocessor:
            self._state_preprocessor = self.cfg.state_preprocessor(
                **self.cfg.state_preprocessor_kwargs,
            )
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        self._current_log_prob: torch.Tensor | None = None
        self._rollout = 0

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize the agent."""
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(enabled=False)

        self.memory.create_tensor(
            name="observations",
            size=self.observation_space,
            dtype=torch.float32,
        )
        self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
        self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
        self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="dones", size=1, dtype=torch.bool)
        self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)

        self._tensors_names = ["observations", "states", "actions", "returns", "log_prob"]
        self.memory.reset()

    def act(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample actions from the policy."""
        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }

        if timestep < self.cfg.random_timesteps:
            self._current_log_prob = None
            return self.policy.random_act(inputs, role="policy")

        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")

        self._current_log_prob = outputs["log_prob"]

        return actions, outputs

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
        """Record environment transitions in rollout memory."""
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

        if not self.training or self._current_log_prob is None:
            return

        if self.cfg.rewards_shaper is not None:
            rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

        dones = torch.logical_or(terminated, truncated)
        self.memory.add_samples(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            log_prob=self._current_log_prob,
        )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Run the hook before the environment interaction."""

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Trigger policy updates after rollout collection."""
        if self.training:
            self._rollout += 1
            if not self._rollout % self.cfg.rollouts:
                started_at = time.perf_counter()
                self.enable_models_training_mode(enabled=True)
                self.update(timestep=timestep, timesteps=timesteps)
                self.enable_models_training_mode(enabled=False)
                elapsed_time_ms = (time.perf_counter() - started_at) * 1_000
                self.track_data("Stats / Algorithm update time (ms)", elapsed_time_ms)

        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def _compute_discounted_returns(self, rollout_steps: int) -> torch.Tensor:
        """Compute discounted returns for the current rollout."""
        rewards = self.memory.get_tensor_by_name("rewards")[:rollout_steps]
        dones = self.memory.get_tensor_by_name("dones")[:rollout_steps]
        returns = torch.zeros_like(rewards)

        discounted_return = torch.zeros((self.memory.num_envs, 1), device=self.device)
        for step in range(rollout_steps - 1, -1, -1):
            discounted_return = rewards[step] + (
                self.cfg.discount_factor * discounted_return * (~dones[step]).float()
            )
            returns[step] = discounted_return

        if not self.cfg.normalize_returns:
            return returns

        flat_returns = returns.view(-1, 1)
        flat_returns = (flat_returns - flat_returns.mean()) / flat_returns.std().clamp_min(1e-8)
        return flat_returns.view_as(returns)

    def _sample_rollout_batches(self, rollout_steps: int) -> list[list[torch.Tensor]]:
        """Sample mini-batches from the current rollout."""
        num_samples = rollout_steps * self.memory.num_envs
        batch_count = max(1, min(self.cfg.mini_batches, num_samples))
        indexes = torch.randperm(num_samples, device=self.device)
        return self.memory.sample_by_index(
            names=self._tensors_names,
            indexes=indexes,
            mini_batches=batch_count,
        )

    def _update_policy_batch(
        self,
        sampled_observations: torch.Tensor,
        sampled_states: torch.Tensor | None,
        sampled_actions: torch.Tensor,
        sampled_returns: torch.Tensor,
    ) -> tuple[float, float]:
        """Update the policy on a single mini-batch."""
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor(
                    sampled_observations,
                    train=True,
                ),
                "states": self._state_preprocessor(sampled_states, train=True),
                "taken_actions": sampled_actions,
            }
            _, outputs = self.policy.act(inputs, role="policy")
            log_prob = outputs["log_prob"]

            policy_loss = -(sampled_returns * log_prob).mean()
            if self.cfg.entropy_loss_scale:
                entropy_loss = (
                    -self.cfg.entropy_loss_scale
                    * self.policy.get_entropy(
                        role="policy",
                    ).mean()
                )
            else:
                entropy_loss = torch.zeros((), device=self.device)

        self.optimizer.zero_grad()
        self.scaler.scale(policy_loss + entropy_loss).backward()

        if config.torch.is_distributed:
            self.policy.reduce_parameters()

        if self.cfg.grad_norm_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.cfg.grad_norm_clip,
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return policy_loss.item(), entropy_loss.item()

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Run one REINFORCE policy update from the current rollout."""
        del timestep, timesteps

        rollout_steps = self.memory.memory_size if self.memory.filled else self.memory.memory_index
        if rollout_steps <= 0:
            return

        returns = self._compute_discounted_returns(rollout_steps)
        self.memory.get_tensor_by_name("returns")[:rollout_steps].copy_(returns)
        sampled_batches = self._sample_rollout_batches(rollout_steps)

        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0

        for (
            sampled_observations,
            sampled_states,
            sampled_actions,
            sampled_returns,
            _sampled_log_prob,
        ) in sampled_batches:
            del _sampled_log_prob
            policy_loss, entropy_loss = self._update_policy_batch(
                sampled_observations=sampled_observations,
                sampled_states=sampled_states,
                sampled_actions=sampled_actions,
                sampled_returns=sampled_returns,
            )
            cumulative_policy_loss += policy_loss
            cumulative_entropy_loss += entropy_loss

        if self.scheduler is not None:
            self.scheduler.step()

        self.track_data("Loss / Policy loss", cumulative_policy_loss / len(sampled_batches))
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / len(sampled_batches))
        if self.scheduler is not None:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

        distribution = self.policy.distribution(role="policy")
        if hasattr(distribution, "stddev"):
            self.track_data("Policy / Standard deviation", distribution.stddev.mean().item())

        self.memory.reset()
