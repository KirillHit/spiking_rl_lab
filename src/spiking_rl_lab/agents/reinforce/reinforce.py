"""REINFORCE agent implementation."""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from packaging import version
from skrl import config
from skrl.memories.torch import RandomMemory

from spiking_rl_lab.agents.base_agent import BaseAgent, BaseAgentCfg
from spiking_rl_lab.agents.builder import register_agent
from spiking_rl_lab.utils.exception import AgentCreationError

if TYPE_CHECKING:
    import gymnasium
    from skrl.envs.wrappers.torch import Wrapper
    from skrl.memories.torch import Memory
    from skrl.models.torch import Model

    from spiking_rl_lab.utils.config import AgentConfig


@dataclasses.dataclass(kw_only=True)
class ReinforceCfg(BaseAgentCfg):
    """Configuration for the REINFORCE agent."""

    rollouts: int = 16
    mini_batches: int = 1
    discount_factor: float = 0.99
    learning_rate: float = 1e-3
    learning_rate_scheduler: type | None = None
    learning_rate_scheduler_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    observation_preprocessor: type | None = None
    observation_preprocessor_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    state_preprocessor: type | None = None
    state_preprocessor_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    random_timesteps: int = 0
    learning_starts: int = 0
    grad_norm_clip: float = 0.5
    entropy_loss_scale: float = 0.0
    rewards_shaper: Any | None = None
    normalize_returns: bool = True
    mixed_precision: bool = False


@register_agent("reinforce")
class Reinforce(BaseAgent):
    """REINFORCE agent implementation."""

    cfg_cls: ClassVar[type[ReinforceCfg]] = ReinforceCfg

    @classmethod
    def build_memory(cls, *, cfg: AgentConfig, env: Wrapper) -> Memory | None:
        """Build rollout memory sized for at least one REINFORCE update window."""
        rollouts = int(cfg.params.get("rollouts", 16))
        memory_size = max(cfg.memory_size, rollouts)
        return RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=cfg.device)

    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
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

        # models
        self.policy = self.models.get("policy", None)
        if self.policy is None:
            msg = "The REINFORCE agent requires a 'policy' model."
            raise AgentCreationError(msg)
        if self.memory is None:
            msg = "The REINFORCE agent requires a rollout memory."
            raise AgentCreationError(msg)
        if not callable(getattr(self.policy, "get_entropy", None)):
            msg = "The REINFORCE agent requires a stochastic 'policy' model."
            raise AgentCreationError(msg)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy

        if config.torch.is_distributed:
            self.policy.broadcast_parameters()

        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(
                device=self._device_type,
                enabled=self.cfg.mixed_precision,
            )
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate)
        self.checkpoint_modules["optimizer"] = self.optimizer

        self.scheduler = self.cfg.learning_rate_scheduler
        if self.scheduler is not None:
            self.scheduler = self.scheduler(
                self.optimizer,
                **self.cfg.learning_rate_scheduler_kwargs,
            )

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

        self._current_log_prob = outputs.get("log_prob")
        if self.training and self._current_log_prob is None:
            msg = "The REINFORCE policy must return 'log_prob' during action sampling."
            raise AgentCreationError(msg)

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
            if not self._rollout % self.cfg.rollouts and timestep >= self.cfg.learning_starts:
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
        indexes = torch.arange(num_samples, device=self.device)
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
