"""PPO agent implementation."""

from __future__ import annotations

import dataclasses
import logging
import time
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from gymnasium.spaces import Discrete
from gymnasium.spaces.utils import flatdim
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveLR

from spiking_rl_lab.agents.a2c.a2c import _as_sequences, compute_gae
from spiking_rl_lab.agents.base_agent import BaseAgent
from spiking_rl_lab.agents.builder import register_agent
from spiking_rl_lab.agents.ppo.ppo_cfg import PPOConfig
from spiking_rl_lab.core.exception import AgentCreationError
from spiking_rl_lab.core.validation import require_shape_fields
from spiking_rl_lab.envs.observations import bootstrap_observations
from spiking_rl_lab.networks.node_network import NodeNetwork
from spiking_rl_lab.networks.shape import DenseTensorShape, TensorShape
from spiking_rl_lab.networks.state import (
    ListState,
    concatenate_states,
    detach_state,
    select_state,
)
from spiking_rl_lab.networks.statistics.activity import (
    collect_forward_outputs,
    mean_spike_activity,
    spike_activity_moments,
)
from spiking_rl_lab.networks.statistics.normalization import (
    apply_collected_batch_norm_statistics,
    collect_batch_norm_statistics,
    has_batch_norm,
)
from spiking_rl_lab.policies.builder import build_policy

if TYPE_CHECKING:
    from collections.abc import Generator

    from skrl.envs.wrappers.torch import Wrapper
    from skrl.memories.torch import Memory

log = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True)
class _Rollout:
    """Chronological tensors read from rollout memory once per update."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    values: torch.Tensor
    log_prob: torch.Tensor
    dones: torch.Tensor


@dataclasses.dataclass(slots=True)
class _BurnIn:
    """Transitions preceding the current rollout used to reconstruct network state."""

    observations: torch.Tensor
    dones: torch.Tensor


@dataclasses.dataclass(slots=True)
class _ReplayResult:
    """States and KL produced by a chronological rollout replay."""

    approximate_kl: float
    policy_state: ListState
    value_state: ListState
    sequence_policy_state: ListState
    sequence_value_state: ListState


@dataclasses.dataclass(slots=True)
class _SequenceBatch:
    """Rollout tensors and boundary states arranged as recurrent sequences."""

    observations: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    dones: torch.Tensor
    policy_state: ListState
    value_state: ListState
    old_log_prob: torch.Tensor
    old_values: torch.Tensor

    @property
    def size(self) -> int:
        """Return the number of independent sequences."""
        return self.observations.shape[1]


@register_agent("ppo")
class PPO(BaseAgent):
    """Proximal policy optimization with recurrent rollout replay."""

    Config: ClassVar[type[PPOConfig]] = PPOConfig

    def __init__(self, cfg: PPOConfig, *, env: Wrapper) -> None:
        """Build networks, policy adapter, optimizer, and optional utilities."""
        sequence_count = env.num_envs * (cfg.rollouts // cfg.sequence_length)
        if cfg.mini_batches > sequence_count:
            msg = "PPO mini_batches must not exceed the number of rollout sequences"
            raise AgentCreationError(msg)

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
                "PPO policy network output",
                self.policy_network.output_shape,
                shape_type=DenseTensorShape,
                fields={"features": self.policy.required_output_features},
            )
            require_shape_fields(
                "PPO value network output",
                self.value_network.output_shape,
                shape_type=DenseTensorShape,
                fields={"features": 1},
            )
        except Exception as exc:
            msg = "Failed to create PPO components"
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
        self._burn_in: _BurnIn | None = None
        self._sequence_policy_states: list[ListState] = []
        self._sequence_value_states: list[ListState] = []
        self._processed_observation: torch.Tensor | None = None
        self._next_observation: torch.Tensor | None = None
        self._current_value: torch.Tensor | None = None
        self._current_log_prob: torch.Tensor | None = None

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
        self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
        self._policy_state = None
        self._value_state = None
        self._burn_in = None
        self._reset_rollout()

    def _reset_rollout(self) -> None:
        """Clear rollout storage without resetting live network states."""
        self.memory.reset()
        self._sequence_policy_states.clear()
        self._sequence_value_states.clear()
        self._processed_observation = None
        self._next_observation = None
        self._current_value = None
        self._current_log_prob = None

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
            if self.training and self.memory.memory_index % self.cfg.sequence_length == 0:
                self._sequence_policy_states.append(detach_state(self._policy_state))
                self._sequence_value_states.append(detach_state(self._value_state))

            policy_features, self._policy_state = self.policy_network(inputs, self._policy_state)
            values, self._value_state = self.value_network(inputs, self._value_state)
            distribution = self.policy.distribution(policy_features)
            actions = distribution.sample() if self.training else distribution.mode()
            if self.training:
                self._current_value = values
                self._current_log_prob = distribution.log_prob(actions)

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
            bootstrap = truncated & ~terminated
            if self.cfg.time_limit_bootstrap and bootstrap.any():
                terminal_observations = bootstrap_observations(
                    next_observations, infos, observation_space=self.observation_space
                )
                next_inputs = torch.flatten(
                    self._observation_preprocessor(terminal_observations, train=False), start_dim=1
                )
                with torch.no_grad():
                    next_values, _ = self.value_network(next_inputs, self._value_state)
                rewards = rewards + self.cfg.discount_factor * next_values * bootstrap

            self.memory.add_samples(
                observations=self._processed_observation,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                values=self._current_value,
                log_prob=self._current_log_prob,
            )
            self._next_observation = next_observations

        self.reset_state(dones)
        self._processed_observation = None
        self._current_value = None
        self._current_log_prob = None

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Prepare the agent before an environment interaction."""

    def reset_state(self, dones: torch.Tensor) -> None:
        """Reset recurrent state for completed environments."""
        self._policy_state = self.policy_network.reset_state(self._policy_state, dones)
        self._value_state = self.value_network.reset_state(self._value_state, dones)

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

    def _read_rollout(self) -> _Rollout:
        """Read the current memory contents in chronological form."""
        terminated = self.memory.get_tensor_by_name("terminated")
        truncated = self.memory.get_tensor_by_name("truncated")
        return _Rollout(
            observations=self.memory.get_tensor_by_name("observations"),
            actions=self.memory.get_tensor_by_name("actions"),
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=terminated,
            truncated=truncated,
            values=self.memory.get_tensor_by_name("values"),
            log_prob=self.memory.get_tensor_by_name("log_prob"),
            dones=torch.logical_or(terminated, truncated),
        )

    def _build_sequence_batch(
        self,
        rollout: _Rollout,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> _SequenceBatch:
        """Arrange a rollout and its saved boundary states as recurrent sequences."""
        sequence_length = self.cfg.sequence_length
        return _SequenceBatch(
            observations=_as_sequences(rollout.observations, sequence_length),
            actions=_as_sequences(rollout.actions, sequence_length),
            old_log_prob=_as_sequences(rollout.log_prob, sequence_length),
            old_values=_as_sequences(rollout.values, sequence_length),
            returns=_as_sequences(returns, sequence_length),
            advantages=_as_sequences(advantages, sequence_length),
            dones=_as_sequences(rollout.dones, sequence_length),
            policy_state=concatenate_states(self._sequence_policy_states),
            value_state=concatenate_states(self._sequence_value_states),
        )

    def _loss(
        self,
        batch: _SequenceBatch,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay one group of recurrent sequences and compute PPO losses."""
        observations = batch.observations[:, indices]
        dones = batch.dones[:, indices]
        policy_state = select_state(batch.policy_state, indices)
        value_state = select_state(batch.value_state, indices)
        policy_outputs = []
        predicted_values = []

        with collect_forward_outputs(
            self.policy_network,
            partial(mean_spike_activity, dim=0),
            detach=not self.cfg.spike_activity_loss_scale,
        ) as activity_terms:
            for step in range(self.cfg.sequence_length):
                policy_output, policy_state = self.policy_network(observations[step], policy_state)
                values, value_state = self.value_network(observations[step], value_state)
                policy_outputs.append(policy_output)
                predicted_values.append(values)
                policy_state = self.policy_network.reset_state(policy_state, dones[step])
                value_state = self.value_network.reset_state(value_state, dones[step])

        policy_outputs = torch.stack(policy_outputs)
        predicted_values = torch.stack(predicted_values)
        actions = batch.actions[:, indices]
        old_log_prob = batch.old_log_prob[:, indices]
        old_values = batch.old_values[:, indices]
        returns = batch.returns[:, indices]
        advantages = batch.advantages[:, indices]

        distribution = self.policy.distribution(policy_outputs)
        ratio = (distribution.log_prob(actions) - old_log_prob).exp()
        surrogate = advantages * ratio
        clipped = advantages * ratio.clamp(1 - self.cfg.ratio_clip, 1 + self.cfg.ratio_clip)
        policy_loss = -torch.minimum(surrogate, clipped).mean()

        if self.cfg.value_clip:
            predicted_values = old_values + (predicted_values - old_values).clamp(
                -self.cfg.value_clip, self.cfg.value_clip
            )
        value_loss = self.cfg.value_loss_scale * torch.nn.functional.mse_loss(
            predicted_values, returns
        )
        entropy_loss = (
            -self.cfg.entropy_loss_scale * distribution.entropy().mean()
            if self.cfg.entropy_loss_scale
            else torch.zeros((), device=self.device)
        )

        neuron_activity = {
            name: torch.stack(layer_terms).mean(dim=0)
            for name, layer_terms in activity_terms.items()
        }
        layer_activity = {name: rates.mean() for name, rates in neuron_activity.items()}
        if neuron_activity:
            mean_activity, activity_penalty = spike_activity_moments(neuron_activity)
        else:
            mean_activity = activity_penalty = torch.zeros((), device=self.device)
        activity_loss = self.cfg.spike_activity_loss_scale * activity_penalty

        for name, activity in layer_activity.items():
            self.track_data(f"Activity / {name}", activity.item())
        if layer_activity:
            self.track_data("Activity / Mean", mean_activity.item())

        return policy_loss, value_loss, entropy_loss, activity_loss

    @torch.no_grad()
    def _burn_in_states(self, observations: torch.Tensor) -> tuple[ListState, ListState]:
        """Reconstruct states at the beginning of a rollout."""
        initial_inputs = self._burn_in.observations[0] if self._burn_in else observations[0]
        policy_state = self.policy_network.initial_state(initial_inputs)
        value_state = self.value_network.initial_state(initial_inputs)

        if self._burn_in:
            for step_observations, step_dones in zip(
                self._burn_in.observations, self._burn_in.dones, strict=True
            ):
                _, policy_state = self.policy_network(step_observations, policy_state)
                _, value_state = self.value_network(step_observations, value_state)
                policy_state = self.policy_network.reset_state(policy_state, step_dones)
                value_state = self.value_network.reset_state(value_state, step_dones)

        return policy_state, value_state

    @torch.no_grad()
    def _replay_rollout(self, rollout: _Rollout) -> _ReplayResult:
        """Replay a rollout chronologically with the current network parameters."""
        policy_state, value_state = self._burn_in_states(rollout.observations)

        sequence_policy_states = []
        sequence_value_states = []
        log_ratios = []
        for step, (step_observations, step_actions, step_old_log_prob, step_dones) in enumerate(
            zip(
                rollout.observations,
                rollout.actions,
                rollout.log_prob,
                rollout.dones,
                strict=True,
            )
        ):
            if step % self.cfg.sequence_length == 0:
                sequence_policy_states.append(detach_state(policy_state))
                sequence_value_states.append(detach_state(value_state))

            policy_outputs, policy_state = self.policy_network(step_observations, policy_state)
            _, value_state = self.value_network(step_observations, value_state)
            distribution = self.policy.distribution(policy_outputs)
            log_ratios.append(distribution.log_prob(step_actions) - step_old_log_prob)
            policy_state = self.policy_network.reset_state(policy_state, step_dones)
            value_state = self.value_network.reset_state(value_state, step_dones)

        log_ratio = torch.stack(log_ratios)
        return _ReplayResult(
            approximate_kl=(torch.expm1(log_ratio) - log_ratio).mean().item(),
            policy_state=detach_state(policy_state),
            value_state=detach_state(value_state),
            sequence_policy_state=concatenate_states(sequence_policy_states),
            sequence_value_state=concatenate_states(sequence_value_states),
        )

    def _remember_burn_in(self, rollout: _Rollout) -> None:
        """Retain the end of this rollout for the next rollout's burn-in."""
        if not self.cfg.state_burn_in:
            self._burn_in = None
            return

        self._burn_in = _BurnIn(
            observations=rollout.observations[-self.cfg.state_burn_in :].detach().clone(),
            dones=rollout.dones[-self.cfg.state_burn_in :].detach().clone(),
        )

    @torch.no_grad()
    def _update_batch_norm_statistics(self, batch: _SequenceBatch) -> None:
        """Update normalization statistics from one final-network rollout replay."""
        if not has_batch_norm(self.policy_network, self.value_network):
            return

        policy_state = batch.policy_state
        value_state = batch.value_state
        with collect_batch_norm_statistics(self.policy_network, self.value_network):
            for step in range(self.cfg.sequence_length):
                _, policy_state = self.policy_network(batch.observations[step], policy_state)
                _, value_state = self.value_network(batch.observations[step], value_state)
                policy_state = self.policy_network.reset_state(policy_state, batch.dones[step])
                value_state = self.value_network.reset_state(value_state, batch.dones[step])

        apply_collected_batch_norm_statistics(self.policy_network, self.value_network)

    def _optimize_policy(self, loss: torch.Tensor) -> None:
        """Apply one clipped actor gradient step."""
        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.policy_grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(self._policy_parameters, self.cfg.policy_grad_norm_clip)
        self.policy_optimizer.step()

    def _optimize_value(self, loss: torch.Tensor) -> None:
        """Apply one clipped critic gradient step."""
        self.value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.value_grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(self._value_parameters, self.cfg.value_grad_norm_clip)
        self.value_optimizer.step()

    def _mini_batch_groups(self, batch_size: int) -> Generator[torch.Tensor]:
        """Yield shuffled mini-batch indices for every learning epoch."""
        for _ in range(self.cfg.learning_epochs):
            yield from torch.randperm(batch_size, device=self.device).tensor_split(
                self.cfg.mini_batches
            )

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Update the policy and value networks from the collected rollout."""
        if not self.memory.filled:
            return

        rollout = self._read_rollout()
        next_inputs = torch.flatten(
            self._observation_preprocessor(self._next_observation, train=False), start_dim=1
        )
        with torch.no_grad():
            last_values, _ = self.value_network(next_inputs, self._value_state)
        returns, advantages = compute_gae(
            rewards=rollout.rewards,
            terminated=rollout.terminated,
            truncated=rollout.truncated,
            values=rollout.values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
        )
        batch = self._build_sequence_batch(rollout, returns, advantages)

        losses = []
        for indices in self._mini_batch_groups(batch.size):
            policy_loss, value_loss, entropy_loss, activity_loss = self._loss(batch, indices)
            self._optimize_policy(policy_loss + entropy_loss + activity_loss)
            self._optimize_value(value_loss)
            losses.append(
                torch.stack(
                    [
                        policy_loss.detach(),
                        value_loss.detach(),
                        entropy_loss.detach(),
                        activity_loss.detach(),
                    ]
                )
            )
            replay = self._replay_rollout(rollout)
            batch.policy_state = replay.sequence_policy_state
            batch.value_state = replay.sequence_value_state
            if self.cfg.kl_threshold and replay.approximate_kl > self.cfg.kl_threshold:
                break

        self._step_schedulers(replay.approximate_kl)

        self.track_data("Learning / KL divergence", replay.approximate_kl)

        if has_batch_norm(self.policy_network, self.value_network):
            self._update_batch_norm_statistics(batch)
            replay = self._replay_rollout(rollout)

        self._policy_state = replay.policy_state
        self._value_state = replay.value_state

        policy_loss, value_loss, entropy_loss, activity_loss = torch.stack(losses).mean(0)

        self.track_data("Learning / Policy updates", len(losses))
        self.track_data("Loss / Policy loss", policy_loss.item())
        self.track_data("Loss / Value loss", value_loss.item())
        if self.cfg.spike_activity_loss_scale:
            self.track_data("Loss / Spike activity loss", activity_loss.item())
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", entropy_loss.item())
        self.track_data(
            "Learning / Policy learning rate", self.policy_optimizer.param_groups[0]["lr"]
        )
        self.track_data(
            "Learning / Value learning rate", self.value_optimizer.param_groups[0]["lr"]
        )

        self._remember_burn_in(rollout)
        self._reset_rollout()

    def _step_schedulers(self, approximate_kl: float) -> None:
        """Advance learning-rate schedules once per rollout."""
        for scheduler in (self.policy_scheduler, self.value_scheduler):
            if isinstance(scheduler, KLAdaptiveLR):
                scheduler.step(approximate_kl)
            elif scheduler is not None:
                scheduler.step()
