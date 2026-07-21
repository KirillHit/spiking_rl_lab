"""REINFORCE agent implementation."""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from gymnasium.spaces import Discrete
from gymnasium.spaces.utils import flatdim
from omegaconf import MISSING
from skrl import config
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
from spiking_rl_lab.networks.types import DenseTensorShape, TensorShape
from spiking_rl_lab.policies.base_policy import BasePolicy
from spiking_rl_lab.policies.builder import PolicyConfig, build_policy

if TYPE_CHECKING:
    from collections.abc import Callable

    from skrl.envs.wrappers.torch import Wrapper
    from skrl.memories.torch import Memory

    from spiking_rl_lab.networks.types import ListState


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

    random_timesteps: int = 0
    """Number of initial timesteps that use random actions instead of the policy."""

    grad_norm_clip: float = 0.5
    """Maximum gradient norm. Set to ``0`` to disable clipping."""

    entropy_loss_scale: float = 0.0
    """Entropy regularization coefficient added to the policy loss."""

    rewards_shaper: str | Callable[..., Any] | None = None
    """Optional reward-shaping callable or dotted import path."""

    normalize_returns: bool = True
    """Whether to normalize returns across the collected rollout."""

    mixed_precision: bool = False
    """Whether to enable automatic mixed precision during optimization."""

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
        require_minimum("random_timesteps", self.random_timesteps, minimum=0)
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


def _flatten_observations(observations: torch.Tensor) -> torch.Tensor:
    """Flatten observations to the dense network input shape."""
    return observations.reshape(observations.shape[0], -1)


def _detach_state[StateT](state: StateT) -> StateT:
    """Detach a nested network state at a truncated-BPTT boundary."""
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, list):
        return [_detach_state(item) for item in state]
    if isinstance(state, tuple):
        values = tuple(_detach_state(item) for item in state)
        return type(state)(*values) if hasattr(state, "_fields") else values
    if isinstance(state, dict):
        return {key: _detach_state(value) for key, value in state.items()}
    return state


def _reset_finished_state[StateT](state: StateT, dones: torch.Tensor) -> StateT:
    """Reset state rows for environments whose episode has ended."""
    if isinstance(state, torch.Tensor):
        mask = (~dones.reshape(-1)).to(device=state.device, dtype=state.dtype)
        return state * mask.reshape(-1, *([1] * (state.ndim - 1)))
    if isinstance(state, list):
        return [_reset_finished_state(item, dones) for item in state]
    if isinstance(state, tuple):
        values = tuple(_reset_finished_state(item, dones) for item in state)
        return type(state)(*values) if hasattr(state, "_fields") else values
    if isinstance(state, dict):
        return {key: _reset_finished_state(value, dones) for key, value in state.items()}
    return state


def _broadcast_modules(*modules: torch.nn.Module) -> None:
    """Broadcast module states in distributed training."""
    for module in modules:
        state = [module.state_dict()]
        torch.distributed.broadcast_object_list(state, 0)
        module.load_state_dict(state[0])


def _reduce_gradients(parameters: tuple[torch.nn.Parameter, ...]) -> None:
    """Average parameter gradients across distributed workers."""
    gradients = [
        parameter.grad.reshape(-1) for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        return

    flattened = torch.cat(gradients)
    torch.distributed.all_reduce(flattened, op=torch.distributed.ReduceOp.SUM)
    offset = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        size = parameter.numel()
        parameter.grad.copy_(
            flattened[offset : offset + size].reshape_as(parameter.grad) / config.torch.world_size
        )
        offset += size


@register_agent("reinforce")
class Reinforce(BaseAgent):
    """Monte Carlo policy-gradient agent with truncated BPTT for stateful networks."""

    Config: ClassVar[type[ReinforceConfig]] = ReinforceConfig

    def build_memory(self, *, env: Wrapper) -> Memory:
        """Build storage for one policy rollout."""
        return RandomMemory(
            memory_size=self.cfg.rollouts,
            num_envs=env.num_envs,
            device=self.device,
        )

    def __init__(self, cfg: ReinforceConfig, *, env: Wrapper) -> None:
        """Build the policy network, policy adapter, and optimizer."""
        self.cfg: ReinforceConfig
        try:
            policy_network = NodeNetwork(
                cfg.policy_network,
                input_shape=TensorShape.dense(flatdim(env.observation_space)),
            ).to(cfg.device)
            require_shape_fields(
                "REINFORCE policy network output",
                policy_network.output_shape,
                shape_type=DenseTensorShape,
                fields={"features": flatdim(env.action_space)},
            )
            policy = build_policy(cfg.policy, action_space=env.action_space).to(cfg.device)
        except Exception as exc:
            msg = "Failed to create REINFORCE policy components"
            raise AgentCreationError(msg) from exc

        if not isinstance(policy, BasePolicy):
            msg = f"REINFORCE policy must inherit BasePolicy, got {type(policy).__name__}"
            raise AgentCreationError(msg)

        super().__init__(cfg, env=env)

        self.policy_network = policy_network
        self.policy = policy
        self.checkpoint_modules.update(policy_network=policy_network, policy=policy)
        self._parameters = tuple(
            parameter
            for module in (policy_network, policy)
            for parameter in module.parameters()
            if parameter.requires_grad
        )

        if config.torch.is_distributed:
            _broadcast_modules(policy_network, policy)

        self.optimizer = torch.optim.Adamax(self._parameters, lr=cfg.learning_rate)
        self.checkpoint_modules["optimizer"] = self.optimizer
        self.scheduler = None
        if cfg.learning_rate_scheduler is not None:
            self.scheduler = cfg.learning_rate_scheduler(
                self.optimizer,
                **cfg.learning_rate_scheduler_kwargs,
            )
            self.checkpoint_modules["scheduler"] = self.scheduler

        if cfg.observation_preprocessor is None:
            self._observation_preprocessor = self._empty_preprocessor
        else:
            kwargs = dict(cfg.observation_preprocessor_kwargs)
            kwargs.setdefault("size", self.observation_space)
            kwargs.setdefault("device", self.device)
            self._observation_preprocessor = cfg.observation_preprocessor(**kwargs)
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor

        self._device_type = torch.device(self.device).type
        self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=cfg.mixed_precision)
        self._hidden_states: ListState | None = None
        self._rollout_initial_state: ListState | None = None
        self._current_inputs: torch.Tensor | None = None

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize rollout storage and keep policy modules deterministic for replay."""
        super().init(trainer_cfg=trainer_cfg)
        self.policy_network.eval()
        self.policy.eval()

        self.memory.create_tensor(
            name="observations",
            size=flatdim(self.observation_space),
            dtype=torch.float32,
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
        self._current_inputs = None

    def act(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample an action and advance the policy-network state."""
        del states, timesteps
        self._current_inputs = None

        if self.training and timestep < self.cfg.random_timesteps:
            actions = torch.as_tensor(
                [self.action_space.sample() for _ in range(observations.shape[0])],
                device=self.device,
            )
            return (actions.unsqueeze(-1) if actions.ndim == 1 else actions), {}

        processed = self._observation_preprocessor(observations, train=self.training)
        inputs = _flatten_observations(processed)
        if self.training and self._rollout_initial_state is None:
            self._rollout_initial_state = self._hidden_states

        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self._device_type,
                enabled=self.cfg.mixed_precision,
            ),
        ):
            features, self._hidden_states = self.policy_network(inputs, self._hidden_states)
            distribution = self.policy.distribution(features)
            actions = distribution.sample() if self.training else distribution.mode()

        if self.training:
            self._current_inputs = inputs
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
        self._hidden_states = _reset_finished_state(self._hidden_states, dones)
        if not self.training or self._current_inputs is None:
            return

        if self.cfg.rewards_shaper is not None:
            rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)
        self.memory.add_samples(
            observations=self._current_inputs,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )
        self._current_inputs = None

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

        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            hidden_states = self._rollout_initial_state
            for step in range(rollout_steps):
                features, hidden_states = self.policy_network(observations[step], hidden_states)
                distribution = self.policy.distribution(features)
                log_prob = distribution.log_prob(actions[step])
                policy_terms.append(-(returns[step] * log_prob).mean())
                if self.cfg.entropy_loss_scale:
                    entropy_terms.append(
                        -self.cfg.entropy_loss_scale * distribution.entropy().mean()
                    )
                hidden_states = _reset_finished_state(hidden_states, dones[step])
                if (step + 1) % self.cfg.sequence_length == 0:
                    hidden_states = _detach_state(hidden_states)

            policy_loss = torch.stack(policy_terms).mean()
            entropy_loss = (
                torch.stack(entropy_terms).mean()
                if entropy_terms
                else torch.zeros((), device=self.device)
            )
        return policy_loss, entropy_loss

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Run one Monte Carlo policy-gradient update."""
        del timestep, timesteps
        rollout_steps = self.memory.memory_size if self.memory.filled else self.memory.memory_index
        if not rollout_steps:
            return

        policy_loss, entropy_loss = self._loss(
            rollout_steps,
            self._discounted_returns(rollout_steps),
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(policy_loss + entropy_loss).backward()

        if config.torch.is_distributed:
            _reduce_gradients(self._parameters)
        if self.cfg.grad_norm_clip:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self._parameters, self.cfg.grad_norm_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

        self.track_data("Loss / Policy loss", policy_loss.item())
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", entropy_loss.item())
        if self.scheduler is not None:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
        self._reset_rollout()
