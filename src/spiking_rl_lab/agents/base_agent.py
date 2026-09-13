"""Shared base classes for agents."""

from __future__ import annotations

import dataclasses
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import mlflow
import numpy as np
from skrl.agents.torch import Agent
from skrl.agents.torch import AgentCfg as SkrlAgentConfig

from spiking_rl_lab.core.factory import ConfiguredBase

if TYPE_CHECKING:
    from collections.abc import Collection

    import torch
    from skrl.envs.wrappers.torch import Wrapper
    from skrl.memories.torch import Memory


class BaseAgent(Agent, ConfiguredBase, ABC):
    """Common utilities for agents used in this project."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(SkrlAgentConfig):
        """Base class for the agent's configuration."""

        device: str = "cpu"

    def __init__(
        self,
        cfg: Config,
        *,
        env: Wrapper,
    ) -> None:
        """Initialize common tracking state."""
        ConfiguredBase.__init__(self, cfg)
        Agent.__init__(
            self,
            cfg=cfg,
            models={},
            memory=None,
            observation_space=env.observation_space,
            state_space=env.state_space,
            action_space=env.action_space,
            device=cfg.device,
        )
        self.memory = self.build_memory(env=env)
        self.last_tracking_metrics: dict[str, float] = {}
        self._tracking_ready: torch.Tensor | None = None

    @abstractmethod
    def build_memory(self, *, env: Wrapper) -> Memory | None:
        """Build agent memory."""

    @abstractmethod
    def reset_state(self, dones: torch.Tensor) -> None:
        """Reset recurrent state for completed environments."""

    def reset_episode_tracking(self, ready: torch.Tensor) -> None:
        """Discard partial returns and track only episodes starting after validation."""
        self._tracking_ready = ready.reshape(-1, 1).clone()
        if self._cumulative_rewards is not None:
            self._cumulative_rewards.zero_()
            self._cumulative_timesteps.zero_()

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
        """Track complete training episodes, excluding tails left by validation."""
        ready = self._tracking_ready
        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated if ready is None else terminated & ready,
            truncated=truncated if ready is None else truncated & ready,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )
        if ready is not None:
            if self._cumulative_rewards is not None:
                self._cumulative_rewards.masked_fill_(~ready, 0)
                self._cumulative_timesteps.masked_fill_(~ready, 0)
            ready.logical_or_(terminated | truncated)

    def post_interaction(
        self,
        *,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Write periodic checkpoints and tracking data."""
        step = timestep + 1
        if self.training and self.checkpoint_interval > 0 and step % self.checkpoint_interval == 0:
            self.write_checkpoint(timestep=step, timesteps=timesteps)
        if self.write_interval > 0 and step % self.write_interval == 0:
            self.write_tracking_data(timestep=step, timesteps=timesteps)

    @property
    def rollout_complete(self) -> bool:
        """Return whether the current rollout is ready for an update."""
        return self.memory is None or self.memory.filled

    @property
    def training_rewards(self) -> Collection[float]:
        """Return rewards collected for training progress tracking."""
        return self._track_rewards

    def write_tracking_data(self, timestep: int, timesteps: int) -> None:
        """Flush tracked metrics to MLflow and reset local buffers."""
        metrics: dict[str, float] = {}
        for key, value in self.tracking_data.items():
            metrics[self._mlflow_key(key)] = self._reduce_tracking_value(key, value)
        self.last_tracking_metrics = metrics

        if metrics and mlflow.active_run() is not None:
            mlflow.log_metrics(
                metrics,
                step=timestep,
                synchronous=False,
            )

        self._track_rewards.clear()
        self._track_timesteps.clear()
        self.tracking_data.clear()

    def _mlflow_key(self, key: str) -> str:
        key = key.replace(" (min)", "_min").replace(" (max)", "_max").replace(" (mean)", "_mean")
        key = re.sub(r"[^0-9A-Za-z_\-\. :/ ]+", "_", key)
        prefix = "Train" if self.training else "Eval"
        return f"{prefix} / {key}"

    @staticmethod
    def _reduce_tracking_value(key: str, value: list[float]) -> float:
        if key.endswith("(min)"):
            return float(np.min(value))
        if key.endswith("(max)"):
            return float(np.max(value))
        return float(np.mean(value))
