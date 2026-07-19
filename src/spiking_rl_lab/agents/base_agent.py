"""Shared base classes for agents."""

from __future__ import annotations

import dataclasses
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import mlflow
import numpy as np
from skrl.agents.torch import Agent, AgentCfg

from spiking_rl_lab.core.factory import ConfiguredBase

if TYPE_CHECKING:
    from skrl.envs.wrappers.torch import Wrapper
    from skrl.memories.torch import Memory
    from skrl.models.torch import Model


class BaseAgent(Agent, ConfiguredBase, ABC):
    """Common utilities for agents used in this project."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(AgentCfg):
        """Base class for the agent's configuration."""

        device: str = "cpu"

    def __init__(
        self,
        cfg: Config,
        *,
        env: Wrapper,
        models: dict[str, Model],
    ) -> None:
        """Initialize common tracking state."""
        ConfiguredBase.__init__(self, cfg)
        Agent.__init__(
            self,
            cfg=cfg,
            models=models,
            memory=None,
            observation_space=env.observation_space,
            state_space=env.state_space,
            action_space=env.action_space,
            device=cfg.device,
        )
        self.memory = self.build_memory(env=env)
        self.last_tracking_metrics: dict[str, float] = {}

    @abstractmethod
    def build_memory(self, *, env: Wrapper) -> Memory | None:
        """Build agent memory."""

    def write_tracking_data(self, timestep: int, timesteps: int) -> None:
        """Flush tracked metrics to MLflow and reset local buffers."""
        del timesteps

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
