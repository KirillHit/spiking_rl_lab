"""Shared base classes for agents."""

from __future__ import annotations

import dataclasses
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import mlflow
import numpy as np
from skrl.agents.torch import Agent, AgentCfg

if TYPE_CHECKING:
    import gymnasium
    import torch
    from skrl.envs.wrappers.torch import Wrapper
    from skrl.memories.torch import Memory
    from skrl.models.torch import Model


@dataclasses.dataclass(kw_only=True)
class BaseAgentCfg(AgentCfg):
    """Base class for the agent's configuration."""


class BaseAgent(Agent, ABC):
    """Common utilities for agents used in this project."""

    cfg_cls: ClassVar[type[BaseAgentCfg]] = BaseAgentCfg

    def __init__(
        self,
        *,
        cfg: BaseAgentCfg,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize common tracking state."""
        super().__init__(
            cfg=cfg,
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
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
