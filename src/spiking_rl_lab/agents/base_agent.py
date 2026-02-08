"""Base agent helpers for MLflow tracking."""

import dataclasses
import re
from abc import ABC

import mlflow
import numpy as np
from skrl.agents.torch import Agent, AgentCfg


@dataclasses.dataclass(kw_only=True)
class BaseAgentCfg(AgentCfg):
    """Base class for the agent's configuration."""


class BaseAgent(Agent, ABC):
    """Common utilities for agents used in this project."""

    def write_tracking_data(self, timestep: int, timesteps: int) -> None:
        """Flush tracked metrics to MLflow and reset local buffers."""
        for key, value in self.tracking_data.items():
            safe_key = self._mlflow_key(key)
            if key.endswith("(min)"):
                mlflow.log_metric(safe_key, np.min(value), timestep)
            elif key.endswith("(max)"):
                mlflow.log_metric(safe_key, np.max(value), timestep)
            else:
                mlflow.log_metric(safe_key, np.mean(value), timestep)
        # reset data containers for next iteration
        self._track_rewards.clear()
        self._track_timesteps.clear()
        self.tracking_data.clear()

    @staticmethod
    def _mlflow_key(key: str) -> str:
        key = key.replace(" (min)", "_min").replace(" (max)", "_max").replace(" (mean)", "_mean")
        return re.sub(r"[^0-9A-Za-z_\-\. :/ ]+", "_", key)
