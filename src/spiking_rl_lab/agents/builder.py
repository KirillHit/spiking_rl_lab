"""Agent factory entry point."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from skrl.memories.torch import RandomMemory

from spiking_rl_lab.utils.exception import AgentCreationError

if TYPE_CHECKING:
    from skrl.envs.wrappers.torch import Wrapper
    from skrl.models.torch import Model

    from spiking_rl_lab.utils.config import AgentConfig

    from .base_agent import BaseAgent

log = logging.getLogger(__name__)


def build_agent(cfg: AgentConfig, env: Wrapper, models: dict[str, Model]) -> BaseAgent:
    """Build an agent according to the provided configuration.

    Raises:
        AgentCreationError: If the agent name is unsupported.

    """
    log.info("Creating agent '%s'...", cfg.name)

    match cfg.name:
        case "ppo":
            from .ppo import PPO

            agent_class = PPO
        case _:
            msg = f"Unsupported agent: {cfg.name}"
            raise AgentCreationError(msg)

    memory = RandomMemory(memory_size=cfg.memory_size, num_envs=env.num_envs, device=cfg.device)

    return agent_class(models, memory, env.observation_space, env.action_space, cfg.device)
