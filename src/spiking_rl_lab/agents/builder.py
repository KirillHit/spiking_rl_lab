"""Agent factory entry point."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

from skrl.memories.torch import RandomMemory

from spiking_rl_lab.agents.base_agent import BaseAgent
from spiking_rl_lab.utils.exception import AgentCreationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from skrl.envs.wrappers.torch import Wrapper
    from skrl.models.torch import Model

    from spiking_rl_lab.utils.config import AgentConfig


log = logging.getLogger(__name__)

TAgent = TypeVar("TAgent", bound="BaseAgent")

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {}


def build_agent(cfg: AgentConfig, env: Wrapper, models: dict[str, Model]) -> BaseAgent:
    """Build an agent according to the provided configuration.

    Raises:
        AgentCreationError: If the agent name is unsupported.

    """
    log.info("Creating agent '%s'...", cfg.name)

    agent_class = AGENT_REGISTRY.get(cfg.name)

    if agent_class is None:
        msg = f"Unsupported agent: {cfg.name}"
        raise AgentCreationError(msg)

    memory = RandomMemory(memory_size=cfg.memory_size, num_envs=env.num_envs, device=cfg.device)

    return agent_class(
        models=models,
        memory=memory,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=cfg.device,
        cfg=cfg.params,
    )


def register_agent(name: str) -> Callable[[type[TAgent]], type[TAgent]]:
    """Register an agent class under a given name."""

    def decorator(cls: type[TAgent]) -> type[TAgent]:
        if not issubclass(cls, BaseAgent):
            msg = f"Registered class must inherit {BaseAgent.__name__}, got: {cls!r}"
            raise TypeError(msg)

        if name in AGENT_REGISTRY:
            msg = f"Agent name '{name}' is already registered"
            raise AgentCreationError(msg)

        AGENT_REGISTRY[name] = cls
        return cls

    return decorator
