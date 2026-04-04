"""Agent factory entry point."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar

from spiking_rl_lab.agents.base_agent import BaseAgent, BaseAgentCfg
from spiking_rl_lab.utils.exception import AgentCreationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from skrl.envs.wrappers.torch import Wrapper
    from skrl.models.torch import Model

    from spiking_rl_lab.utils.config import AgentConfig


log = logging.getLogger(__name__)

TAgent = TypeVar("TAgent", bound="BaseAgent")

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {}


def _build_agent_cfg(agent_cls: type[BaseAgent], params: dict[str, Any]) -> BaseAgentCfg:
    """Build a typed agent config for ``agent_cls`` from raw params."""
    cfg_cls = agent_cls.cfg_cls
    if not issubclass(cfg_cls, BaseAgentCfg):
        msg = f"{agent_cls.__name__}.cfg_cls must inherit BaseAgentCfg"
        raise AgentCreationError(msg)

    try:
        return cfg_cls(**params)
    except Exception as exc:
        msg = f"Invalid config for agent '{agent_cls.__name__}'"
        raise AgentCreationError(msg) from exc


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

    typed_cfg = _build_agent_cfg(agent_class, cfg.params)

    try:
        memory = agent_class.build_memory(cfg=cfg, env=env)
        return agent_class(
            models=models,
            memory=memory,
            observation_space=env.observation_space,
            state_space=env.state_space,
            action_space=env.action_space,
            device=cfg.device,
            cfg=typed_cfg,
        )
    except AgentCreationError:
        raise
    except Exception as exc:
        msg = f"Failed to create agent '{cfg.name}'"
        raise AgentCreationError(msg) from exc


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
