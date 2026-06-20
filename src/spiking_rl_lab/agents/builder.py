"""Agent factory entry point."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, TypeVar

from spiking_rl_lab.agents.base_agent import BaseAgent
from spiking_rl_lab.core.exception import AgentCreationError
from spiking_rl_lab.core.factory import (
    FactoryConfig,
    RegistrySpec,
    build_configured_instance,
    register_in_registry,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from skrl.envs.wrappers.torch import Wrapper
    from skrl.models.torch import Model


log = logging.getLogger(__name__)
_BUILTIN_AGENT_MODULES = ("spiking_rl_lab.agents.reinforce.reinforce",)
_REGISTERED_BUILTIN_MODULES: set[str] = set()

TAgent = TypeVar("TAgent", bound="BaseAgent")


@dataclass(kw_only=True, slots=True)
class AgentConfig(FactoryConfig):
    """RL algorithm configuration."""

    device: str = "cpu"


AGENT_REGISTRY: dict[str, type[BaseAgent]] = {}
AGENT_SPEC = RegistrySpec[BaseAgent](
    registry=AGENT_REGISTRY,
    base_cls=BaseAgent,
    error_cls=AgentCreationError,
    kind="agent",
)


def _register_builtin_agents() -> None:
    """Import built-in agents once so decorators register them."""
    for module_name in _BUILTIN_AGENT_MODULES:
        if module_name in _REGISTERED_BUILTIN_MODULES:
            continue
        import_module(module_name)
        _REGISTERED_BUILTIN_MODULES.add(module_name)


def build_agent(cfg: AgentConfig, env: Wrapper, models: dict[str, Model]) -> BaseAgent:
    """Build an agent according to the provided configuration.

    Raises:
        AgentCreationError: If the agent name is unsupported.

    """
    log.info("Creating agent '%s'...", cfg.name)
    _register_builtin_agents()

    try:
        agent = build_configured_instance(
            cfg,
            spec=AGENT_SPEC,
            dependencies={
                "models": models,
                "memory": None,
                "observation_space": env.observation_space,
                "state_space": env.state_space,
                "action_space": env.action_space,
                "device": cfg.device,
            },
        )
        agent.memory = agent.build_memory(env=env)
    except AgentCreationError:
        raise
    except Exception as exc:
        msg = f"Failed to create agent '{cfg.name}'"
        raise AgentCreationError(msg) from exc
    else:
        return agent


def register_agent(name: str) -> Callable[[type[TAgent]], type[TAgent]]:
    """Register an agent class under a given name."""
    return register_in_registry(name, AGENT_SPEC)
