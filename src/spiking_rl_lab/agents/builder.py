"""Agent factory entry point."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from spiking_rl_lab.agents.base_agent import BaseAgent
from spiking_rl_lab.core.exception import AgentCreationError
from spiking_rl_lab.core.factory import (
    FactoryConfig,
    RegistrySpec,
    build_configured_instance,
    import_registry_modules,
    register_in_registry,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from skrl.envs.wrappers.torch import Wrapper

log = logging.getLogger(__name__)
_BUILTIN_AGENT_MODULES = ("spiking_rl_lab.agents.reinforce",)

TAgent = TypeVar("TAgent", bound="BaseAgent")


AGENT_REGISTRY: dict[str, type[BaseAgent]] = {}
AGENT_SPEC = RegistrySpec[BaseAgent](
    registry=AGENT_REGISTRY,
    base_cls=BaseAgent,
    error_cls=AgentCreationError,
    kind="agent",
)


@dataclass(kw_only=True, slots=True)
class AgentConfig(FactoryConfig):
    """Configuration for a registered agent."""


def build_agent(cfg: AgentConfig, env: Wrapper) -> BaseAgent:
    """Build an agent according to the provided configuration.

    Raises:
        AgentCreationError: If the agent name is unsupported.

    """
    log.info("Creating agent '%s'...", cfg.name)
    import_registry_modules(_BUILTIN_AGENT_MODULES, AGENT_SPEC)

    try:
        agent = build_configured_instance(
            cfg,
            spec=AGENT_SPEC,
            dependencies={"env": env},
        )
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
