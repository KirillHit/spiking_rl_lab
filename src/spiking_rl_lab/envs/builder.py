"""Environment factory entry point."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiking_rl_lab.core.exception import EnvironmentCreationError
from spiking_rl_lab.core.factory import (
    ConfiguredBase,
    FactoryConfig,
    RegistrySpec,
    build_configured_instance,
    register_in_registry,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from skrl.envs.wrappers.torch import Wrapper

log = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class EnvConfig(FactoryConfig):
    """Registry-backed environment backend configuration."""


class BaseEnvBackend(ConfiguredBase, ABC):
    """Base class for environment backend adapters."""

    def __init__(self, cfg: object) -> None:
        """Store backend configuration."""
        super().__init__(cfg)

    @abstractmethod
    def build(self) -> Wrapper:
        """Build and return a wrapped environment."""


ENV_BACKEND_REGISTRY: dict[str, type[BaseEnvBackend]] = {}
ENV_BACKEND_SPEC = RegistrySpec[BaseEnvBackend](
    registry=ENV_BACKEND_REGISTRY,
    base_cls=BaseEnvBackend,
    error_cls=EnvironmentCreationError,
    kind="environment backend",
)


def register_env_backend(name: str) -> Callable[[type[BaseEnvBackend]], type[BaseEnvBackend]]:
    """Register an environment backend class under a given name."""
    return register_in_registry(name, ENV_BACKEND_SPEC)


def build_env(cfg: EnvConfig) -> Wrapper:
    """Build a skrl-wrapped environment according to the configured backend.

    Raises:
        EnvironmentCreationError: If the backend is unsupported.

    """
    log.info(
        "Creating environment using backend '%s'...",
        cfg.name,
    )

    if cfg.name == "gymnasium":
        from . import gymnasium as _gymnasium

        del _gymnasium

    backend = build_configured_instance(
        cfg,
        spec=ENV_BACKEND_SPEC,
    )
    return backend.build()
