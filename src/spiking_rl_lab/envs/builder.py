"""Environment factory entry point."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiking_rl_lab.core.exception import EnvironmentCreationError
from spiking_rl_lab.core.factory import (
    FactoryConfig,
    RegistrySpec,
    build_configured_instance,
    import_registry_modules,
    register_in_registry,
)
from spiking_rl_lab.envs.base_env import BaseEnvBackend

if TYPE_CHECKING:
    from collections.abc import Callable

    from skrl.envs.wrappers.torch import Wrapper

log = logging.getLogger(__name__)


ENV_BACKEND_MODULES = ["spiking_rl_lab.envs.gymnasium"]
ENV_BACKEND_REGISTRY: dict[str, type[BaseEnvBackend]] = {}
ENV_BACKEND_SPEC = RegistrySpec[BaseEnvBackend](
    registry=ENV_BACKEND_REGISTRY,
    base_cls=BaseEnvBackend,
    error_cls=EnvironmentCreationError,
    kind="environment backend",
)


@dataclass(kw_only=True, slots=True)
class EnvironmentConfig(FactoryConfig):
    """Configuration for a registered environment backend."""


def register_env(name: str) -> Callable[[type[BaseEnvBackend]], type[BaseEnvBackend]]:
    """Register an environment backend class under a given name."""
    return register_in_registry(name, ENV_BACKEND_SPEC)


def build_env(cfg: EnvironmentConfig) -> Wrapper:
    """Build a skrl-wrapped environment according to the configured backend."""
    log.info("Creating environment using backend '%s'...", cfg.name)
    import_registry_modules(ENV_BACKEND_MODULES, ENV_BACKEND_SPEC)

    backend = build_configured_instance(
        cfg,
        spec=ENV_BACKEND_SPEC,
    )
    return backend.build()
