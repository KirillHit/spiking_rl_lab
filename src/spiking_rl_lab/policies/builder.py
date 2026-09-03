"""Policy factory entry point."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiking_rl_lab.core.exception import PolicyCreationError
from spiking_rl_lab.core.factory import (
    FactoryConfig,
    RegistrySpec,
    build_configured_instance,
    import_registry_modules,
    register_in_registry,
)
from spiking_rl_lab.policies.base_policy import BasePolicy

if TYPE_CHECKING:
    from collections.abc import Callable

    import gymnasium

log = logging.getLogger(__name__)


POLICY_MODULES = ["spiking_rl_lab.policies.beta", "spiking_rl_lab.policies.standard"]
POLICY_REGISTRY: dict[str, type[BasePolicy]] = {}
POLICY_SPEC = RegistrySpec[BasePolicy](
    registry=POLICY_REGISTRY,
    base_cls=BasePolicy,
    error_cls=PolicyCreationError,
    kind="policy",
)


@dataclass(kw_only=True, slots=True)
class PolicyConfig(FactoryConfig):
    """Configuration for a registered policy adapter."""


def register_policy(name: str) -> Callable[[type[BasePolicy]], type[BasePolicy]]:
    """Register a policy class under ``name``."""
    return register_in_registry(name, POLICY_SPEC)


def build_policy(cfg: PolicyConfig, *, action_space: gymnasium.Space) -> BasePolicy:
    """Build one configured policy adapter."""
    log.info("Creating policy '%s'...", cfg.name)
    import_registry_modules(POLICY_MODULES, POLICY_SPEC)
    return build_configured_instance(
        cfg,
        spec=POLICY_SPEC,
        dependencies={"action_space": action_space},
    )
