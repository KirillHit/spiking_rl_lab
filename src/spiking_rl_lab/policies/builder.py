"""Policy factory entry point."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING

from spiking_rl_lab.core.exception import PolicyCreationError
from spiking_rl_lab.core.factory import (
    FactoryConfig,
    RegistrySpec,
    build_configured_instance,
    register_in_registry,
)
from spiking_rl_lab.policies.base_policy import BasePolicy

if TYPE_CHECKING:
    from collections.abc import Callable

    import gymnasium

log = logging.getLogger(__name__)


POLICY_MODULES = ["spiking_rl_lab.policies.standard"]
POLICY_REGISTRY: dict[str, type[BasePolicy]] = {}
POLICY_SPEC = RegistrySpec[BasePolicy](
    registry=POLICY_REGISTRY,
    base_cls=BasePolicy,
    error_cls=PolicyCreationError,
    kind="policy",
)


def register_policy(name: str) -> Callable[[type[BasePolicy]], type[BasePolicy]]:
    """Register a policy class under ``name``."""
    return register_in_registry(name, POLICY_SPEC)


def _register_policy_modules() -> None:
    """Import policy implementations so decorators register them."""
    for module_name in POLICY_MODULES:
        try:
            import_module(module_name)
        except ImportError as exc:
            msg = f"Failed to import policy module '{module_name}': {exc}"
            raise PolicyCreationError(msg) from exc


def build_policy(
    cfg: FactoryConfig,
    *,
    action_space: gymnasium.Space,
) -> BasePolicy:
    """Build one configured policy adapter."""
    _register_policy_modules()
    log.info("Creating policy '%s'...", cfg.name)
    try:
        return build_configured_instance(
            cfg,
            spec=POLICY_SPEC,
            dependencies={"action_space": action_space},
        )
    except PolicyCreationError:
        raise
    except Exception as exc:
        msg = f"Failed to create policy '{cfg.name}'"
        raise PolicyCreationError(msg) from exc
