"""Model factory entry point."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING

from spiking_rl_lab.core.exception import ModelCreationError
from spiking_rl_lab.core.factory import (
    FactoryConfig,
    RegistrySpec,
    build_configured_instance,
    register_in_registry,
)
from spiking_rl_lab.models.base_model import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from skrl.envs.wrappers.torch import Wrapper
    from skrl.models.torch import Model

log = logging.getLogger(__name__)


MODEL_MODULES = ["spiking_rl_lab.models.skrl"]
MODEL_REGISTRY: dict[str, type[BaseModel]] = {}
MODEL_SPEC = RegistrySpec[BaseModel](
    registry=MODEL_REGISTRY,
    base_cls=BaseModel,
    error_cls=ModelCreationError,
    kind="model",
)


def register_model(name: str) -> Callable[[type[BaseModel]], type[BaseModel]]:
    """Register a model class under a given name."""
    return register_in_registry(name, MODEL_SPEC)


def _register_model_modules() -> None:
    """Import model implementation modules so decorators register them."""
    for module_name in MODEL_MODULES:
        try:
            import_module(module_name)
        except ImportError as exc:
            msg = f"Failed to import model module '{module_name}': {exc}"
            raise ModelCreationError(msg) from exc


def build_model(
    cfg: FactoryConfig,
    env: Wrapper,
    *,
    device: str | torch.device | None,
) -> Model:
    """Build one configured model for an agent."""
    observation_space = env.observation_space
    state_space = env.state_space
    action_space = env.action_space
    _register_model_modules()
    log.info("Creating model '%s'...", cfg.name)

    try:
        return build_configured_instance(
            cfg,
            spec=MODEL_SPEC,
            dependencies={
                "observation_space": observation_space,
                "state_space": state_space,
                "action_space": action_space,
                "device": device,
            },
        )
    except ModelCreationError:
        raise
    except Exception as exc:
        msg = f"Failed to create model '{cfg.name}'"
        raise ModelCreationError(msg) from exc
