"""Model factory entry point."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

from omegaconf import MISSING

from spiking_rl_lab.core.exception import ModelCreationError
from spiking_rl_lab.core.factory import (
    FactoryConfig,
    RegistrySpec,
    build_configured_instance,
    register_in_registry,
)
from spiking_rl_lab.models.base_model import BaseModel
from spiking_rl_lab.networks.builder import NetworkBuildContext, NetworkConfig

if TYPE_CHECKING:
    from collections.abc import Callable

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


@dataclass(kw_only=True, slots=True)
class ModelConfig(FactoryConfig):
    """Configuration for a single model instance."""

    device: str = "cpu"
    role: str = MISSING


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


def build_models(
    cfg: list[ModelConfig],
    networks_cfg: dict[str, NetworkConfig],
    env: Wrapper,
) -> dict[str, Model]:
    """Build experiment models."""
    models: dict[str, Model] = {}
    observation_space = env.observation_space
    state_space = env.state_space
    action_space = env.action_space
    network_builder = NetworkBuildContext(networks_cfg, env)
    _register_model_modules()

    for model_cfg in cfg:
        role = model_cfg.role
        log.info("Creating model '%s' with role '%s'...", model_cfg.name, role)

        if role in models:
            msg = f"Duplicate model role '{role}' in models config"
            raise ModelCreationError(msg)

        try:
            model = build_configured_instance(
                model_cfg,
                spec=MODEL_SPEC,
                dependencies={
                    "observation_space": observation_space,
                    "state_space": state_space,
                    "action_space": action_space,
                    "device": model_cfg.device,
                    "network_builder": network_builder,
                },
            )
            models[role] = model
        except ModelCreationError:
            raise
        except Exception as exc:
            msg = f"Failed to create model for role '{role}'"
            raise ModelCreationError(msg) from exc

    return models
