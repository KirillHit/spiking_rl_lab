"""Model factory entry point."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from spiking_rl_lab.utils.exception import ModelCreationError

if TYPE_CHECKING:
    from skrl.envs.wrappers.torch import Wrapper
    from skrl.models.torch import Model

    from spiking_rl_lab.utils.config import ModelsConfig

log = logging.getLogger(__name__)


def build_models(cfg: ModelsConfig, env: Wrapper) -> dict[str, Model]:
    """Build models according to the provided configuration.

    Raises:
        ModelCreationError: If the model name is unsupported.

    """
    models: dict[str, Model] = {}

    for model_cfg in cfg.models:
        log.info("Creating model '%s' with role '%s'...", model_cfg.name, model_cfg.role.value)

        match model_cfg.name:
            case "mlp":
                from .mlp import Policy, Value

                if model_cfg.role.value == "policy":
                    model_class = Policy
                else:
                    model_class = Value

            case _:
                msg = f"Unsupported model: {model_cfg.name}!"
                raise ModelCreationError(msg)

        models[model_cfg.role.value] = model_class(
            env.observation_space,
            env.action_space,
            model_cfg.device,
            **model_cfg.params,
        )

    return models
