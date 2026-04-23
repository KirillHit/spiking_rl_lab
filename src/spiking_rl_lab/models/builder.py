"""Model factory entry point."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gymnasium as gym

from spiking_rl_lab.models.base_model import (
    CategoricalPolicyModel,
    DeterministicPolicyModel,
    GaussianPolicyModel,
    ValueModel,
)
from spiking_rl_lab.utils.config import ModelRole
from spiking_rl_lab.utils.exception import ModelCreationError

if TYPE_CHECKING:
    from skrl.envs.wrappers.torch import Wrapper
    from skrl.models.torch import Model

    from spiking_rl_lab.utils.config import ModelsConfig

log = logging.getLogger(__name__)


def _select_policy_class(action_space: gym.Space, *, gaussian: bool = True) -> type:
    """Select policy model class for the action space.

    Args:
        action_space: Environment action space.
        gaussian: Whether continuous policies should be stochastic.

    Returns:
        A concrete policy model class.

    Raises:
        ModelCreationError: If the action space is unsupported.

    """
    if isinstance(action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
        return CategoricalPolicyModel
    if isinstance(action_space, gym.spaces.Box):
        return GaussianPolicyModel if gaussian else DeterministicPolicyModel

    msg = f"Unsupported action_space for policy model: {type(action_space)}"
    raise ModelCreationError(msg)


def build_models(cfg: ModelsConfig, env: Wrapper) -> dict[str, Model]:
    """Build experiment models.

    Args:
        cfg: Model configuration group.
        env: Wrapped environment.

    Returns:
        Models indexed by role name.

    Raises:
        ModelCreationError: If model construction fails.

    """
    models: dict[str, Model] = {}
    observation_space = env.observation_space
    state_space = env.state_space
    action_space = env.action_space

    for model_cfg in cfg.models:
        role_name = model_cfg.role.value
        log.info("Creating model with role '%s'...", role_name)

        if role_name in models:
            msg = f"Duplicate model role '{role_name}' in models config"
            raise ModelCreationError(msg)

        if model_cfg.role is ModelRole.policy:
            cls = _select_policy_class(action_space, gaussian=model_cfg.gaussian)
        elif model_cfg.role is ModelRole.value:
            cls = ValueModel
        else:
            msg = f"Unsupported model role: {model_cfg.role}"
            raise ModelCreationError(msg)

        try:
            models[role_name] = cls(
                cfg=model_cfg.model,
                observation_space=observation_space,
                state_space=state_space,
                action_space=action_space,
                device=model_cfg.device,
            )
        except ModelCreationError:
            raise
        except Exception as exc:
            msg = f"Failed to create model for role '{role_name}'"
            raise ModelCreationError(msg) from exc

    return models
