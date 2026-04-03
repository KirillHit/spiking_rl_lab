"""Model factory entry point."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar

import gymnasium as gym
from skrl.models.torch import CategoricalMixin, DeterministicMixin, GaussianMixin, Model

from spiking_rl_lab.models.base_model import BaseModel, BaseModelCfg
from spiking_rl_lab.utils.config import ModelRole
from spiking_rl_lab.utils.exception import ModelCreationError

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from skrl.envs.wrappers.torch import Wrapper

    from spiking_rl_lab.utils.config import ModelsConfig

log = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound="BaseModel")

MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def _build_model_cfg(model_cls: type[BaseModel], params: dict[str, Any]) -> BaseModelCfg:
    """Build a typed model config for ``model_cls`` from raw params."""
    cfg_cls = model_cls.cfg_cls
    if not issubclass(cfg_cls, BaseModelCfg):
        msg = f"{model_cls.__name__}.cfg_cls must inherit {BaseModelCfg.__name__}"
        raise ModelCreationError(msg)

    try:
        return cfg_cls(**params)
    except Exception as exc:
        msg = f"Invalid config for model '{model_cls.__name__}'"
        raise ModelCreationError(msg) from exc


def _select_policy_mixin(action_space: gym.Space, *, gaussian: bool = True) -> type:
    """Select policy mixin for the action space.

    Args:
        action_space: Environment action space.
        gaussian: Whether continuous policies should be stochastic.

    Returns:
        A skrl mixin class.

    Raises:
        ModelCreationError: If the action space is unsupported.

    """
    if isinstance(action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
        return CategoricalMixin
    if isinstance(action_space, gym.spaces.Box):
        return GaussianMixin if gaussian else DeterministicMixin

    msg = f"Unsupported action_space for policy mixin: {type(action_space)}"
    raise ModelCreationError(msg)


def _init_policy_mixin(model: BaseModel, mixin_cls: type) -> None:
    """Initialize policy mixin from ``model.cfg``.

    Args:
        model: Model instance.
        mixin_cls: Selected skrl mixin.

    Raises:
        ModelCreationError: If the mixin is unsupported.

    """
    cfg = model.cfg
    if mixin_cls is CategoricalMixin:
        CategoricalMixin.__init__(model, unnormalized_log_prob=cfg.unnormalized_log_prob)
        return
    if mixin_cls is GaussianMixin:
        GaussianMixin.__init__(
            model,
            clip_actions=cfg.clip_actions,
            clip_mean_actions=cfg.clip_mean_actions,
            clip_log_std=cfg.clip_log_std,
            min_log_std=cfg.min_log_std,
            max_log_std=cfg.max_log_std,
            reduction=cfg.reduction,
        )
        return
    if mixin_cls is DeterministicMixin:
        DeterministicMixin.__init__(model, clip_actions=cfg.clip_actions)
        return

    msg = f"Unsupported policy mixin: {mixin_cls}"
    raise ModelCreationError(msg)


def _get_policy_hook(mixin_cls: type) -> str:
    """Get policy hook name expected by a mixin.

    Args:
        mixin_cls: Selected skrl mixin.

    Returns:
        Method name on the core model.

    Raises:
        ModelCreationError: If the mixin is unsupported.

    """
    if mixin_cls is CategoricalMixin:
        return "compute_categorical"
    if mixin_cls is GaussianMixin:
        return "compute_gaussian"
    if mixin_cls is DeterministicMixin:
        return "compute_deterministic"

    msg = f"Unsupported policy mixin: {mixin_cls}"
    raise ModelCreationError(msg)


def _build_policy_class(core_cls: type[BaseModel], mixin_cls: type) -> type[BaseModel]:
    """Build policy wrapper class.

    Args:
        core_cls: Core model class.
        mixin_cls: Selected skrl mixin.

    Returns:
        Wrapped policy model class.

    """
    name = f"{core_cls.__name__}__{mixin_cls.__name__}"
    hook_name = _get_policy_hook(mixin_cls)
    if not callable(getattr(core_cls, hook_name, None)):
        msg = f"{core_cls.__name__} must implement {hook_name}() for policy role"
        raise ModelCreationError(msg)

    class Mixed(mixin_cls, core_cls):
        __module__ = core_cls.__module__
        cfg_cls = core_cls.cfg_cls

        def __init__(
            self,
            *,
            cfg: BaseModelCfg,
            observation_space: gym.Space | None = None,
            state_space: gym.Space | None = None,
            action_space: gym.Space | None = None,
            device: str | torch.device | None = None,
        ) -> None:
            core_cls.__init__(
                self,
                cfg=cfg,
                observation_space=observation_space,
                state_space=state_space,
                action_space=action_space,
                device=device,
            )
            _init_policy_mixin(self, mixin_cls)

        def compute(
            self,
            inputs: dict[str, Any],
            role: str = "",
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            return getattr(self, hook_name)(inputs, role)

    Mixed.__name__ = name
    Mixed.__qualname__ = name

    return Mixed


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
        log.info("Creating model '%s' with role '%s'...", model_cfg.name, role_name)

        core_cls = MODEL_REGISTRY.get(model_cfg.name)
        if core_cls is None:
            msg = f"Unsupported model: {model_cfg.name}"
            raise ModelCreationError(msg)

        if role_name in models:
            msg = f"Duplicate model role '{role_name}' in models config"
            raise ModelCreationError(msg)

        typed_cfg = _build_model_cfg(core_cls, model_cfg.params)

        if model_cfg.role is ModelRole.policy:
            mixin_cls = _select_policy_mixin(action_space, gaussian=model_cfg.gaussian)
            cls = _build_policy_class(core_cls, mixin_cls)
        elif model_cfg.role is ModelRole.value:
            cls = core_cls
        else:
            msg = f"Unsupported model role: {model_cfg.role}"
            raise ModelCreationError(msg)

        try:
            models[role_name] = cls(
                observation_space=observation_space,
                state_space=state_space,
                action_space=action_space,
                device=model_cfg.device,
                cfg=typed_cfg,
            )
        except ModelCreationError:
            raise
        except Exception as exc:
            msg = f"Failed to create model '{model_cfg.name}' for role '{role_name}'"
            raise ModelCreationError(msg) from exc

    return models


def register_model(name: str) -> Callable[[type[TModel]], type[TModel]]:
    """Register model class.

    Args:
        name: Registry key.

    Returns:
        Class decorator.

    """

    def decorator(cls: type[TModel]) -> type[TModel]:
        if not issubclass(cls, BaseModel):
            msg = f"Registered class must inherit {BaseModel.__name__}, got: {cls!r}"
            raise TypeError(msg)

        if name in MODEL_REGISTRY:
            msg = f"Model name '{name}' is already registered"
            raise ModelCreationError(msg)

        MODEL_REGISTRY[name] = cls
        return cls

    return decorator
