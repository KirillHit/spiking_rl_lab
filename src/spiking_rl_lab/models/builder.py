"""Model factory entry point."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar

import gymnasium as gym
from skrl.models.torch import CategoricalMixin, DeterministicMixin, GaussianMixin, Model

from spiking_rl_lab.models.base_model import BaseModel, BaseModelCfg
from spiking_rl_lab.utils.exception import ModelCreationError

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from skrl.envs.wrappers.torch import Wrapper

    from spiking_rl_lab.utils.config import ModelsConfig

log = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound="BaseModel")

MODEL_REGISTRY: dict[str, type[BaseModel]] = {}

_MIXED_CACHE: dict[tuple[type, type], type] = {}


def _select_mixin(action_space: gym.Space, *, gaussian: bool = True) -> type:
    if isinstance(action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
        return CategoricalMixin
    if isinstance(action_space, gym.spaces.Box):
        return GaussianMixin if gaussian else DeterministicMixin
    msg = f"Unsupported action_space: {type(action_space)}"
    raise TypeError(msg)


def _get_mixed_class(core_cls: type[BaseModel], mixin_cls: type) -> type[BaseModel]:
    key = (core_cls, mixin_cls)
    if key in _MIXED_CACHE:
        return _MIXED_CACHE[key]

    name = f"{core_cls.__name__}__{mixin_cls.__name__}"

    class Mixed(mixin_cls, core_cls):
        __name__ = name

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

            if mixin_cls is CategoricalMixin:
                CategoricalMixin.__init__(self, unnormalized_log_prob=cfg.unnormalized_log_prob)
            elif mixin_cls is GaussianMixin:
                GaussianMixin.__init__(
                    self,
                    clip_actions=cfg.clip_actions,
                    clip_mean_actions=cfg.clip_mean_actions,
                    clip_log_std=cfg.clip_log_std,
                    min_log_std=cfg.min_log_std,
                    max_log_std=cfg.max_log_std,
                    reduction=cfg.reduction,
                )
            elif mixin_cls is DeterministicMixin:
                DeterministicMixin.__init__(self, clip_actions=cfg.clip_actions)
            else:
                msg = f"Unsupported mixin: {mixin_cls}"
                raise TypeError(msg)

        def compute(
            self,
            inputs: dict[str, Any],
            *,
            role: str = "",
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            if mixin_cls is CategoricalMixin:
                return self.compute_categorical(inputs, role)
            if mixin_cls is GaussianMixin:
                return self.compute_gaussian(inputs, role)
            if mixin_cls is DeterministicMixin:
                return self.compute_deterministic(inputs, role)
            msg = f"Unsupported mixin: {mixin_cls}"
            raise TypeError(msg)

    _MIXED_CACHE[key] = Mixed
    return Mixed


def build_models(cfg: ModelsConfig, env: Wrapper) -> dict[str, Model]:
    """Build models according to the provided configuration.

    Raises:
        ModelCreationError: If the model name is unsupported.

    """
    models: dict[str, Model] = {}

    for model_cfg in cfg.models:
        log.info("Creating model '%s' with role '%s'...", model_cfg.name, model_cfg.role.value)

        core_cls = MODEL_REGISTRY.get(model_cfg.name)
        if core_cls is None:
            msg = f"Unsupported model: {model_cfg.name}"
            raise ModelCreationError(msg)

        cls: type[BaseModel] = core_cls
        if core_cls.requires_policy_mixin():
            mixin_cls = _select_mixin(env.action_space, gaussian=model_cfg.gaussian)
            cls = _get_mixed_class(core_cls, mixin_cls)

        models[model_cfg.role.value] = cls(
            observation_space=env.observation_space,
            state_space=env.state_space,
            action_space=env.action_space,
            device=model_cfg.device,
            cfg=model_cfg.params,
        )

    return models


def register_model(name: str) -> Callable[[type[TModel]], type[TModel]]:
    """Register a model class under a given name."""

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
