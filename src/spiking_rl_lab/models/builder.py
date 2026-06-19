"""Model factory entry point."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypeVar

import gymnasium as gym
from omegaconf import MISSING

from spiking_rl_lab.core.exception import ModelCreationError
from spiking_rl_lab.core.factory import (
    FactoryConfig,
    RegistrySpec,
    build_configured_instance,
    register_in_registry,
)
from spiking_rl_lab.models.base_model import (
    BaseModel,
    CategoricalPolicyModel,
    DeterministicPolicyModel,
    GaussianPolicyModel,
    ValueModel,
)
from spiking_rl_lab.network.builder import build_network
from spiking_rl_lab.network.shape import DenseTensorShape, TensorShape

if TYPE_CHECKING:
    from collections.abc import Callable

    from skrl.envs.wrappers.torch import Wrapper
    from skrl.models.torch import Model

    from spiking_rl_lab.network.builder import NetworkConfig

log = logging.getLogger(__name__)
TModel = TypeVar("TModel", bound="BaseModel")

AUTO_MODEL_NAME = "auto"


class ModelRole(StrEnum):
    """Role of a model within an agent's architecture."""

    policy = "policy"
    value = "value"


class PolicyType(StrEnum):
    """Policy semantics required by an agent."""

    stochastic = "stochastic"
    deterministic = "deterministic"


@dataclass(kw_only=True, slots=True)
class ModelConfig(FactoryConfig):
    """Configuration for a single model instance."""

    name: str = AUTO_MODEL_NAME
    role: ModelRole = MISSING
    device: str = "cpu"
    policy_type: PolicyType = PolicyType.stochastic
    network: str = MISSING


MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "categorical_policy": CategoricalPolicyModel,
    "gaussian_policy": GaussianPolicyModel,
    "deterministic_policy": DeterministicPolicyModel,
    "value": ValueModel,
}
MODEL_SPEC = RegistrySpec[BaseModel](
    registry=MODEL_REGISTRY,
    base_cls=BaseModel,
    error_cls=ModelCreationError,
    kind="model",
)


def _observation_shape(observation_space: gym.Space) -> TensorShape:
    """Return the flattened observation shape consumed by base models."""
    return TensorShape.dense(gym.spaces.utils.flatdim(observation_space))


def _select_policy_name(action_space: gym.Space, *, policy_type: PolicyType) -> str:
    """Select a policy model registry name for the action space."""
    policy_type = PolicyType(policy_type)
    if isinstance(action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
        if policy_type is PolicyType.deterministic:
            msg = "Deterministic policy model is unsupported for discrete action spaces"
            raise ModelCreationError(msg)
        return "categorical_policy"
    if isinstance(action_space, gym.spaces.Box):
        return "gaussian_policy" if policy_type is PolicyType.stochastic else "deterministic_policy"

    msg = f"Unsupported action_space for policy model: {type(action_space)}"
    raise ModelCreationError(msg)


def _resolve_model_name(model_cfg: ModelConfig, action_space: gym.Space) -> str:
    """Resolve ``auto`` to a concrete model registry name."""
    if model_cfg.name != AUTO_MODEL_NAME:
        return model_cfg.name
    role = ModelRole(model_cfg.role)
    if role is ModelRole.policy:
        return _select_policy_name(action_space, policy_type=model_cfg.policy_type)
    if role is ModelRole.value:
        return "value"

    msg = f"Unsupported model role: {model_cfg.role}"
    raise ModelCreationError(msg)


def _network_cfg(cfg: dict[str, NetworkConfig], model_cfg: ModelConfig) -> NetworkConfig:
    """Return the network config referenced by a model config."""
    network_cfg = cfg.get(model_cfg.network)
    if network_cfg is None:
        available = ", ".join(sorted(cfg)) or "<empty>"
        msg = (
            f"Unsupported network reference '{model_cfg.network}' for model role "
            f"'{model_cfg.role}'. Available networks: {available}"
        )
        raise ModelCreationError(msg)
    return network_cfg


def _model_factory_config(model_cfg: ModelConfig, model_name: str) -> ModelConfig:
    """Build a concrete factory config from a Hydra-backed model config."""
    return ModelConfig(
        name=model_name,
        role=ModelRole(model_cfg.role),
        device=model_cfg.device,
        policy_type=PolicyType(model_cfg.policy_type),
        network=model_cfg.network,
        params=dict(model_cfg.params),
    )


def _validate_network_output(model: BaseModel, role_name: str) -> None:
    """Validate the network output width expected by skrl models."""
    output_shape = model.network_output_shape
    if not isinstance(output_shape, DenseTensorShape):
        msg = f"Model role '{role_name}' requires dense network output, got {output_shape.kind}"
        raise ModelCreationError(msg)

    expected = 1 if role_name == ModelRole.value.value else model.num_actions
    if output_shape.features != expected:
        msg = (
            f"Model role '{role_name}' requires network output width {expected}, "
            f"got {output_shape.features}"
        )
        raise ModelCreationError(msg)


def build_models(
    cfg: list[ModelConfig],
    networks_cfg: dict[str, NetworkConfig],
    env: Wrapper,
) -> dict[str, Model]:
    """Build experiment models.

    Args:
        cfg: Model configuration group.
        networks_cfg: Named network configurations.
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
    input_shape = _observation_shape(observation_space)

    for model_cfg in cfg:
        role_name = ModelRole(model_cfg.role).value
        model_name = _resolve_model_name(model_cfg, action_space)
        log.info("Creating model '%s' with role '%s'...", model_name, role_name)

        if role_name in models:
            msg = f"Duplicate model role '{role_name}' in models config"
            raise ModelCreationError(msg)

        try:
            network = build_network(_network_cfg(networks_cfg, model_cfg), input_shape)
            model = build_configured_instance(
                _model_factory_config(model_cfg, model_name),
                spec=MODEL_SPEC,
                dependencies={
                    "observation_space": observation_space,
                    "state_space": state_space,
                    "action_space": action_space,
                    "device": model_cfg.device,
                    "network": network,
                },
            )
            _validate_network_output(model, role_name)
            models[role_name] = model
        except ModelCreationError:
            raise
        except Exception as exc:
            msg = f"Failed to create model for role '{role_name}'"
            raise ModelCreationError(msg) from exc

    return models


def register_model(name: str) -> Callable[[type[TModel]], type[TModel]]:
    """Register a model class under a given name."""
    return register_in_registry(name, MODEL_SPEC)
