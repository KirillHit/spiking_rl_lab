"""Build a Gymnasium-backed environment and wrap it for skrl."""

import logging
from dataclasses import dataclass

import gymnasium as gym
from skrl.envs.wrappers.torch import Wrapper, wrap_env

from spiking_rl_lab.core.exception import EnvironmentCreationError
from spiking_rl_lab.envs.builder import BaseEnvBackend, register_env_backend

log = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class GymnasiumEnvConfig:
    """Gymnasium backend configuration."""

    id: str
    render: bool = False
    n_envs: int = 1


@register_env_backend("gymnasium")
class GymnasiumBackend(BaseEnvBackend):
    """Gymnasium environment backend."""

    Config = GymnasiumEnvConfig

    def build(self) -> Wrapper:
        """Build a Gymnasium-backed environment and wrap it for skrl."""
        return build_gymnasium(self._cfg)


def build_gymnasium(cfg: GymnasiumEnvConfig) -> Wrapper:
    """Build a Gymnasium-backed environment and wrap it for skrl."""
    if cfg.n_envs < 1:
        msg = f"Number of environments must be >= 1 (got {cfg.n_envs})"
        raise EnvironmentCreationError(msg)

    try:
        if cfg.n_envs == 1:
            env = gym.make(cfg.id, render_mode="human" if cfg.render else None)
        else:
            env = gym.make_vec(
                cfg.id,
                num_envs=cfg.n_envs,
                vectorization_mode="sync",
                render_mode="human" if cfg.render else None,
            )
    except Exception as exc:
        msg = f"Failed to create Gymnasium environment '{cfg.id}': {exc}"
        raise EnvironmentCreationError(msg) from exc

    try:
        wrapped_env = wrap_env(env, verbose=False)
    except Exception as exc:
        msg = f"Failed to wrap Gymnasium environment '{cfg.id}': {exc}"
        raise EnvironmentCreationError(msg) from exc

    return wrapped_env
