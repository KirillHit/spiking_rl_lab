"""Build a Gymnasium-backed environment and wrap it for skrl."""

import logging

import gymnasium as gym
from skrl.envs.wrappers.torch import Wrapper, wrap_env

from spiking_rl_lab.utils.config import EnvConfig
from spiking_rl_lab.utils.exception import EnvironmentCreationError

log = logging.getLogger(__name__)


def build_gymnasium(cfg: EnvConfig) -> Wrapper:
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
