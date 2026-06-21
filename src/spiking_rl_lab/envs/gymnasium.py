"""Build a Gymnasium-backed environment and wrap it for skrl."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiking_rl_lab.core.exception import EnvironmentCreationError
from spiking_rl_lab.envs.base_env import BaseEnvBackend
from spiking_rl_lab.envs.builder import register_env

if TYPE_CHECKING:
    from skrl.envs.wrappers.torch import Wrapper

log = logging.getLogger(__name__)


@register_env("gymnasium")
class GymnasiumBackend(BaseEnvBackend):
    """Gymnasium environment backend."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEnvBackend.Config):
        """Gymnasium backend configuration."""

        id: str
        render: bool = False
        n_envs: int = 1

        def __post_init__(self) -> None:
            """Validate environment backend parameters."""
            if self.n_envs < 1:
                msg = f"Number of environments must be >= 1 (got {self.n_envs})"
                raise ValueError(msg)

    def build(self) -> Wrapper:
        """Build a Gymnasium-backed environment and wrap it for skrl."""
        import gymnasium as gym
        from skrl.envs.wrappers.torch import wrap_env

        try:
            if self._cfg.n_envs == 1:
                env = gym.make(self._cfg.id, render_mode="human" if self._cfg.render else None)
            else:
                env = gym.make_vec(
                    self._cfg.id,
                    num_envs=self._cfg.n_envs,
                    vectorization_mode="sync",
                    render_mode="human" if self._cfg.render else None,
                )
        except Exception as exc:
            msg = f"Failed to create Gymnasium environment '{self._cfg.id}': {exc}"
            raise EnvironmentCreationError(msg) from exc

        try:
            wrapped_env = wrap_env(env, verbose=False)
        except Exception as exc:
            msg = f"Failed to wrap Gymnasium environment '{self._cfg.id}': {exc}"
            raise EnvironmentCreationError(msg) from exc

        return wrapped_env
