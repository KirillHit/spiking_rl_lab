"""Build an Isaac Lab environment and wrap it for skrl."""

from __future__ import annotations

import logging
import sys
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiking_rl_lab.core.exception import EnvironmentCreationError
from spiking_rl_lab.envs.base_env import BaseEnvBackend
from spiking_rl_lab.envs.builder import register_env

if TYPE_CHECKING:
    from isaaclab.app import AppLauncher
    from skrl.envs.wrappers.torch import Wrapper

log = logging.getLogger(__name__)

_APP_LAUNCHER: AppLauncher | None = None


def _ensure_app_launcher(*, headless: bool, device: str) -> AppLauncher:
    """Launch Isaac Sim once per process; reuse the existing app on later calls."""
    global _APP_LAUNCHER  # noqa: PLW0603
    if _APP_LAUNCHER is not None:
        return _APP_LAUNCHER

    from isaaclab.app import AppLauncher

    # Kit reads sys.argv; Hydra overrides like experiment=... must not reach it.
    saved_argv = sys.argv.copy()
    try:
        sys.argv = [saved_argv[0]]
        log.info("Launching Isaac Sim (headless=%s, device=%s)...", headless, device)
        _APP_LAUNCHER = AppLauncher(headless=headless, device=device)
    finally:
        sys.argv = saved_argv

    return _APP_LAUNCHER


@register_env("isaaclab")
class IsaacLabBackend(BaseEnvBackend):
    """Isaac Lab environment backend."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseEnvBackend.Config):
        """Isaac Lab backend configuration."""

        id: str
        n_envs: int = 64
        device: str = "cuda:0"
        headless: bool = True

        def __post_init__(self) -> None:
            """Validate environment backend parameters."""
            if self.n_envs < 1:
                msg = f"Number of environments must be >= 1 (got {self.n_envs})"
                raise ValueError(msg)

    def build(self) -> Wrapper:
        """Build an Isaac Lab environment and wrap it for skrl."""
        import gymnasium as gym
        from skrl.envs.wrappers.torch import wrap_env

        _ensure_app_launcher(headless=self._cfg.headless, device=self._cfg.device)

        # Task gym ids register on import; must happen after AppLauncher.
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        try:
            env_cfg = parse_env_cfg(
                self._cfg.id,
                device=self._cfg.device,
                num_envs=self._cfg.n_envs,
            )
            env_cfg.compute_final_obs = True
            env = gym.make(self._cfg.id, cfg=env_cfg)
        except Exception as exc:
            msg = f"Failed to create Isaac Lab environment '{self._cfg.id}': {exc}"
            raise EnvironmentCreationError(msg) from exc

        try:
            wrapped_env = wrap_env(env, wrapper="isaaclab", verbose=False)
        except Exception as exc:
            with suppress(Exception):
                env.close()
            msg = f"Failed to wrap Isaac Lab environment '{self._cfg.id}': {exc}"
            raise EnvironmentCreationError(msg) from exc

        return wrapped_env
