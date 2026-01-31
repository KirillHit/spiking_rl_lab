"""Environment factory entry point."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from spiking_rl_lab.utils.config import EnvBackend, EnvConfig
from spiking_rl_lab.utils.exception import EnvironmentCreationError

if TYPE_CHECKING:
    from skrl.envs.wrappers.torch import Wrapper

log = logging.getLogger(__name__)


def build_env(cfg: EnvConfig) -> Wrapper:
    """Build a skrl-wrapped environment according to the configured backend.

    Raises:
        EnvironmentCreationError: If the backend is unsupported.

    """
    log.info(
        "Creating environment '%s' with %d parallel envs using backend '%s'...",
        cfg.id,
        cfg.n_envs,
        cfg.backend.value,
    )

    match cfg.backend:
        case EnvBackend.gymnasium:
            from .gymnasium import build_gymnasium

            return build_gymnasium(cfg)

        case _:
            msg = f"Unsupported environment backend: {cfg.backend}"
            raise EnvironmentCreationError(msg)
