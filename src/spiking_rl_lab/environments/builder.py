"""Environment factory entry point."""

import logging

from skrl.envs.wrappers.torch import Wrapper

from spiking_rl_lab.utils.config import EnvBackend, EnvConfig
from spiking_rl_lab.utils.exception import EnvironmentCreationError

from .gymnasium import build_gymnasium

log = logging.getLogger(__name__)


def build_env(cfg: EnvConfig) -> Wrapper:
    """Build a skrl-wrapped environment according to the configured backend.

    Args:
        cfg (EnvConfig): Environment configuration.

    Returns:
        Wrapper: skrl-compatible environment wrapper.

    Raises:
        EnvironmentCreationError: If the backend is unsupported.

    """
    log.info(
        "Creating environment '%s' with %d parallel envs using backend '%s'",
        cfg.id,
        cfg.n_envs,
        cfg.backend,
    )

    match cfg.backend:
        case EnvBackend.gymnasium:
            return build_gymnasium(cfg)

        case _:
            msg = f"Unsupported environment backend: {cfg.backend}"
            raise EnvironmentCreationError(msg)
