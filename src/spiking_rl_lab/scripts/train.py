"""Training entry points for configured experiments."""

from spiking_rl_lab.utils.config import BaseConfig


def train(cfg: BaseConfig) -> None:
    """Train an agent end-to-end with the provided configuration."""
    raise NotImplementedError
