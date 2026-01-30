"""Training entry points for configured experiments."""

from spiking_rl_lab.utils.config import BaseConfig


def train(cfg: BaseConfig) -> None:
    """Run a training session using the provided experiment configuration."""
    raise NotImplementedError
