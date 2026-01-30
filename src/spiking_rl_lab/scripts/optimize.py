"""Hyperparameter optimization entry points for configured experiments."""

from spiking_rl_lab.utils.config import BaseConfig


def optimize(cfg: BaseConfig) -> None:
    """Run hyperparameter optimization for the configured experiment."""
    raise NotImplementedError
