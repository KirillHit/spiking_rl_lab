"""Evaluation entry points for configured experiments."""

from spiking_rl_lab.utils.config import BaseConfig


def evaluate(cfg: BaseConfig) -> None:
    """Run an evaluation session for the configured experiment."""
    raise NotImplementedError
