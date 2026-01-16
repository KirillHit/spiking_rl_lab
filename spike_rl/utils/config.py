"""Configuration dataclasses.

These dataclasses can be used with jsonargparse to read YAML configs, perform CLI overrides.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class EnvConfig:
    """Environment configuration."""

    id: str
    render_mode: str = ""
    n_envs: int | None = None


@dataclass
class AlgorithmConfig:
    """RL algorithm configuration."""

    name: str
    params: dict[str, Any]


@dataclass
class PolicyConfig:
    """Policy network configuration."""

    name: str
    params: dict[str, Any]


@dataclass
class TrainingConfig:
    """Training settings."""

    total_timesteps: int


@dataclass
class OptunaParameter:
    """Definition of a single hyperparameter to optimize and its search space."""

    parameter: str


@dataclass
class OptunaConfig:
    """Hyperparameter optimization configuration for Optuna."""

    n_trials: int
    parameters: list[OptunaParameter]


@dataclass
class LoggingConfig:
    """Logging configuration (backends and folder)."""

    loggers: list[str]
    folder: str | None = None
    level: str = "INFO"
    cmd_log_path: str | None = None


@dataclass
class Config:
    """Top-level experiment configuration containing mode, seed, and subconfigs."""

    mode: str
    seed: int
    env: EnvConfig
    algorithm: AlgorithmConfig
    policy: PolicyConfig
    training: TrainingConfig
    logging: LoggingConfig
    run_name: str | None = None
    optuna: OptunaConfig | None = None
