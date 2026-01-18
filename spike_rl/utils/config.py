"""Configuration dataclasses.

These dataclasses can be used with jsonargparse to read YAML configs, perform CLI overrides.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvNormalizationConfig:
    """Environment normalization configuration."""

    enable: bool
    vecnorm_path: str | None
    norm_obs: bool
    norm_reward: bool
    params: dict[str, Any]


@dataclass
class EnvConfig:
    """Environment configuration."""

    id: str
    render_mode: str | None
    n_envs: int | None
    normalization: EnvNormalizationConfig


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
    n_eval_episodes: int = 10


@dataclass
class OptunaParameter:
    """Definition of a single hyperparameter to optimize and its search space."""

    parameter: str  # Full path to the parameter in the config, e.g., "algorithm.learning_rate"
    type: str = "float"  # Parameter type: "float", "int", or "categorical"
    low: float | None = None  # Lower bound (for float/int)
    high: float | None = None  # Upper bound (for float/int)
    log: bool = False  # Whether to use log scale (for float)
    choices: list | None = None  # List of possible values (for categorical)


@dataclass
class OptunaConfig:
    """Hyperparameter optimization configuration for Optuna."""

    n_trials: int  # Number of trials to run
    n_jobs: int
    parameters: list[OptunaParameter] = field(default_factory=list)  # List of parameters


@dataclass
class LoggingConfig:
    """Logging configuration (backends and folder)."""

    folder: str | None = None
    level: str = "INFO"


@dataclass
class Config:
    """Top-level experiment configuration containing mode, seed, and subconfigs."""

    mode: str
    seed: int
    experiment_name: str
    checkpoint_path: str | None
    device: str
    env: EnvConfig
    algorithm: AlgorithmConfig
    policy: PolicyConfig
    training: TrainingConfig
    logging: LoggingConfig
    optuna: OptunaConfig | None
