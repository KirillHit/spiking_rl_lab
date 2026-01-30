"""Utilities for Hydra configuration registration."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class EnvBackend(str, Enum):
    """..."""

    gymnasium = "gymnasium"


@dataclass
class EnvConfig:
    """Environment configuration."""

    backend: EnvBackend = MISSING
    id: str = MISSING
    render: bool = False
    n_envs: int = 1


@dataclass
class AlgorithmConfig:
    """RL algorithm configuration."""

    name: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyConfig:
    """Policy network configuration."""

    name: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerConfig:
    """Training settings."""

    mode: str = MISSING
    seed: int = 42
    experiment_name: str = MISSING
    checkpoint_path: str | None = None
    device: str = "cpu"
    output_dir: Path = MISSING
    deterministic: bool = False

    total_timesteps: int = MISSING
    n_eval_episodes: int = MISSING
    n_eval_step: int = MISSING


@dataclass
class OptunaParameter:
    """Definition of a single hyperparameter to optimize and its search space."""

    parameter: str = MISSING  # Full path to the parameter in the config
    type: str = "float"  # Parameter type: "float", "int", or "categorical"
    low: float | None = None  # Lower bound (for float/int)
    high: float | None = None  # Upper bound (for float/int)
    log: bool = False  # Whether to use log scale (for float)
    choices: list | None = None  # List of possible values (for categorical)


@dataclass
class OptunaConfig:
    """Hyperparameter optimization configuration for Optuna."""

    n_trials: int = MISSING  # Number of trials to run
    n_jobs: int = MISSING
    parameters: list[OptunaParameter] = field(default_factory=list)  # List of parameters


@dataclass
class BaseConfig:
    """Top-level experiment configuration containing mode, seed, and subconfigs."""

    env: EnvConfig = field(default_factory=EnvConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)


def register_configs() -> None:
    """Register the root config schema in Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=BaseConfig)
