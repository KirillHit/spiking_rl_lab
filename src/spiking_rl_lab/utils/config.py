"""Utilities for Hydra configuration registration."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class EnvBackend(str, Enum):
    """Supported environment backends."""

    gymnasium = "gymnasium"


@dataclass
class EnvConfig:
    """Environment configuration."""

    backend: EnvBackend = MISSING
    id: str = MISSING
    render: bool = False
    n_envs: int = 1


@dataclass
class AgentConfig:
    """RL algorithm configuration."""

    name: str = MISSING
    device: str = "cpu"
    memory_size: int = 1024
    params: dict[str, Any] = field(default_factory=dict)


class ModelRole(str, Enum):
    """Role of a model within an agent's architecture."""

    policy = "policy"
    value = "value"


@dataclass
class ModelConfig:
    """Configuration for a single model instance."""

    name: str = MISSING
    role: ModelRole = MISSING
    device: str = "cpu"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelsConfig:
    """Collection of model configurations for an experiment."""

    models: list[ModelConfig] = field(default_factory=list)


class RunnerMode(str, Enum):
    """Supported runner modes."""

    train = "train"
    evaluate = "evaluate"
    optimize = "optimize"


@dataclass
class RunnerConfig:
    """Training settings."""

    mode: RunnerMode = MISSING
    seed: int = 42
    deterministic: bool = False
    experiment_name: str = MISSING
    output_dir: Path = MISSING
    mlflow_dir: Path = "experiments"


@dataclass
class TrainerConfig:
    """Training settings."""

    use_parallel: bool = True
    params: dict[str, Any] = field(default_factory=dict)


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
    agent: AgentConfig = field(default_factory=AgentConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)


def register_configs() -> None:
    """Register the root config schema in Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=BaseConfig)
