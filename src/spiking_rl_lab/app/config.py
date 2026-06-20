"""Utilities for Hydra configuration registration."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from spiking_rl_lab.agents.builder import AgentConfig
from spiking_rl_lab.envs.builder import EnvConfig
from spiking_rl_lab.models.builder import ModelConfig
from spiking_rl_lab.networks.builder import NetworkConfig


class RunnerMode(StrEnum):
    """Supported runner modes."""

    train = "train"
    evaluate = "evaluate"
    optimize = "optimize"


@dataclass(slots=True)
class RunnerConfig:
    """Training settings."""

    mode: RunnerMode = MISSING
    seed: int = 42
    deterministic: bool = False
    experiment_name: str = MISSING
    output_dir: Path = MISSING
    checkpoint_path: Path | None = None
    dagshub_repo_owner: str = MISSING
    dagshub_repo_name: str = MISSING


@dataclass(slots=True)
class TrainerConfig:
    """Training settings."""

    use_parallel: bool = True
    eval_timesteps: int = 10000
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptunaParameter:
    """Definition of a single hyperparameter to optimize and its search space."""

    parameter: str = MISSING  # Full path to the parameter in the config
    type: str = "float"  # Parameter type: "float", "int", or "categorical"
    low: float | None = None  # Lower bound (for float/int)
    high: float | None = None  # Upper bound (for float/int)
    log: bool = False  # Whether to use log scale (for float)
    choices: list | None = None  # List of possible values (for categorical)


@dataclass(slots=True)
class OptunaConfig:
    """Hyperparameter optimization configuration for Optuna."""

    n_trials: int = MISSING  # Number of trials to run
    n_jobs: int = MISSING
    parameters: list[OptunaParameter] = field(default_factory=list)  # List of parameters


@dataclass(slots=True)
class BaseConfig:
    """Top-level experiment configuration containing mode, seed, and subconfigs."""

    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    networks: dict[str, NetworkConfig] = field(default_factory=dict)
    models: list[ModelConfig] = field(default_factory=list)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)


def register_configs() -> None:
    """Register the root config schema in Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=BaseConfig)
