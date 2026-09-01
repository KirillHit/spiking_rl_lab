"""Utilities for applying Optuna search parameters to application configs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

    from spiking_rl_lab.app.config import BaseConfig, OptunaParameter


def suggest_value(trial: optuna.Trial, parameter: OptunaParameter) -> object:
    """Sample a value from one configured search space."""
    match parameter.type:
        case "float":
            if parameter.low is None or parameter.high is None:
                msg = f"Float parameter '{parameter.parameter}' requires low and high"
                raise ValueError(msg)
            return trial.suggest_float(
                parameter.parameter,
                parameter.low,
                parameter.high,
                log=parameter.log,
            )
        case "int":
            if parameter.low is None or parameter.high is None:
                msg = f"Int parameter '{parameter.parameter}' requires low and high"
                raise ValueError(msg)
            return trial.suggest_int(
                parameter.parameter,
                int(parameter.low),
                int(parameter.high),
            )
        case "categorical":
            if not parameter.choices:
                msg = f"Categorical parameter '{parameter.parameter}' requires choices"
                raise ValueError(msg)
            return trial.suggest_categorical(parameter.parameter, parameter.choices)
        case _:
            msg = f"Unsupported Optuna parameter type: {parameter.type}"
            raise ValueError(msg)


def set_config_value(cfg: BaseConfig, parameter: str, value: object) -> None:
    """Set a nested config value addressed by a dotted path."""
    path = parameter.split(".")
    target: object = cfg
    for key in path[:-1]:
        if isinstance(target, dict):
            target = target[key]
        elif isinstance(target, list):
            target = target[int(key)]
        else:
            target = getattr(target, key)

    key = path[-1]
    if isinstance(target, dict):
        if key not in target:
            msg = f"Unknown Optuna parameter: {parameter}"
            raise KeyError(msg)
        target[key] = value
    elif isinstance(target, list):
        target[int(key)] = value
    else:
        if not hasattr(target, key):
            msg = f"Unknown Optuna parameter: {parameter}"
            raise AttributeError(msg)
        setattr(target, key, value)
