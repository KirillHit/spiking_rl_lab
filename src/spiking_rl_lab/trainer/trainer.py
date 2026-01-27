"""Trainer module that runs reinforcement learning experiments."""

import logging
from pathlib import Path

import mlflow
import optuna
import torch
from jsonargparse import Namespace

from spiking_rl_lab.trainer.common import CustomEvalCallback, build_model, evaluate
from spiking_rl_lab.utils.config import BaseConfig

log = logging.getLogger(__name__)


class Trainer:
    """Trainer class for running RL experiments with a given configuration."""

    def run(self, cfg: BaseConfig) -> None:
        """Run the training loop using the provided configuration."""
        log.info("Starting Trainer in mode: %s", cfg.trainer.mode)
        log.info("Experiment name: %s", cfg.trainer.experiment_name)

        torch.manual_seed(cfg.trainer.seed)

        match cfg.trainer.mode:
            case "train":
                self.train(cfg)
            case "evaluate":
                evaluate(cfg)
            case "optimize":
                self.optimize(cfg)
            case _:
                log.error("Unsupported mode: %s!", cfg.trainer.mode)

        log.info("Trainer finished.")

    def train(self, cfg: Namespace) -> tuple[BaseAlgorithm, float, float] | None:
        """Run training procedure for the given experiment configuration."""
        log.info("Preparing for training...")
        model = build_model(cfg)
        if model is None:
            return None

        if isinstance(model.env, VecNormalize):
            model.env.training = True

        with mlflow.start_run(run_name=self.gen_run_name(cfg)) as run:
            cfg_dict = cfg.as_dict()
            cfg_dict.pop("optuna", None)
            cfg_dict.pop("logging", None)

            log_git_diff_artifact(Path(cfg.logging.folder))

            eval_callback = CustomEvalCallback(cfg, cfg.training.n_eval_step)

            log.info(f"Starting training for {cfg.training.total_timesteps} timesteps...")
            model.learn(
                total_timesteps=cfg.training.total_timesteps,
                progress_bar=True,
                callback=eval_callback,
            )

            log_model_metadata(run, Path(cfg.logging.folder))

        return model, eval_callback.best_mean_reward, eval_callback.best_std_reward

    def optimize(self, cfg: Namespace) -> None:
        """Run hyperparameter optimization using Optuna."""
        log.info("Starting hyperparameter optimization...")

        def objective(trial: optuna.Trial) -> float:
            """Objective function for Optuna trials."""
            trial_cfg = cfg.clone()

            trial_cfg.env.render_mode = None

            for param in cfg.optuna.parameters:
                keys = param.parameter.split(".")

                target = trial_cfg
                for k in keys[:-1]:
                    target = getattr(target, k)

                if param.type == "float":
                    value = trial.suggest_float(
                        param.parameter,
                        param.low,
                        param.high,
                        log=param.log,
                    )
                elif param.type == "int":
                    value = trial.suggest_int(
                        param.parameter,
                        int(param.low),
                        int(param.high),
                        log=param.log,
                    )
                elif param.type == "categorical":
                    value = trial.suggest_categorical(param.parameter, param.choices)
                else:
                    msg = f"Unsupported Optuna parameter type: {param.type}"
                    raise ValueError(msg)

                if keys[-1] not in target:
                    msg = f"Config has no parameter '{param.parameter}'"
                    raise AttributeError(msg)
                target[keys[-1]] = value

            result = self.train(trial_cfg)
            if result is None:
                return float("-inf")

            return result[1]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=cfg.optuna.n_trials, n_jobs=cfg.optuna.n_jobs)

        log.info(f"Best trial: {study.best_trial.number} -> {study.best_value:.2f}")
        log.info(f"Best parameters: {study.best_params}")
