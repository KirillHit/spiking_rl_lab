"""Trainer module that runs reinforcement learning experiments."""

import datetime
from pathlib import Path

import mlflow
import optuna
import torch
from flatten_dict import flatten
from jsonargparse import Namespace
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize

from spike_rl.utils.common import CustomEvalCallback, build_model, evaluate
from spike_rl.utils.logger import console_logger, logger
from spike_rl.utils.mlflow import (
    log_git_diff_artifact,
    log_model_metadata,
    setup_mlflow,
)


class Trainer:
    """Trainer class for running RL experiments with a given configuration."""

    def run(self, cfg: Namespace) -> None:
        """Run the training loop using the provided configuration.

        Args:
            cfg (Namespace): Configuration namespace.

        """
        console_logger.configure(cfg.logging.level, str(Path(cfg.logging.folder) / "stdout.log"))

        logger.info(f"Starting Trainer in mode {cfg.mode}...")
        logger.info(f"Experiment name: {cfg.experiment_name}")
        logger.info(f"Parameters: {cfg.as_dict()}")

        setup_mlflow(Path("experiments"), cfg.experiment_name)

        torch.manual_seed(cfg.seed)

        match cfg.mode:
            case "train":
                self.train(cfg)
            case "evaluate":
                evaluate(cfg)
            case "optimize":
                self.optimize(cfg)
            case _:
                logger.error("Unsupported mode: cfg.mode!")

        logger.info("Trainer finished.")

    def train(self, cfg: Namespace) -> tuple[BaseAlgorithm, float, float] | None:
        """Run training procedure for the given experiment configuration."""
        logger.info("Preparing for training...")
        model = build_model(cfg)
        if model is None:
            return None

        if isinstance(model.env, VecNormalize):
            model.env.training = True

        with mlflow.start_run(run_name=self.gen_run_name(cfg)) as run:
            cfg_dict = cfg.as_dict()
            cfg_dict.pop("optuna", None)
            cfg_dict.pop("logging", None)
            mlflow.log_params(flatten(cfg_dict, "path"))

            log_git_diff_artifact(Path(cfg.logging.folder))

            eval_callback = CustomEvalCallback(cfg, cfg.training.n_eval_step)

            logger.info(f"Starting training for {cfg.training.total_timesteps} timesteps...")
            model.learn(
                total_timesteps=cfg.training.total_timesteps,
                progress_bar=True,
                callback=eval_callback,
            )

            log_model_metadata(run, Path(cfg.logging.folder))

        return model, eval_callback.best_mean_reward, eval_callback.best_std_reward

    def optimize(self, cfg: Namespace) -> None:
        """Run hyperparameter optimization using Optuna."""
        logger.info("Starting hyperparameter optimization...")

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

        logger.info(f"Best trial: {study.best_trial.number} -> {study.best_value:.2f}")
        logger.info(f"Best parameters: {study.best_params}")

    def gen_run_name(self, cfg: Namespace) -> str:
        """Generate a deterministic run name based on the experiment configuration."""
        ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
        return f"{cfg.algorithm.name}_{cfg.policy.name}_{cfg.env.id}_{ts}"
