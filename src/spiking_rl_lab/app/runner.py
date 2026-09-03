"""Experiment runner coordinating environment, model, agent, and trainer lifecycle."""

from __future__ import annotations

import datetime
import logging
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

import mlflow
import optuna
from flatten_dict import flatten
from skrl.trainers.torch import ParallelTrainer, SequentialTrainer, Trainer
from skrl.utils import set_seed

from spiking_rl_lab.agents.builder import build_agent
from spiking_rl_lab.app.config import BaseConfig, RunnerMode
from spiking_rl_lab.app.optuna import set_config_value, suggest_value
from spiking_rl_lab.app.tracking import (
    config_to_dict,
    log_artifact_if_exists,
    log_git_diff_artifact,
    log_hardware_info,
    log_model_metadata,
    setup_mlflow,
)
from spiking_rl_lab.core.exception import SpikingRLLabError, TrainerCreationError
from spiking_rl_lab.envs.builder import build_env

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from spiking_rl_lab.agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class Runner:
    """High-level entry point for running experiments."""

    def run(self, cfg: BaseConfig) -> None:
        """Run an experiment according to the provided configuration."""
        log.info("Starting SpikingRL Lab in mode '%s'...", cfg.runner.mode.value)
        log.info("Experiment name: %s", cfg.runner.experiment_name)

        set_seed(cfg.runner.seed, deterministic=cfg.runner.deterministic)

        setup_mlflow(
            cfg.runner.dagshub_repo_owner,
            cfg.runner.dagshub_repo_name,
            cfg.runner.experiment_name,
        )

        match cfg.runner.mode:
            case RunnerMode.train:
                self.train(cfg)
            case RunnerMode.evaluate:
                self.evaluate(cfg)
            case RunnerMode.optimize:
                self.optimize(cfg)
            case _:
                msg = f"Unknown running mode: {cfg.runner.mode}"
                raise ValueError(msg)

        log.info("SpikingRL Lab finished.")

    def train(self, cfg: BaseConfig) -> float:
        """Run the training loop."""
        with mlflow.start_run(run_name=self._generate_run_name(cfg)) as run:
            try:
                return self._train(cfg)
            finally:
                log_model_metadata(run, cfg.runner.output_dir)
                log_artifact_if_exists(cfg.runner.output_dir / "run.log")

    def evaluate(self, cfg: BaseConfig, checkpoint_path: Path | None = None) -> float:
        """Run the evaluation loop."""
        eval_cfg = self._prepare_eval_config(cfg, checkpoint_path)

        with self._trainer_context(eval_cfg) as trainer:
            log.info("Starting evaluation...")
            trainer.eval()
            score = trainer.agents.last_tracking_metrics.get("Eval / Reward / Total reward_mean")

        if score is None:
            msg = "Evaluation finished without a tracked total reward metric"
            raise SpikingRLLabError(msg)

        log.info("Evaluation mean reward: %.6g", score)
        return score

    def optimize(self, cfg: BaseConfig) -> None:
        """Run hyperparameter optimization."""
        if not cfg.optuna.parameters:
            msg = "Optimize mode requires at least one Optuna parameter"
            raise ValueError(msg)

        log.info(
            "Starting optimization: direction=%s, trials=%d, jobs=%d",
            cfg.optuna.direction,
            cfg.optuna.n_trials,
            cfg.optuna.n_jobs,
        )
        with mlflow.start_run(run_name=f"{self._generate_run_name(cfg)}_optimize") as run:
            study = optuna.create_study(direction=cfg.optuna.direction)
            study.optimize(
                lambda trial: self._objective(trial, cfg, run.info.run_id),
                n_trials=cfg.optuna.n_trials,
                n_jobs=cfg.optuna.n_jobs,
                timeout=cfg.optuna.timeout,
                catch=(Exception,),
            )
            best_run_id = study.best_trial.user_attrs["mlflow_run_id"]
            client = mlflow.tracking.MlflowClient()
            client.set_tag(best_run_id, "optuna.best_trial", "true")
            mlflow.log_metric("optuna.best_value", study.best_value)
            mlflow.log_params(
                {
                    "optuna.direction": cfg.optuna.direction,
                    "optuna.n_trials": cfg.optuna.n_trials,
                    "optuna.n_jobs": cfg.optuna.n_jobs,
                    **{
                        f"optuna.best_params.{key}": value
                        for key, value in study.best_params.items()
                    },
                },
            )
            log_artifact_if_exists(cfg.runner.output_dir / "run.log")

        log.info("Best trial: %d", study.best_trial.number)
        log.info("Best value: %.6g", study.best_value)
        log.info("Best parameters: %s", study.best_params)

    def _objective(
        self,
        trial: optuna.Trial,
        cfg: BaseConfig,
        parent_run_id: str,
    ) -> float:
        """Train and evaluate one sampled hyperparameter configuration."""
        trial_cfg = deepcopy(cfg)
        trial_cfg = replace(
            trial_cfg,
            runner=replace(
                trial_cfg.runner,
                output_dir=cfg.runner.output_dir / "trials" / f"trial_{trial.number:04d}",
            ),
        )
        for parameter in trial_cfg.optuna.parameters:
            set_config_value(
                trial_cfg,
                parameter.parameter,
                suggest_value(trial, parameter),
            )

        try:
            with mlflow.start_run(
                run_name=self._generate_run_name(trial_cfg),
                nested=True,
                parent_run_id=parent_run_id,
            ) as run:
                trial.set_user_attr("mlflow_run_id", run.info.run_id)
                try:
                    score = self._train(trial_cfg)
                finally:
                    log_model_metadata(run, trial_cfg.runner.output_dir)
        except Exception as exc:
            trial.set_user_attr("error", repr(exc))
            log.exception("Trial %d failed", trial.number)
            raise

        return score

    def _train(self, cfg: BaseConfig) -> float:
        """Run training inside an active MLflow run."""
        log_git_diff_artifact(cfg.runner.output_dir)
        cfg_dict = config_to_dict(cfg)
        cfg_dict.pop("optuna", None)
        mlflow.log_params(flatten(cfg_dict, "path"))
        log_artifact_if_exists(cfg.runner.output_dir / ".hydra" / "config.yaml")
        log_hardware_info(cfg.runner.output_dir)

        try:
            with self._trainer_context(cfg) as trainer:
                log.info("Starting training...")
                trainer.train()

            best_checkpoint = cfg.runner.output_dir / "checkpoints" / "best_agent.pt"
            if not best_checkpoint.exists():
                msg = f"Best checkpoint was not found: {best_checkpoint}"
                raise FileNotFoundError(msg)

            return self.evaluate(cfg, checkpoint_path=best_checkpoint)
        except SpikingRLLabError:
            log.exception("Training failed!")
            raise
        finally:
            log_artifact_if_exists(cfg.runner.output_dir / "checkpoints" / "best_agent.pt")

    @contextmanager
    def _trainer_context(self, cfg: BaseConfig) -> Generator[Trainer, None, None]:
        """Yield a configured trainer and close its environment afterwards.

        Returns:
            Trainer: Configured skrl trainer instance.

        Raises:
            TrainerCreationError: If trainer initialization fails.

        """
        env = build_env(cfg.env)
        try:
            agent = build_agent(cfg.agent, env=env)
            agent.experiment_dir = cfg.runner.output_dir
            self._load_checkpoint_if_configured(
                agent=agent,
                checkpoint_path=cfg.runner.checkpoint_path,
            )

            try:
                trainer_class = ParallelTrainer if cfg.trainer.use_parallel else SequentialTrainer
                trainer = trainer_class(env=env, agents=agent, cfg=cfg.trainer.params)
            except Exception as exc:
                msg = "Failed to create trainer"
                raise TrainerCreationError(msg) from exc

            yield trainer
        finally:
            log.info("Closing environment...")
            env.close()

    def _load_checkpoint_if_configured(
        self,
        *,
        agent: BaseAgent,
        checkpoint_path: Path | None,
    ) -> None:
        """Load agent weights from a checkpoint path if configured."""
        if checkpoint_path is None:
            return

        if not checkpoint_path.exists():
            msg = f"Checkpoint file does not exist: {checkpoint_path}"
            raise TrainerCreationError(msg)

        log.info("Loading agent checkpoint from '%s'...", checkpoint_path)
        try:
            agent.load(str(checkpoint_path))
        except Exception as exc:
            msg = f"Failed to load checkpoint: {checkpoint_path}"
            raise TrainerCreationError(msg) from exc

    def _generate_run_name(self, cfg: BaseConfig) -> str:
        """Generate a deterministic run name based on the experiment configuration."""
        ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
        env_id = cfg.env.params.get("id", cfg.env.name)
        return f"{ts}_{env_id}_{cfg.agent.name}"

    def _prepare_eval_config(
        self,
        cfg: BaseConfig,
        checkpoint_path: Path | None = None,
    ) -> BaseConfig:
        """Create a config for evaluation without mutating the original one."""
        trainer_params = {**cfg.trainer.params, "timesteps": cfg.trainer.eval_timesteps}
        agent_params = {**cfg.agent.params}
        experiment = dict(agent_params.get("experiment", {}))
        experiment["write_interval"] = cfg.trainer.eval_timesteps
        agent_params["experiment"] = experiment

        return replace(
            cfg,
            agent=replace(cfg.agent, params=agent_params),
            runner=replace(
                cfg.runner,
                checkpoint_path=checkpoint_path
                if checkpoint_path is not None
                else cfg.runner.checkpoint_path,
            ),
            trainer=replace(cfg.trainer, params=trainer_params),
        )
