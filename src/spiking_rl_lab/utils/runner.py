"""Experiment runner coordinating environment, model, agent, and trainer lifecycle."""

import datetime
import logging

import mlflow
from flatten_dict import flatten
from skrl.trainers.torch import ParallelTrainer, SequentialTrainer, Trainer
from skrl.utils import set_seed

from spiking_rl_lab.agents import build_agent
from spiking_rl_lab.envs import build_env
from spiking_rl_lab.models import build_models
from spiking_rl_lab.utils.config import BaseConfig, RunnerMode
from spiking_rl_lab.utils.exception import SpikingRLLabError, TrainerCreationError
from spiking_rl_lab.utils.mlflow import (
    config_to_dict,
    log_git_diff_artifact,
    log_model_metadata,
    setup_mlflow,
)

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

    def train(self, cfg: BaseConfig) -> None:
        """Run the training loop."""
        with mlflow.start_run(run_name=self._generate_run_name(cfg)) as run:
            log_git_diff_artifact(cfg.runner.output_dir)
            cfg_dict = config_to_dict(cfg)
            cfg_dict.pop("optuna", None)
            mlflow.log_params(flatten(cfg_dict, "path"))
            mlflow.log_artifact(str(cfg.runner.output_dir / ".hydra" / "config.yaml"))

            try:
                trainer = self._generate_trainer(cfg)

                log.info("Starting training...")
                trainer.train()
            except SpikingRLLabError:
                log.exception("Training failed!")

            log_model_metadata(run, cfg.runner.output_dir)
            mlflow.log_artifact(str(cfg.runner.output_dir / "run.log"))
            mlflow.log_artifact(str(cfg.runner.output_dir / "checkpoints" / "best_agent.pt"))

    def evaluate(self, cfg: BaseConfig) -> None:
        """Run the evaluation loop."""
        trainer = self._generate_trainer(cfg)
        trainer.eval()

    def optimize(self, cfg: BaseConfig) -> None:
        """Run hyperparameter optimization."""
        raise NotImplementedError

    def _generate_trainer(self, cfg: BaseConfig) -> Trainer:
        """Instantiate environment, models, agent, and trainer.

        Returns:
            Trainer: Configured skrl trainer instance.

        Raises:
            TrainerCreationError: If trainer initialization fails.

        """
        env = build_env(cfg.env)
        models = build_models(cfg.models, env)
        agent = build_agent(cfg.agent, env, models)
        agent.experiment_dir = cfg.runner.output_dir

        try:
            trainer_class = ParallelTrainer if cfg.trainer.use_parallel else SequentialTrainer
            return trainer_class(env=env, agents=agent, cfg=cfg.trainer.params)
        except Exception as exc:
            msg = "Failed to create trainer"
            raise TrainerCreationError(msg) from exc

    def _generate_run_name(self, cfg: BaseConfig) -> str:
        """Generate a deterministic run name based on the experiment configuration."""
        ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
        return f"{ts}_{cfg.env.id}_{cfg.agent.name}"
