"""Trainer module that runs reinforcement learning experiments."""

import datetime
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import mlflow
import torch
from flatten_dict import flatten
from jsonargparse import Namespace
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm, BasePolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

from spike_rl.utils.logger import console_logger, logger
from spike_rl.utils.mlflow import (
    log_git_diff_artifact,
    log_model_metadata,
    sb3_logger,
    setup_mlflow,
)

ALGORITHM_REGISTRY: dict[str, type[BaseAlgorithm]] = {
    "ppo": PPO,
    "a2c": A2C,
}

POLICY_REGISTRY: dict[str, str | type[BasePolicy]] = {
    "mlp": "MlpPolicy",
    "cnn": "CnnPolicy",
}


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

        setup_mlflow(Path("experiments"), cfg.experiment_name)

        torch.manual_seed(cfg.seed)

        match cfg.mode:
            case "train":
                self.train(cfg)
            case "evaluate":
                self.evaluate(cfg)
            case "optimize":
                pass
            case _:
                logger.error("Unsupported mode: cfg.mode!")

        logger.info("Trainer finished.")

    def train(self, cfg: Namespace) -> BaseAlgorithm | None:
        """Run training procedure for the given experiment configuration."""
        logger.info("Preparing for training...")
        model = self.build_model(cfg)
        if model is None:
            return None

        env = model.get_env()
        if isinstance(env, VecNormalize):
            env.training = True
            env.norm_reward = False

        with mlflow.start_run(run_name=self.gen_run_name(cfg)) as run:
            cfg_dict = cfg.as_dict()
            cfg_dict.pop("optuna", None)
            cfg_dict.pop("logging", None)
            mlflow.log_params(flatten(cfg_dict, "path"))

            log_git_diff_artifact(Path(cfg.logging.folder))

            logger.info(f"Starting training for {cfg.training.total_timesteps} timesteps...")
            model.learn(
                total_timesteps=cfg.training.total_timesteps,
                progress_bar=True,
            )

            save_path = Path(cfg.logging.folder) / "model.zip"
            logger.info(f"Saving model to {save_path}...")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(save_path)
            mlflow.log_artifact(save_path)

            mean_reward, std_reward = self.evaluate(cfg, model)
            mlflow.log_metrics({"mean_reward": mean_reward, "std_reward": std_reward})

            log_model_metadata(run, Path(cfg.logging.folder))

        return model

    def evaluate(self, cfg: Namespace, model: BaseAlgorithm | None = None) -> tuple[float, float]:
        """Run policy evaluation and log results."""
        logger.info("Preparing for validating...")
        if model is None:
            model = self.build_model(cfg)
            if model is None:
                return None, None

        env = model.get_env()
        if isinstance(env, VecNormalize):
            env.training = False
            env.norm_reward = cfg.env.normalization.norm_reward

        mean_reward, std_reward = evaluate_policy(
            model,
            model.get_env(),
            n_eval_episodes=cfg.training.n_eval_episodes,
        )

        logger.info(
            f"Validation result: mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}",
        )

        return mean_reward, std_reward

    def build_model(self, cfg: Namespace) -> BaseAlgorithm | None:
        """Create the environment and RL model according to the configuration."""
        logger.info("Creating environments...")

        def make_env() -> Callable[[], gym.Env]:
            """Create an environment factory for vectorized environments."""

            def _init() -> gym.Env:
                """Initialize and return a single Gymnasium environment."""
                render_mode = None if cfg.env.n_envs > 1 else cfg.env.render_mode
                return gym.make(cfg.env.id, render_mode=render_mode)

            return _init

        if cfg.env.n_envs > 1:
            env = SubprocVecEnv([make_env() for _ in range(cfg.env.n_envs)])
        else:
            env = DummyVecEnv([make_env()])
        env.seed(cfg.seed)

        if cfg.env.normalization.enable:
            env = VecNormalize(
                env,
                norm_obs=cfg.env.normalization.norm_obs,
                norm_reward=cfg.env.normalization.norm_reward,
                **cfg.env.normalization.params,
            )

        logger.info(f"Created environment {cfg.env.id} with {cfg.env.n_envs} parallel envs.")

        logger.info("Building model...")

        try:
            algo_cls = ALGORITHM_REGISTRY[cfg.algorithm.name]
        except KeyError:
            logger.error(f"Unsupported algorithm: {cfg.algorithm.name}.")
            return None

        try:
            policy = POLICY_REGISTRY[cfg.policy.name]
        except KeyError:
            logger.error(f"Unsupported policy: {cfg.policy.name}.")
            return None

        if cfg.checkpoint_path is not None:
            logger.info(f"Loading model from checkpoint: {cfg.checkpoint_path}")
            model = algo_cls.load(
                cfg.checkpoint_path,
                env=env,
                device=cfg.device,
            )
        else:
            logger.info("Creating new model instance")
            model = algo_cls(
                policy=policy,
                env=env,
                device=cfg.device,
                policy_kwargs=cfg.policy.params,
                verbose=1,
                **cfg.algorithm.params,
            )

        model.set_logger(sb3_logger)

        return model

    def gen_run_name(self, cfg: Namespace) -> str:
        """Generate a deterministic run name based on the experiment configuration."""
        ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
        return f"{cfg.algorithm.name}_{cfg.policy.name}_{cfg.env.id}_{ts}"
