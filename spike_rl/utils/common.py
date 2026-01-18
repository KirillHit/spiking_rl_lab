"""Common utilities for building environments, models and evaluation in SB3-based experiments."""

from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import mlflow
from jsonargparse import Namespace
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm, BasePolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
    sync_envs_normalization,
)

from spike_rl.utils.logger import logger
from spike_rl.utils.mlflow import sb3_logger

ALGORITHM_REGISTRY: dict[str, type[BaseAlgorithm]] = {
    "ppo": PPO,
    "a2c": A2C,
}

POLICY_REGISTRY: dict[str, str | type[BasePolicy]] = {
    "mlp": "MlpPolicy",
    "cnn": "CnnPolicy",
}


def build_env(cfg: Namespace) -> VecEnv:
    """Create and configure a vectorized Gymnasium environment."""
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
        if cfg.env.normalization.vecnorm_path is not None:
            logger.info(f"Loading VecNormalize from: {cfg.env.normalization.vecnorm_path}")
            env = VecNormalize.load(cfg.env.normalization.vecnorm_path, env)
        else:
            env = VecNormalize(
                env,
                norm_obs=cfg.env.normalization.norm_obs,
                norm_reward=cfg.env.normalization.norm_reward,
                **cfg.env.normalization.params,
            )

    logger.info(f"Created environment {cfg.env.id} with {cfg.env.n_envs} parallel envs.")

    return env


def build_model(cfg: Namespace) -> BaseAlgorithm | None:
    """Create the environment and RL model according to the configuration."""
    env = build_env(cfg)

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


def save_model(cfg: Namespace, model: BaseAlgorithm) -> None:
    """Save the trained model and associated normalization statistics."""
    save_path = Path(cfg.logging.folder) / "model.zip"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model to {save_path}...")
    model.save(save_path)

    is_mlflow = mlflow.active_run() is not None
    if is_mlflow:
        mlflow.log_artifact(save_path)

    if isinstance(model.env, VecNormalize):
        vecnorm_save_path = save_path.parent / "vecnorm.pkl"
        model.env.save(vecnorm_save_path)
        if is_mlflow:
            mlflow.log_artifact(vecnorm_save_path)


def evaluate(
    cfg: Namespace,
    model: BaseAlgorithm | None = None,
    eval_env: VecEnv | None = None,
    *,
    verbose: bool = True,
) -> tuple[float, float] | None:
    """Run policy evaluation and log results."""
    if model is None:
        model = build_model(cfg)
        if model is None:
            return None
        eval_env = model.get_env()
    elif eval_env is None:
        eval_env = build_env(cfg)

    if isinstance(model.env, VecNormalize):
        sync_envs_normalization(model.env, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=cfg.training.n_eval_episodes,
    )

    if verbose:
        logger.info(
            f"Evaluation result: mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}",
        )

    return mean_reward, std_reward


class CustomEvalCallback(BaseCallback):
    """Periodically evaluates the policy during training and logs results to MLflow."""

    def __init__(self, cfg: Namespace, n_eval_step: int, *, verbose: bool = True) -> None:
        """Initialize evaluation callback with a fixed evaluation frequency."""
        super().__init__()
        self.cfg = cfg.clone()
        self.n_eval_step = n_eval_step
        self.verbose = verbose

        self.eval_env = build_env(cfg)
        self.last_time_trigger = 0
        self.best_mean_reward = float("-inf")
        self.best_std_reward = 0.0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_eval_step:
            self.last_time_trigger = self.num_timesteps
            return self._on_event()
        return True

    def _on_event(self) -> bool:
        mean_reward, std_reward = evaluate(
            self.cfg,
            self.model,
            self.eval_env,
            verbose=self.verbose,
        )
        mlflow.log_metrics(
            {"val/mean_reward": mean_reward, "val/std_reward": std_reward},
            step=self.num_timesteps,
        )

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.best_std_reward = std_reward
            save_model(self.cfg, self.model)
            mlflow.log_metrics({"best_mean_reward": mean_reward, "best_std_reward": std_reward})

        return True
