"""Evaluation entry points for configured experiments."""

from spiking_rl_lab.utils.config import BaseConfig


def evaluate(cfg: BaseConfig):
    pass


""" def evaluate(
    cfg: BaseConfig,
    model: BaseAlgorithm | None = None,
    eval_env: VecEnv | None = None,
    *,
    verbose: bool = True,
) -> tuple[float, float] | None:
    
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

    return mean_reward, std_reward """
