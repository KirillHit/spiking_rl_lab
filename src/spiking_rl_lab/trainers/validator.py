"""Validation on the training environment between policy updates."""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import numpy as np
import torch
import tqdm

from spiking_rl_lab.core.validation import require_positive

if TYPE_CHECKING:
    from collections.abc import Collection

    from skrl.envs.wrappers.torch import Wrapper

    from spiking_rl_lab.agents.base_agent import BaseAgent

type State = torch.Tensor | dict[str, State] | None

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ValidationConfig:
    """Validation frequency and episode count."""

    episodes: int = 20
    min_interval: int = 50000
    disable_progressbar: bool = False

    def __post_init__(self) -> None:
        """Reject invalid validation settings."""
        require_positive("episodes", self.episodes)
        require_positive("min_interval", self.min_interval)


class Validator:
    """Evaluate an agent while advancing its current environment."""

    def __init__(self, cfg: ValidationConfig, env: Wrapper) -> None:
        """Keep the validation settings and training environment."""
        self._cfg = cfg
        self._env = env
        self._best_train_score = -math.inf
        self._best_score = -math.inf
        self._train_improved = False
        self._last_validation_step = 0

    @property
    def score(self) -> float:
        """Return the best validation score."""
        return self._best_score

    def should_validate(
        self,
        *,
        timestep: int,
        timesteps: int,
        train_rewards: Collection[float],
        rollout_complete: bool,
    ) -> bool:
        """Return whether training progress warrants validation."""
        step = timestep + 1
        if train_rewards:
            train_score = float(np.mean(train_rewards))
            if train_score > self._best_train_score:
                self._best_train_score = train_score
                self._train_improved = True

        due = self._train_improved and step - self._last_validation_step >= self._cfg.min_interval
        return (due and rollout_complete) or step == timesteps

    def validate(
        self,
        *,
        agent: BaseAgent,
        observations: torch.Tensor,
        states: State,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, State]:
        """Evaluate, save an improved checkpoint, and return the current environment state."""
        step = timestep + 1
        score, min_score, max_score, observations, states = self._evaluate(
            agent=agent,
            observations=observations,
            states=states,
            ready=False,
            timestep=timestep,
            timesteps=timesteps,
        )

        self._last_validation_step = step
        self._train_improved = False

        if not all(math.isfinite(value) for value in (score, min_score, max_score)):
            log.warning("Skipped validation at step %d: reward is not finite", step)
            return observations, states

        log.info(
            "Validation at step %d: mean reward %.6g, min %.6g, max %.6g",
            step,
            score,
            min_score,
            max_score,
        )
        if mlflow.active_run() is not None:
            mlflow.log_metrics(
                {
                    "Eval / Reward / Total reward_mean": score,
                    "Eval / Reward / Total reward_min": min_score,
                    "Eval / Reward / Total reward_max": max_score,
                },
                step=step,
            )
        if score > self._best_score:
            checkpoint = Path(agent.experiment_dir) / "checkpoints" / "best_agent.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            agent.save(str(checkpoint))
            self._best_score = score
            log.info("Saved best_agent.pt at step %d: %.6g", step, score)
        return observations, states

    def evaluate(self, *, agent: BaseAgent) -> float:
        """Evaluate the configured number of complete episodes from a reset."""
        observations, _ = self._env.reset()
        states = self._env.state()
        score, _, _, _, _ = self._evaluate(
            agent=agent,
            observations=observations,
            states=states,
            ready=True,
            timestep=0,
            timesteps=0,
        )
        return score

    def _evaluate(
        self,
        *,
        agent: BaseAgent,
        observations: torch.Tensor,
        states: State,
        ready: bool,
        timestep: int,
        timesteps: int,
    ) -> tuple[float, float, float, torch.Tensor, State]:
        """Run complete evaluation episodes and retain the resulting environment state."""
        agent.enable_training_mode(enabled=False)
        ready = torch.full(
            (self._env.num_envs,),
            ready,
            dtype=torch.bool,
            device=agent.device,
        )
        returns = torch.zeros(self._env.num_envs, device=agent.device)
        counts = torch.zeros(self._env.num_envs, dtype=torch.int64, device=agent.device)
        quotas = torch.tensor(
            [(self._cfg.episodes + i) // self._env.num_envs for i in range(self._env.num_envs)],
            device=agent.device,
        )
        scores: list[float] = []

        try:
            with (
                torch.no_grad(),
                tqdm.tqdm(
                    total=self._cfg.episodes,
                    desc="Validation",
                    disable=self._cfg.disable_progressbar,
                    file=sys.stdout,
                    leave=False,
                ) as progress,
            ):
                while len(scores) < self._cfg.episodes:
                    actions, _ = agent.act(
                        observations,
                        states,
                        timestep=timestep,
                        timesteps=timesteps,
                    )
                    next_observations, rewards, terminated, truncated, _ = self._env.step(actions)
                    next_states = self._env.state()
                    returns.add_(rewards.reshape(-1) * ready)
                    dones = (terminated | truncated).reshape(-1)
                    agent.reset_state(dones)
                    for index in dones.nonzero().flatten().tolist():
                        if ready[index] and counts[index] < quotas[index]:
                            scores.append(returns[index].item())
                            counts[index] += 1
                            progress.update()
                        ready[index] = True
                        returns[index] = 0

                    if self._env.num_envs == 1 and dones.any():
                        observations, _ = self._env.reset()
                        states = self._env.state()
                    else:
                        observations, states = next_observations, next_states

                agent.reset_episode_tracking(dones)
                return (
                    sum(scores) / len(scores),
                    min(scores),
                    max(scores),
                    observations,
                    states,
                )
        finally:
            agent.enable_training_mode(enabled=True)
