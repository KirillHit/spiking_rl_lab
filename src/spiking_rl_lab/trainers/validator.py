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


@dataclass(slots=True, frozen=True)
class ValidationResult:
    """Store and report the result of one validation run."""

    scores: np.ndarray
    metrics: dict[str, tuple[object, ...]]

    def validate(self) -> bool:
        """Return whether the validation result contains finite rewards."""
        return self.scores.size > 0 and bool(np.isfinite(self.scores).all())

    def log_console(self, step: int | None = None) -> None:
        """Print a readable validation report through the application logger."""
        if not self.validate():
            name = "Evaluation" if step is None else f"Validation at step {step}"
            log.warning("%s produced an invalid result", name)
            return
        lines = [
            "Evaluation:" if step is None else f"Validation at step {step}:",
            (
                f"  reward: mean={self.scores.mean():.6g}, min={self.scores.min():.6g}, "
                f"max={self.scores.max():.6g}"
            ),
        ]
        for key, values in self.metrics.items():
            lines.append(f"  {key}: mean={np.asarray(values).mean():.6g}")
        log.info("\n".join(lines))

    def log_mlflow(self, step: int | None = None) -> None:
        """Log validation summaries to the active MLflow run."""
        if not self.validate() or mlflow.active_run() is None:
            return

        metrics = {
            "Eval / Reward / Total reward_mean": float(self.scores.mean()),
            "Eval / Reward / Total reward_min": float(self.scores.min()),
            "Eval / Reward / Total reward_max": float(self.scores.max()),
        }
        for key, values in self.metrics.items():
            metrics[f"Eval / Metrics / {key}"] = float(np.asarray(values).mean())
        mlflow.log_metrics(metrics, step=step)


@dataclass(slots=True)
class ValidationConfig:
    """Validation frequency and episode count."""

    episodes: int = 20
    min_interval: int = 50000
    max_interval: int = 100000
    disable_progressbar: bool = False

    def __post_init__(self) -> None:
        """Reject invalid validation settings."""
        require_positive("episodes", self.episodes)
        require_positive("min_interval", self.min_interval)
        require_positive("max_interval", self.max_interval)
        if self.max_interval < self.min_interval:
            msg = (
                f"max_interval must be >= min_interval "
                f"(got {self.max_interval} < {self.min_interval})"
            )
            raise ValueError(msg)


class Validator:
    """Evaluate an agent while advancing its current environment."""

    def __init__(self, cfg: ValidationConfig, env: Wrapper) -> None:
        """Keep the validation settings and training environment."""
        self._cfg = cfg
        self._env = env
        self.reset_progress_reference()
        self._last_validation_step = 0

    @property
    def score(self) -> float:
        """Return the best validation score."""
        return self._best_score

    def reset_progress_reference(self) -> None:
        """Reset reward baselines after the environment difficulty changes."""
        self._best_train_score = -math.inf
        self._best_score = -math.inf
        self._train_improved = False

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

        elapsed = step - self._last_validation_step
        due = elapsed >= self._cfg.max_interval or (
            self._train_improved and elapsed >= self._cfg.min_interval
        )
        return (due and rollout_complete) or step == timesteps

    def validate(
        self,
        *,
        agent: BaseAgent,
        observations: torch.Tensor,
        states: State,
        timestep: int,
        timesteps: int,
    ) -> tuple[ValidationResult, torch.Tensor, State]:
        """Evaluate, save an improved checkpoint, and return the current environment state."""
        step = timestep + 1
        result, observations, states = self._evaluate(
            agent=agent,
            observations=observations,
            states=states,
            ready=False,
            timestep=timestep,
            timesteps=timesteps,
        )

        self._last_validation_step = step
        self._train_improved = False

        if not result.validate():
            log.warning("Validation at step %d produced an invalid result", step)
            return result, observations, states

        score = float(result.scores.mean())
        if score > self._best_score:
            checkpoint = Path(agent.experiment_dir) / "checkpoints" / "best_agent.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            agent.save(str(checkpoint))
            self._best_score = score
            log.info("Saved best_agent.pt at step %d: %.6g", step, score)

        return result, observations, states

    def evaluate(self, *, agent: BaseAgent) -> ValidationResult:
        """Evaluate the configured number of complete episodes from a reset."""
        observations, _ = self._env.reset()
        states = self._env.state()
        result, _, _ = self._evaluate(
            agent=agent,
            observations=observations,
            states=states,
            ready=True,
            timestep=0,
            timesteps=0,
        )
        return result

    def _evaluate(
        self,
        *,
        agent: BaseAgent,
        observations: torch.Tensor,
        states: State,
        ready: bool,
        timestep: int,
        timesteps: int,
    ) -> tuple[ValidationResult, torch.Tensor, State]:
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
        episode_metrics: dict[str, list[object]] = {}

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
                    next_observations, rewards, terminated, truncated, infos = self._env.step(
                        actions
                    )
                    next_states = self._env.state()
                    returns.add_(rewards.reshape(-1) * ready)
                    dones = (terminated | truncated).reshape(-1)
                    agent.reset_state(dones)
                    for index in dones.nonzero().flatten().tolist():
                        if ready[index] and counts[index] < quotas[index]:
                            scores.append(returns[index].item())
                            for key, value in self._terminal_info(infos, index).items():
                                if isinstance(value, (bool, int, float, np.bool_, np.number)):
                                    episode_metrics.setdefault(key, []).append(value)
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
                result = ValidationResult(
                    scores=np.asarray(scores),
                    metrics={key: tuple(values) for key, values in episode_metrics.items()},
                )
                return result, observations, states
        finally:
            agent.enable_training_mode(enabled=True)

    def _terminal_info(self, infos: dict[str, object], index: int) -> dict[str, object]:
        """Return terminal info for one single or vector environment."""
        if self._env.num_envs == 1:
            return infos

        final_info = infos.get("final_info")
        if not isinstance(final_info, dict):
            return {}

        return {key: value[index] for key, value in final_info.items() if not key.startswith("_")}
