"""Sequential training with validation on the active environment."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import tqdm
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from skrl.utils import ScopedTimer

from spiking_rl_lab.trainers.validator import ValidationConfig, Validator

if TYPE_CHECKING:
    from skrl.agents.torch import Agent
    from skrl.envs.wrappers.torch import Wrapper

    from spiking_rl_lab.trainers.validator import ValidationResult


@dataclass(kw_only=True)
class TrainerConfig(SequentialTrainerCfg):
    """Sequential trainer settings with validation."""

    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def __post_init__(self) -> None:
        """Build the nested validation configuration."""
        if not isinstance(self.validation, ValidationConfig):
            self.validation = ValidationConfig(**self.validation)


class Trainer(SequentialTrainer):
    """Run one agent and resume from the environment state left by validation."""

    def __init__(
        self,
        *,
        env: Wrapper,
        agents: Agent,
        cfg: TrainerConfig | dict[str, Any] | None = None,
    ) -> None:
        """Initialize the training loop and its validator."""
        cfg = TrainerConfig(**cfg) if isinstance(cfg, dict) else cfg or TrainerConfig()
        super().__init__(env=env, agents=agents, cfg=cfg)
        self.validator = Validator(cfg.validation, env)

    @property
    def validation_score(self) -> float:
        """Return the best validation score observed during training."""
        return self.validator.score

    def evaluate(self) -> None:
        """Evaluate the agent for the configured number of episodes."""
        self._maximize_curriculum()
        result = self.validator.evaluate(agent=self.agents)
        result.log_console()

    def demo(self, *, fps: float) -> None:
        """Demonstrate the agent indefinitely without collecting training data."""
        if self.env.num_envs != 1:
            msg = "Demo mode requires exactly one environment"
            raise RuntimeError(msg)

        frame_period = 1.0 / fps
        next_frame = time.monotonic()

        agent = self.agents
        agent.enable_training_mode(enabled=False)
        self._maximize_curriculum()
        observations, _ = self.env.reset()
        states = self.env.state()

        try:
            while True:
                with torch.no_grad():
                    actions, _ = agent.act(
                        observations,
                        states,
                        timestep=0,
                        timesteps=0,
                    )
                    next_observations, _, terminated, truncated, _ = self.env.step(actions)
                    next_states = self.env.state()

                dones = (terminated | truncated).reshape(-1)
                agent.reset_state(dones)
                if dones.any():
                    observations, _ = self.env.reset()
                    states = self.env.state()
                else:
                    observations, states = next_observations, next_states

                next_frame += frame_period
                delay = next_frame - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_frame = time.monotonic()
        finally:
            agent.enable_training_mode(enabled=True)

    def train(self) -> None:
        """Train the agent and run scheduled validation between interactions."""
        if self.num_simultaneous_agents != 1:
            msg = "Validation trainer supports one agent"
            raise RuntimeError(msg)

        agent = self.agents
        agent.enable_training_mode(enabled=True)
        self._reset_curriculum()
        observations, infos = self.env.reset()
        states = self.env.state()

        timesteps = self.cfg.timesteps
        progress = tqdm.tqdm(
            range(timesteps),
            disable=self.cfg.disable_progressbar,
            file=sys.stdout,
            smoothing=0.01,
        )
        for timestep in progress:
            agent.pre_interaction(timestep=timestep, timesteps=timesteps)

            with torch.no_grad():
                with ScopedTimer() as timer:
                    actions, _ = agent.act(
                        observations,
                        states,
                        timestep=timestep,
                        timesteps=timesteps,
                    )
                    agent.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)

                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(
                        actions
                    )
                    next_states = self.env.state()
                    agent.track_data("Stats / Env stepping time (ms)", timer.elapsed_time_ms)

                if not self.cfg.headless and not timestep % self.cfg.render_interval:
                    self.env.render()

                agent.record_transition(
                    observations=observations,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=timesteps,
                )

                if self.cfg.environment_info in infos:
                    for key, value in infos[self.cfg.environment_info].items():
                        if isinstance(value, torch.Tensor) and value.numel() == 1:
                            agent.track_data(key if "/" in key else f"Info / {key}", value.item())

            validate = self.validator.should_validate(
                timestep=timestep,
                timesteps=timesteps,
                train_rewards=agent.training_rewards,
                rollout_complete=agent.rollout_complete,
            )
            agent.post_interaction(timestep=timestep, timesteps=timesteps)

            observations, states = next_observations, next_states
            if self.env.num_envs == 1 and (terminated.any() or truncated.any()):
                with torch.no_grad():
                    observations, infos = self.env.reset()
                    states = self.env.state()

            if validate:
                result, observations, states = self.validator.validate(
                    agent=agent,
                    observations=observations,
                    states=states,
                    timestep=timestep,
                    timesteps=timesteps,
                )
                result.log_console(timestep + 1)
                result.log_mlflow(timestep + 1)
                if timestep + 1 < timesteps and result.validate():
                    self._update_curriculum(result)

    def _reset_curriculum(self) -> None:
        """Reset curriculum before training."""
        self._curriculum_call("reset_curriculum")

    def _maximize_curriculum(self) -> None:
        """Maximize curriculum difficulty for evaluation."""
        self._curriculum_call("maximize_curriculum")

    def _update_curriculum(self, result: ValidationResult) -> None:
        """Update curriculum after validation."""
        updates = self._curriculum_call("update_curriculum", result)
        if updates and any(updates):
            self.validator.reset_progress_reference()

    def _curriculum_call(self, method: str, *args: object) -> tuple[object, ...] | None:
        """Call a curriculum capability on every underlying Gymnasium environment."""
        if self.env.num_envs > 1:
            supported = self.env.call("has_wrapper_attr", method)
            if not any(supported):
                return None
            if not all(supported):
                msg = f"Curriculum capability '{method}' is only available on some environments"
                raise RuntimeError(msg)
            return self.env.call(method, *args)

        if not self.env.has_wrapper_attr(method):
            return None
        return (self.env.get_wrapper_attr(method)(*args),)
