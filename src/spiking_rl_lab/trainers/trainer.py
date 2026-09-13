"""Sequential training with validation on the active environment."""

from __future__ import annotations

import sys
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

    def evaluate(self) -> float:
        """Evaluate the agent for the configured number of episodes."""
        return self.validator.evaluate(agent=self.agents)

    def train(self) -> None:
        """Train the agent and run scheduled validation between interactions."""
        if self.num_simultaneous_agents != 1:
            msg = "Validation trainer supports one agent"
            raise RuntimeError(msg)

        agent = self.agents
        agent.enable_training_mode(enabled=True)
        observations, infos = self.env.reset()
        states = self.env.state()

        timesteps = self.cfg.timesteps
        progress = tqdm.tqdm(
            range(timesteps),
            disable=self.cfg.disable_progressbar,
            file=sys.stdout,
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

            if self.env.num_envs > 1:
                observations, states = next_observations, next_states
            elif terminated.any() or truncated.any():
                with torch.no_grad():
                    observations, infos = self.env.reset()
                    states = self.env.state()
            else:
                observations, states = next_observations, next_states

            if validate:
                observations, states = self.validator.validate(
                    agent=agent,
                    observations=observations,
                    states=states,
                    timestep=timestep,
                    timesteps=timesteps,
                )
