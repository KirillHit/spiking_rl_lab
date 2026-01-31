"""..."""

import copy
import itertools
from collections.abc import Mapping
from typing import Any

import gymnasium
import torch
from packaging import version
from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR, KLAdaptiveRL
from torch import nn
from torch.nn import functional

from .base_agent import BaseAgent

PPO_DEFAULT_CONFIG = {
    "rollouts": 16,  # number of rollouts before updating
    "learning_epochs": 8,  # number of learning epochs during each update
    "mini_batches": 2,  # number of mini batches during each learning epoch
    "discount_factor": 0.99,  # discount factor (gamma)
    "lambda": 0.95,  # TD(lambda) coefficient (lam) for computing returns and advantages
    "learning_rate": 1e-3,  # learning rate
    "learning_rate_scheduler": None,  # learning rate scheduler class (see torch.optim.lr_scheduler)
    # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})
    "learning_rate_scheduler_kwargs": {},
    "state_preprocessor": None,  # state preprocessor class (see skrl.resources.preprocessors)
    # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "state_preprocessor_kwargs": {},
    "value_preprocessor": None,  # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},  # value preprocessor's kwargs (e.g. {"size": 1})
    "random_timesteps": 0,  # random exploration steps
    "learning_starts": 0,  # learning starts after this many steps
    "grad_norm_clip": 0.5,  # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,  # clipping coefficient for computing the clipped surrogate objective
    # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "value_clip": 0.2,
    "clip_predicted_values": False,  # clip predicted values during value loss computation
    "entropy_loss_scale": 0.0,  # entropy loss scaling factor
    "value_loss_scale": 1.0,  # value loss scaling factor
    "kl_threshold": 0,  # KL divergence threshold for early stopping
    # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "rewards_shaper": None,
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)
    "mixed_precision": False,  # enable automatic mixed precision for higher performance
    "experiment": {
        "directory": "",  # experiment's parent directory
        "experiment_name": "",  # experiment name
        "write_interval": "auto",  # TensorBoard writing interval (timesteps)
        "checkpoint_interval": "auto",  # interval for checkpoints (timesteps)
        "store_separately": False,  # whether to store checkpoints separately
        "wandb": False,  # whether to use Weights & Biases
        "wandb_kwargs": {},  # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    },
}


class PPO(BaseAgent):
    """..."""

    def __init__(  # noqa: PLR0912, PLR0915
        self,
        models: Mapping[str, Model],
        memory: Memory | tuple[Memory, ...] | None = None,
        observation_space: int | tuple[int, ...] | gymnasium.Space | None = None,
        action_space: int | tuple[int, ...] | gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        """Proximal Policy Optimization (PPO).

        https://arxiv.org/abs/1707.06347

        Args:
            models: Models used by the agent.
            memory: Memory to store transitions. If a tuple is provided, the first element
                is used for training and the rest receive only environment transitions.
            observation_space: Observation/state space or shape.
            action_space: Action space or shape.
            device: Device on which tensors/arrays are or will be allocated. If None, the
                device is ``"cuda"`` when available, otherwise ``"cpu"``.
            cfg: Configuration dictionary.

        Raises:
            KeyError: If the models dictionary is missing a required key.

        """
        cfg = PPO_DEFAULT_CONFIG.copy()
        cfg["rollouts"] = 1024  # memory_size
        cfg["learning_epochs"] = 10
        cfg["mini_batches"] = 32
        cfg["discount_factor"] = 0.9
        cfg["lambda"] = 0.95
        cfg["learning_rate"] = 1e-3
        cfg["learning_rate_scheduler"] = KLAdaptiveRL
        cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
        cfg["grad_norm_clip"] = 0.5
        cfg["ratio_clip"] = 0.2
        cfg["value_clip"] = 0.2
        cfg["clip_predicted_values"] = False
        cfg["entropy_loss_scale"] = 0.0
        cfg["value_loss_scale"] = 0.5
        cfg["kl_threshold"] = 0
        cfg["mixed_precision"] = True
        cfg["state_preprocessor"] = RunningStandardScaler
        cfg["state_preprocessor_kwargs"] = {"size": observation_space, "device": device}
        cfg["value_preprocessor"] = RunningStandardScaler
        cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
        # logging to TensorBoard and write checkpoints (in timesteps)
        cfg["experiment"]["write_interval"] = 500
        cfg["experiment"]["checkpoint_interval"] = 5000
        cfg["experiment"]["directory"] = "runs/torch/Pendulum"

        _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info("Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(
                device=self._device_type,
                enabled=self._mixed_precision,
            )
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.policy.parameters(), self.value.parameters()),
                    lr=self._learning_rate,
                )
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer,
                    **self.cfg["learning_rate_scheduler_kwargs"],
                )

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(
                **self.cfg["state_preprocessor_kwargs"],
            )
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(
                **self.cfg["value_preprocessor_kwargs"],
            )
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Mapping[str, Any] | None = None) -> None:
        """Initialize the agent.

        Args:
            trainer_cfg: Trainer configuration overrides.

        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(
                name="states",
                size=self.observation_space,
                dtype=torch.float32,
            )
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            # tensors sampled during training
            self._tensors_names = [
                "states",
                "actions",
                "log_prob",
                "values",
                "returns",
                "advantages",
            ]

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

    def act(
        self,
        states: torch.Tensor,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Process states and sample actions from the policy.

        Args:
            states: Environment states.
            timestep: Current timestep.
            timesteps: Total number of timesteps.

        Returns:
            Tuple of (actions, log_prob, outputs).

        """
        # sample random actions
        if timestep < self._random_timesteps:
            return self.policy.random_act(
                {"states": self._state_preprocessor(states)},
                role="policy",
            )

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, log_prob, outputs = self.policy.act(
                {"states": self._state_preprocessor(states)},
                role="policy",
            )
            self._current_log_prob = log_prob

        return actions, log_prob, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: str,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory.

        Args:
            states: Observations/states used to make the decision.
            actions: Actions taken by the agent.
            rewards: Immediate rewards achieved by the current actions.
            next_states: Next observations/states of the environment.
            terminated: Signals indicating episode termination.
            truncated: Signals indicating episode truncation.
            infos: Additional environment information.
            timestep: Current timestep.
            timesteps: Total number of timesteps.

        """
        super().record_transition(
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
            infos,
            timestep,
            timesteps,
        )

        if self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                values, _, _ = self.value.act(
                    {"states": self._state_preprocessor(states)},
                    role="value",
                )
                values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob,
                    values=values,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Run callback before interacting with the environment.

        Args:
            timestep: Current timestep.
            timesteps: Total number of timesteps.

        """

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Run callback after interacting with the environment.

        Args:
            timestep: Current timestep.
            timesteps: Total number of timesteps.

        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:  # noqa: C901, PLR0912, PLR0915
        """Run the main PPO update step.

        Args:
            timestep: Current timestep.
            timesteps: Total number of timesteps.

        """

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute the Generalized Advantage Estimator (GAE).

            Args:
                rewards: Rewards obtained by the agent.
                dones: Signals indicating episode termination.
                values: Value estimates for current states.
                next_values: Value estimates for next states.
                discount_factor: Discount factor.
                lambda_coefficient: GAE lambda coefficient.

            Returns:
                Tuple of (returns, advantages).

            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor
                    * not_dones[i]
                    * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with (
            torch.no_grad(),
            torch.autocast(device_type=self._device_type, enabled=self._mixed_precision),
        ):
            self.value.train(mode=False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())},
                role="value",
            )
            self.value.train(mode=True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated")
            | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(
            names=self._tensors_names,
            mini_batches=self._mini_batches,
        )

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    sampled_states_proc = self._state_preprocessor(sampled_states, train=not epoch)

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states_proc, "taken_actions": sampled_actions},
                        role="policy",
                    )

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = (
                            -self._entropy_loss_scale
                            * self.policy.get_entropy(role="policy").mean()
                        )
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio,
                        1.0 - self._ratio_clip,
                        1.0 + self._ratio_clip,
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _, _ = self.value.act(
                        {"states": sampled_states_proc},
                        role="value",
                    )

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values,
                            min=-self._value_clip,
                            max=self._value_clip,
                        )
                    value_loss = self._value_loss_scale * functional.mse_loss(
                        sampled_returns,
                        predicted_values,
                    )

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()),
                            self._grad_norm_clip,
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data(
            "Loss / Policy loss",
            cumulative_policy_loss / (self._learning_epochs * self._mini_batches),
        )
        self.track_data(
            "Loss / Value loss",
            cumulative_value_loss / (self._learning_epochs * self._mini_batches),
        )
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss",
                cumulative_entropy_loss / (self._learning_epochs * self._mini_batches),
            )

        self.track_data(
            "Policy / Standard deviation",
            self.policy.distribution(role="policy").stddev.mean().item(),
        )

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
