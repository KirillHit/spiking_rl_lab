"""REINFORCE agent implementation."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar

from spiking_rl_lab.agents.base_agent import BaseAgent, BaseAgentCfg
from spiking_rl_lab.agents.builder import register_agent
from spiking_rl_lab.utils.exception import AgentCreationError

if TYPE_CHECKING:
    import gymnasium
    import torch
    from skrl.memories.torch import Memory
    from skrl.models.torch import Model


@dataclasses.dataclass(kw_only=True)
class ReinforceCfg(BaseAgentCfg):
    """Configuration for the REINFORCE agent."""

    learning_rate: float = 1e-3


@register_agent("reinforce")
class Reinforce(BaseAgent):
    """REINFORCE agent implementation."""

    cfg_cls: ClassVar[type[ReinforceCfg]] = ReinforceCfg

    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: ReinforceCfg,
    ) -> None:
        """REINFORCE agent implementation."""
        self.cfg: ReinforceCfg
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        if self.policy is None:
            msg = "The REINFORCE agent requires a 'policy' model."
            raise AgentCreationError(msg)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
