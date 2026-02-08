"""Simple multilayer perceptron model for skrl Torch policies and values."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from spiking_rl_lab.models.base_model import BaseModel, BaseModelCfg
from spiking_rl_lab.models.builder import register_model

if TYPE_CHECKING:
    import gymnasium
    import torch


@dataclasses.dataclass(kw_only=True)
class MLPCfg(BaseModelCfg):
    """Configuration for the MLP model."""


@register_model("mlp")
class MLP(BaseModel):
    """REINFORCE agent implementation."""

    def __init__(
        self,
        *,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: MLPCfg | dict | None = None,
    ) -> None:
        """REINFORCE agent implementation."""
        if cfg is None:
            cfg = {}
        self.cfg: MLPCfg
        super().__init__(
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=MLPCfg(**cfg) if isinstance(cfg, dict) else cfg,
        )
