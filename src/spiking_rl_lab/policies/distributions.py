"""Project-level action-distribution interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class ActionDistribution(ABC):
    """Distribution of environment actions with project-standard tensor shapes."""

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """Sample a stochastic action for an exploratory rollout."""

    @abstractmethod
    def mode(self) -> torch.Tensor:
        """Return the deterministic representative action for evaluation."""

    @abstractmethod
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Score given actions for a policy-gradient loss, as ``[batch, 1]``."""

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """Return distribution uncertainty for an entropy bonus, as ``[batch, 1]``."""
