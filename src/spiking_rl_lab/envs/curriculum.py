"""Curriculum capability for environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spiking_rl_lab.trainers.validator import ValidationResult


class Curriculum(ABC):
    """Define the curriculum lifecycle implemented by concrete environments."""

    @abstractmethod
    def reset_curriculum(self) -> None:
        """Reset curriculum before training."""

    @abstractmethod
    def update_curriculum(self, result: ValidationResult) -> bool:
        """Update curriculum from validation and report whether it changed."""

    @abstractmethod
    def maximize_curriculum(self) -> None:
        """Select the curriculum state used for evaluation and demonstration."""
