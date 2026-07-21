"""Shared base classes for environment backends."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from spiking_rl_lab.core.factory import ConfiguredBase

if TYPE_CHECKING:
    from skrl.envs.wrappers.torch import Wrapper


class BaseEnvBackend(ConfiguredBase, ABC):
    """Base class for environment backend adapters."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config:
        """Base environment backend configuration."""

    def __init__(self, cfg: Config) -> None:
        """Store backend configuration."""
        super().__init__(cfg)

    @abstractmethod
    def build(self) -> Wrapper:
        """Build and return a wrapped environment."""
