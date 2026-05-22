"""Shared abstractions for network nodes."""

from __future__ import annotations

import copy
import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, get_args, get_origin

import torch
from torch import nn

if TYPE_CHECKING:
    from spiking_rl_lab.network.shape import TensorShape

type ListState = list[Any | None | ListState]


@dataclasses.dataclass(kw_only=True, slots=True)
class BaseNodeCfg:
    """Base node configuration."""


class BaseNode[CfgT: BaseNodeCfg](ABC, nn.Module):
    """Base class for network nodes."""

    cfg_cls: ClassVar[type[BaseNodeCfg]] = BaseNodeCfg

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Infer config class from ``BaseNode[Cfg]`` annotation."""
        super().__init_subclass__(**kwargs)

        for base in getattr(cls, "__orig_bases__", ()):
            origin = get_origin(base)
            if origin is not BaseNode:
                continue

            args = get_args(base)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseNodeCfg):
                cls.cfg_cls = args[0]
                return

    def __init__(self, cfg: CfgT, input_shape: TensorShape) -> None:
        """Store node configuration."""
        super().__init__()
        self._cfg: CfgT = cfg
        self._input_shape = input_shape
        self._output_shape = input_shape

    @property
    def cfg(self) -> CfgT:
        """Return a detached configuration copy."""
        return copy.deepcopy(self._cfg)

    @property
    def output_shape(self) -> TensorShape:
        """Return output shape."""
        return self._output_shape

    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        state: ListState | None = None,
    ) -> tuple[torch.Tensor, ListState]:
        """Compute the next output and state."""
