"""Network node registration and builders."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from spiking_rl_lab.network.nodes.base_node import BaseNode
from spiking_rl_lab.utils.exception import NetworkCreationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from spiking_rl_lab.network.shape import TensorShape
    from spiking_rl_lab.utils.config import NetworkNodeCfg

TNode = TypeVar("TNode", bound=BaseNode)

NODES_REGISTRY: dict[str, type[BaseNode]] = {}


def register_node(name: str) -> Callable[[type[TNode]], type[TNode]]:
    """Register a node class under a given name."""

    def decorator(cls: type[TNode]) -> type[TNode]:
        if not issubclass(cls, BaseNode):
            msg = f"Registered class must inherit {BaseNode.__name__}, got: {cls!r}"
            raise TypeError(msg)

        if name in NODES_REGISTRY:
            msg = f"Node name '{name}' is already registered"
            raise NetworkCreationError(msg)

        NODES_REGISTRY[name] = cls
        return cls

    return decorator


def build_node(node_cfg: NetworkNodeCfg, input_shape: TensorShape) -> BaseNode:
    """Build a registered network node from configuration."""
    name = node_cfg.type
    cls = NODES_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(NODES_REGISTRY)) or "<none>"
        msg = f"Unsupported network node '{name}'. Available nodes: {available}"
        raise NetworkCreationError(msg)

    try:
        concrete_node_cfg = cls.cfg_cls(**node_cfg.params)
        return cls(cfg=concrete_node_cfg, input_shape=input_shape)
    except NetworkCreationError:
        raise
    except Exception as exc:
        msg = f"Failed to create network node '{name}'"
        raise NetworkCreationError(msg) from exc
