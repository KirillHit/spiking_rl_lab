"""Network node registration and builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from spiking_rl_lab.core.exception import NetworkCreationError
from spiking_rl_lab.core.factory import (
    FactoryConfig,
    RegistrySpec,
    build_configured_instance,
    register_in_registry,
)
from spiking_rl_lab.network.nodes.base_node import BaseNode

if TYPE_CHECKING:
    from collections.abc import Callable

    from spiking_rl_lab.network.shape import TensorShape

TNode = TypeVar("TNode", bound=BaseNode)


@dataclass(kw_only=True, slots=True)
class NetworkNodeCfg(FactoryConfig):
    """Configuration for a single network node."""


NODES_REGISTRY: dict[str, type[BaseNode]] = {}
NODE_SPEC = RegistrySpec[BaseNode](
    registry=NODES_REGISTRY,
    base_cls=BaseNode,
    error_cls=NetworkCreationError,
    kind="network node",
)


def register_node(name: str) -> Callable[[type[TNode]], type[TNode]]:
    """Register a node class under a given name."""
    return register_in_registry(name, NODE_SPEC)


def build_node(node_cfg: NetworkNodeCfg, input_shape: TensorShape) -> BaseNode:
    """Build a registered network node from configuration."""
    return build_configured_instance(
        node_cfg,
        spec=NODE_SPEC,
        dependencies={"input_shape": input_shape},
    )
