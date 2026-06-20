"""Network node registration and builders."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

from spiking_rl_lab.core.exception import NetworkCreationError
from spiking_rl_lab.core.factory import (
    FactoryConfig,
    RegistrySpec,
    build_configured_instance,
    register_in_registry,
)
from spiking_rl_lab.networks.nodes.base_node import BaseNode

if TYPE_CHECKING:
    from collections.abc import Callable

    from spiking_rl_lab.networks.shape import TensorShape

_BUILTIN_NODE_MODULES = (
    "spiking_rl_lab.networks.nodes.common_nodes",
    "spiking_rl_lab.networks.nodes.activations",
)
_REGISTERED_BUILTIN_MODULES: set[str] = set()


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


def register_node(name: str) -> Callable[[type[BaseNode]], type[BaseNode]]:
    """Register a node class under a given name."""
    return register_in_registry(name, NODE_SPEC)


def _register_builtin_nodes() -> None:
    """Import built-in nodes once so decorators register them."""
    for module_name in _BUILTIN_NODE_MODULES:
        if module_name in _REGISTERED_BUILTIN_MODULES:
            continue
        import_module(module_name)
        _REGISTERED_BUILTIN_MODULES.add(module_name)


def build_node(node_cfg: NetworkNodeCfg, input_shape: TensorShape) -> BaseNode:
    """Build a registered network node from configuration."""
    _register_builtin_nodes()
    return build_configured_instance(
        node_cfg,
        spec=NODE_SPEC,
        dependencies={"input_shape": input_shape},
    )
