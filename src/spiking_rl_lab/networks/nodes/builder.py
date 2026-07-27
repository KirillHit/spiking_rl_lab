"""Network node factory entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiking_rl_lab.core.exception import NetworkCreationError
from spiking_rl_lab.core.factory import (
    FactoryConfig,
    RegistrySpec,
    build_configured_instance,
    import_registry_modules,
    register_in_registry,
)
from spiking_rl_lab.networks.nodes.base_node import BaseNode

if TYPE_CHECKING:
    from collections.abc import Callable

    from spiking_rl_lab.networks.shape import TensorShape


NODE_MODULES = (
    "spiking_rl_lab.networks.nodes.standard.linear",
    "spiking_rl_lab.networks.nodes.standard.convolutions",
    "spiking_rl_lab.networks.nodes.encodings.population",
    "spiking_rl_lab.networks.nodes.decodings.opponent",
    "spiking_rl_lab.networks.nodes.standard.activations",
    "spiking_rl_lab.networks.nodes.spiking.activations",
)
NODE_REGISTRY: dict[str, type[BaseNode]] = {}
NODE_SPEC = RegistrySpec[BaseNode](
    registry=NODE_REGISTRY,
    base_cls=BaseNode,
    error_cls=NetworkCreationError,
    kind="network node",
)


@dataclass(kw_only=True, slots=True)
class NodeConfig(FactoryConfig):
    """Configuration for one registered network node."""


def register_node(name: str) -> Callable[[type[BaseNode]], type[BaseNode]]:
    """Register a node class under a given name."""
    return register_in_registry(name, NODE_SPEC)


def build_node(cfg: NodeConfig, input_shape: TensorShape) -> BaseNode:
    """Build a registered network node from configuration."""
    import_registry_modules(NODE_MODULES, NODE_SPEC)
    return build_configured_instance(
        cfg,
        spec=NODE_SPEC,
        dependencies={"input_shape": input_shape},
    )
