"""Network node factory entry point."""

from __future__ import annotations

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

    from spiking_rl_lab.networks.types import TensorShape


NODE_MODULES = (
    "spiking_rl_lab.networks.nodes.standard.linear",
    "spiking_rl_lab.networks.nodes.standard.convolutions",
    "spiking_rl_lab.networks.nodes.standard.normalizations",
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


def register_node(name: str) -> Callable[[type[BaseNode]], type[BaseNode]]:
    """Register a node class under a given name."""
    return register_in_registry(name, NODE_SPEC)


def _register_node_modules() -> None:
    """Import built-in node modules so decorators register them."""
    for module_name in NODE_MODULES:
        try:
            import_module(module_name)
        except ImportError as exc:
            msg = f"Failed to import network node module '{module_name}': {exc}"
            raise NetworkCreationError(msg) from exc


def build_node(node_cfg: FactoryConfig, input_shape: TensorShape) -> BaseNode:
    """Build a registered network node from configuration."""
    _register_node_modules()
    return build_configured_instance(
        node_cfg,
        spec=NODE_SPEC,
        dependencies={"input_shape": input_shape},
    )
