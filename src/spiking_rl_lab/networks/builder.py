"""Network factory entry point."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

import gymnasium as gym

from spiking_rl_lab.core.exception import NetworkCreationError
from spiking_rl_lab.core.factory import (
    FactoryConfig,
    RegistrySpec,
    build_configured_instance,
    register_in_registry,
)
from spiking_rl_lab.networks.base_network import BaseNetwork
from spiking_rl_lab.networks.shape import TensorShape

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from skrl.envs.wrappers.torch import Wrapper

log = logging.getLogger(__name__)


NETWORK_MODULES = ["spiking_rl_lab.networks.node_network"]
NETWORK_REGISTRY: dict[str, type[BaseNetwork]] = {}
NETWORK_SPEC = RegistrySpec[BaseNetwork](
    registry=NETWORK_REGISTRY,
    base_cls=BaseNetwork,
    error_cls=NetworkCreationError,
    kind="network",
)


@dataclass(kw_only=True, slots=True)
class NetworkConfig(FactoryConfig):
    """Configuration for a registered network."""

    name: str = "node_graph"


def register_network(name: str) -> Callable[[type[BaseNetwork]], type[BaseNetwork]]:
    """Register a network class under a given name."""
    return register_in_registry(name, NETWORK_SPEC)


def _register_network_modules() -> None:
    """Import network implementation modules so decorators register them."""
    for module_name in NETWORK_MODULES:
        try:
            import_module(module_name)
        except ImportError as exc:
            msg = f"Failed to import network module '{module_name}': {exc}"
            raise NetworkCreationError(msg) from exc


def build_network(cfg: NetworkConfig, input_shape: TensorShape) -> BaseNetwork:
    """Build a network from configuration."""
    _register_network_modules()
    try:
        log.info("Creating network '%s'...", cfg.name)
        return build_configured_instance(
            cfg,
            spec=NETWORK_SPEC,
            dependencies={"input_shape": input_shape},
        )
    except NetworkCreationError:
        raise
    except Exception as exc:
        msg = f"Failed to create network '{cfg.name}'"
        raise NetworkCreationError(msg) from exc


class NetworkBuildContext:
    """Build and cache named networks for model construction."""

    def __init__(self, cfg: Mapping[str, NetworkConfig], env: Wrapper) -> None:
        """Initialize network build context."""
        self._cfg = cfg
        self._observation_shape = TensorShape.dense(gym.spaces.utils.flatdim(env.observation_space))
        self._networks: dict[str, BaseNetwork] = {}
        self._input_shapes: dict[str, TensorShape] = {}

    def require(self, name: str, input_shape: TensorShape | None = None) -> BaseNetwork:
        """Return a named network, building it if needed."""
        input_shape = input_shape or self._observation_shape

        network = self._networks.get(name)
        if network is not None:
            if self._input_shapes[name] != input_shape:
                msg = f"Network '{name}' was already built with another input shape"
                raise NetworkCreationError(msg)
            return network

        network_cfg = self._cfg.get(name)
        if network_cfg is None:
            available = ", ".join(sorted(self._cfg)) or "<empty>"
            msg = f"Unsupported network reference '{name}'. Available networks: {available}"
            raise NetworkCreationError(msg)

        network = build_network(network_cfg, input_shape)
        self._networks[name] = network
        self._input_shapes[name] = input_shape
        return network
