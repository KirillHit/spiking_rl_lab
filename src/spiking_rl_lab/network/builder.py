"""Network factory entry point."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from spiking_rl_lab.core.exception import NetworkCreationError
from spiking_rl_lab.core.factory import (
    FactoryConfig,
    RegistrySpec,
    build_configured_instance,
    register_in_registry,
)
from spiking_rl_lab.network.network import BaseNetwork, Network

if TYPE_CHECKING:
    from collections.abc import Callable

    from spiking_rl_lab.network.shape import TensorShape

log = logging.getLogger(__name__)
TNetwork = TypeVar("TNetwork", bound="BaseNetwork")


@dataclass(kw_only=True, slots=True)
class NetworkConfig(FactoryConfig):
    """Configuration for a registered network."""

    name: str = "sequential"


NETWORK_REGISTRY: dict[str, type[BaseNetwork]] = {
    "sequential": Network,
}
NETWORK_SPEC = RegistrySpec[BaseNetwork](
    registry=NETWORK_REGISTRY,
    base_cls=BaseNetwork,
    error_cls=NetworkCreationError,
    kind="network",
)


def build_network(cfg: NetworkConfig, input_shape: TensorShape) -> BaseNetwork:
    """Build a network from configuration."""
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


def register_network(name: str) -> Callable[[type[TNetwork]], type[TNetwork]]:
    """Register a network class under a given name."""
    return register_in_registry(name, NETWORK_SPEC)
