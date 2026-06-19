"""Shared factory helpers for configured registries."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from omegaconf import MISSING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, MutableMapping


@dataclass(kw_only=True, slots=True)
class FactoryConfig:
    """Registry-backed component configuration."""

    name: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


class ConfiguredBase(ABC):
    """Base class for components that validate their typed config."""

    Config: ClassVar[type[object]]

    def __init__(self, cfg: object) -> None:
        """Validate and store a typed configuration instance."""
        if not isinstance(cfg, self.Config):
            msg = (
                f"{self.__class__.__name__} expects {self.Config.__name__}, "
                f"got {type(cfg).__name__}"
            )
            raise TypeError(msg)
        self._cfg = cfg


@dataclass(frozen=True, kw_only=True, slots=True)
class RegistrySpec[TBase]:
    """Registry metadata needed to instantiate a configured component."""

    registry: MutableMapping[str, type[TBase]]
    base_cls: type[TBase]
    error_cls: type[Exception]
    kind: str


def build_configured_instance[TBase](
    cfg: FactoryConfig,
    spec: RegistrySpec[TBase],
    dependencies: Mapping[str, object] | None = None,
) -> TBase:
    """Instantiate a registry entry after validating its config."""
    if not hasattr(cfg, "name") or not hasattr(cfg, "params"):
        missing = [attr for attr in ("name", "params") if not hasattr(cfg, attr)]
        msg = f"Invalid config for {spec.kind}: missing attribute(s) {', '.join(missing)}"
        raise spec.error_cls(msg)
    component_class = spec.registry.get(cfg.name)

    if component_class is None:
        available = ", ".join(sorted(spec.registry)) or "<empty>"
        msg = f"Unsupported {spec.kind} name: {cfg.name}. Available {spec.kind}s: {available}"
        raise spec.error_cls(msg)

    if not issubclass(component_class, spec.base_cls):
        msg = f"Registered class must inherit {spec.base_cls.__name__}, got: {component_class!r}"
        raise spec.error_cls(msg)

    try:
        component_cfg = component_class.Config(**cfg.params)
    except TypeError as exc:
        msg = (
            f"Invalid config for {spec.kind} '{cfg.name}' "
            f"({component_class.Config.__name__}): {exc}"
        )
        raise spec.error_cls(msg) from exc

    return component_class(component_cfg, **(dependencies or {}))


def register_in_registry[TBase](
    name: str,
    spec: RegistrySpec[TBase],
) -> Callable[[type[TBase]], type[TBase]]:
    """Register a class in a registry with validation."""

    def decorator(cls: type[TBase]) -> type[TBase]:
        if not issubclass(cls, spec.base_cls):
            msg = f"Registered class must inherit {spec.base_cls.__name__}, got: {cls!r}"
            raise spec.error_cls(msg)

        if name in spec.registry:
            msg = f"{spec.kind.capitalize()} name '{name}' is already registered"
            raise spec.error_cls(msg)

        spec.registry[name] = cls
        return cls

    return decorator
