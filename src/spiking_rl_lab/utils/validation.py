"""Helpers for validating and resolving typed fields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hydra.utils import get_class, get_object

if TYPE_CHECKING:
    from collections.abc import Callable


def validate_min(name: str, value: float, *, minimum: float) -> None:
    """Validate that ``value`` is greater than or equal to ``minimum``."""
    if value < minimum:
        msg = f"{name} must be >= {minimum} (got {value})"
        raise ValueError(msg)


def validate_positive(name: str, value: float) -> None:
    """Validate that ``value`` is strictly positive."""
    if value <= 0:
        msg = f"{name} must be > 0 (got {value})"
        raise ValueError(msg)


def validate_range(name: str, value: float, *, minimum: float, maximum: float) -> None:
    """Validate that ``value`` falls within the inclusive range."""
    if not minimum <= value <= maximum:
        msg = f"{name} must be in [{minimum}, {maximum}] (got {value})"
        raise ValueError(msg)


def validate_optional_callable(name: str, value: object | None) -> None:
    """Validate that an optional config hook is callable when provided."""
    if value is not None and not callable(value):
        msg = f"{name} must be callable or None"
        raise TypeError(msg)


def resolve_optional_class(name: str, value: str | type[Any] | None) -> type[Any] | None:
    """Resolve an optional dotted class path into a class object."""
    if value is None:
        return None

    if isinstance(value, str):
        try:
            value = get_class(value)
        except Exception as exc:
            msg = f"{name} must reference an importable class (got {value!r})"
            raise TypeError(msg) from exc

    if not isinstance(value, type):
        msg = f"{name} must be a class, dotted class path, or None"
        raise TypeError(msg)

    return value


def resolve_optional_callable(
    name: str,
    value: str | Callable[..., Any] | None,
) -> Callable[..., Any] | None:
    """Resolve an optional dotted callable path into a callable object."""
    if value is None:
        return None

    if isinstance(value, str):
        try:
            value = get_object(value)
        except Exception as exc:
            msg = f"{name} must reference an importable callable (got {value!r})"
            raise TypeError(msg) from exc

    if not callable(value):
        msg = f"{name} must be callable, dotted callable path, or None"
        raise TypeError(msg)

    return value
