"""Helpers for requiring valid typed fields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hydra.utils import get_class, get_object

if TYPE_CHECKING:
    from collections.abc import Callable


def require_minimum(name: str, value: float, *, minimum: float) -> None:
    """Raise if ``value`` is less than ``minimum``."""
    if value < minimum:
        msg = f"{name} must be >= {minimum} (got {value})"
        raise ValueError(msg)


def require_positive(name: str, value: float) -> None:
    """Raise if ``value`` is not strictly positive."""
    if value <= 0.0:
        msg = f"{name} must be positive, got {value}."
        raise ValueError(msg)


def require_range(name: str, value: float, *, minimum: float, maximum: float) -> None:
    """Raise if ``value`` is outside the inclusive range."""
    if not minimum <= value <= maximum:
        msg = f"{name} must be in [{minimum}, {maximum}] (got {value})"
        raise ValueError(msg)


def require_optional_class(name: str, value: str | type[Any] | None) -> type[Any] | None:
    """Require an optional class value, accepting dotted class paths."""
    if value is None:
        return None

    if isinstance(value, str):
        try:
            return get_class(value)
        except Exception as exc:
            msg = f"{name} must reference an importable class (got {value!r})"
            raise TypeError(msg) from exc

    if isinstance(value, type):
        return value

    msg = f"{name} must be a class, dotted class path, or None"
    raise TypeError(msg)


def require_optional_callable(
    name: str,
    value: str | Callable[..., Any] | None,
) -> Callable[..., Any] | None:
    """Require an optional callable value, accepting dotted callable paths."""
    if value is None:
        return None

    if isinstance(value, str):
        try:
            resolved_callable = get_object(value)
        except Exception as exc:
            msg = f"{name} must reference an importable callable (got {value!r})"
            raise TypeError(msg) from exc

        if not callable(resolved_callable):
            msg = f"{name} must reference a callable (got {value!r})"
            raise TypeError(msg)
        return resolved_callable

    if callable(value):
        return value

    msg = f"{name} must be callable, dotted callable path, or None"
    raise TypeError(msg)
