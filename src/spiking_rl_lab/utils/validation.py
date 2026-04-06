"""Shared validation helpers for typed configuration objects."""

from __future__ import annotations


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
