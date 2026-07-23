"""Common types for node-based networks."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod

import numpy as np

from spiking_rl_lab.core.validation import require_positive


class TensorShape(ABC):
    """Base interface for tensor shapes without the batch dimension."""

    @property
    @abstractmethod
    def dims(self) -> np.ndarray:
        """Return tensor dimensions without the batch dimension."""

    @property
    @abstractmethod
    def fields(self) -> dict[str, int]:
        """Return tensor dimensions keyed by semantic field name."""

    @classmethod
    def dense(cls, features: int) -> DenseTensorShape:
        """Build a flat dense-network shape ``[batch, features]``."""
        return DenseTensorShape(features=features)

    @classmethod
    def sequence(cls, channels: int, length: int) -> SequenceTensorShape:
        """Build a channel-first 1D shape ``[batch, channels, length]``."""
        return SequenceTensorShape(channels=channels, length=length)

    @classmethod
    def image(cls, channels: int, height: int, width: int) -> ImageTensorShape:
        """Build a channel-first image shape ``[batch, channels, height, width]``."""
        return ImageTensorShape(channels=channels, height=height, width=width)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class DenseTensorShape(TensorShape):
    """Dense networks: ``[batch, features]``."""

    features: int

    @property
    def dims(self) -> np.ndarray:
        """Return tensor dimensions without the batch dimension."""
        return np.asarray([self.features], dtype=np.int64)

    @property
    def fields(self) -> dict[str, int]:
        """Return tensor dimensions keyed by semantic field name."""
        return {"features": self.features}

    def __post_init__(self) -> None:
        """Validate shape dimensions."""
        require_positive("DenseTensorShape.features", self.features)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SequenceTensorShape(TensorShape):
    """1D tensors: ``[batch, channels, length]``."""

    channels: int
    length: int

    @property
    def dims(self) -> np.ndarray:
        """Return tensor dimensions without the batch dimension."""
        return np.asarray([self.channels, self.length], dtype=np.int64)

    @property
    def fields(self) -> dict[str, int]:
        """Return tensor dimensions keyed by semantic field name."""
        return {"channels": self.channels, "length": self.length}

    def __post_init__(self) -> None:
        """Validate shape dimensions."""
        require_positive("SequenceTensorShape.channels", self.channels)
        require_positive("SequenceTensorShape.length", self.length)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ImageTensorShape(TensorShape):
    """Images: ``[batch, channels, height, width]``."""

    channels: int
    height: int
    width: int

    @property
    def dims(self) -> np.ndarray:
        """Return tensor dimensions without the batch dimension."""
        return np.asarray([self.channels, self.height, self.width], dtype=np.int64)

    @property
    def fields(self) -> dict[str, int]:
        """Return tensor dimensions keyed by semantic field name."""
        return {"channels": self.channels, "height": self.height, "width": self.width}

    def __post_init__(self) -> None:
        """Validate shape dimensions."""
        require_positive("ImageTensorShape.channels", self.channels)
        require_positive("ImageTensorShape.height", self.height)
        require_positive("ImageTensorShape.width", self.width)


def require_shape[ShapeT: TensorShape](
    name: str,
    shape: TensorShape,
    shape_type: type[ShapeT] | tuple[type[ShapeT], ...],
) -> ShapeT:
    """Require a tensor shape of the expected type."""
    if isinstance(shape, shape_type):
        return shape

    if isinstance(shape_type, tuple):
        expected = " or ".join(item.__name__ for item in shape_type)
    else:
        expected = shape_type.__name__
    msg = f"{name} must be {expected}, got {type(shape).__name__}"
    raise TypeError(msg)
