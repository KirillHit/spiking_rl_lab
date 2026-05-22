"""Tensor shape helpers for network builders."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from enum import StrEnum

import numpy as np

from spiking_rl_lab.utils.validation import validate_positive


class TensorShapeKind(StrEnum):
    """Supported tensor shape kinds."""

    DENSE = "dense"
    SEQUENCE = "sequence"
    IMAGE = "image"


class TensorShape(ABC):
    """Base interface for tensor shapes without the batch dimension."""

    @property
    @abstractmethod
    def kind(self) -> TensorShapeKind:
        """Return the tensor shape kind."""

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
    def kind(self) -> TensorShapeKind:
        """Return the tensor shape kind."""
        return TensorShapeKind.DENSE

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
        validate_positive("DenseTensorShape.features", self.features)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SequenceTensorShape(TensorShape):
    """1D tensors: ``[batch, channels, length]``."""

    channels: int
    length: int

    @property
    def kind(self) -> TensorShapeKind:
        """Return the tensor shape kind."""
        return TensorShapeKind.SEQUENCE

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
        validate_positive("SequenceTensorShape.channels", self.channels)
        validate_positive("SequenceTensorShape.length", self.length)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ImageTensorShape(TensorShape):
    """Images: ``[batch, channels, height, width]``."""

    channels: int
    height: int
    width: int

    @property
    def kind(self) -> TensorShapeKind:
        """Return the tensor shape kind."""
        return TensorShapeKind.IMAGE

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
        validate_positive("ImageTensorShape.channels", self.channels)
        validate_positive("ImageTensorShape.height", self.height)
        validate_positive("ImageTensorShape.width", self.width)
