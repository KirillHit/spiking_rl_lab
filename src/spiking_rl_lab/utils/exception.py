"""Custom exceptions for spiking_rl_lab."""


class SpikingRLLabError(Exception):
    """Base exception for spiking_rl_lab errors."""


class EnvironmentCreationError(SpikingRLLabError):
    """Raised when an environment cannot be created."""
