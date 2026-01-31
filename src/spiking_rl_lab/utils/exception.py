"""Custom exceptions for spiking_rl_lab."""


class SpikingRLLabError(Exception):
    """Base exception for spiking_rl_lab errors."""


class EnvironmentCreationError(SpikingRLLabError):
    """Raised when an environment cannot be created."""


class ModelCreationError(SpikingRLLabError):
    """Raised when a model cannot be created."""


class AgentCreationError(SpikingRLLabError):
    """Raised when an agent cannot be created."""


class TrainerCreationError(SpikingRLLabError):
    """Raised when a trainer cannot be created."""
