"""Observation handling for Gymnasium same-step autoreset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space

if TYPE_CHECKING:
    import torch
    from gymnasium import Space


def bootstrap_observations(
    next_observations: torch.Tensor,
    infos: object,
    *,
    observation_space: Space,
) -> torch.Tensor:
    """Restore terminal observations replaced by Gymnasium same-step autoreset."""
    if not isinstance(infos, dict) or "final_obs" not in infos:
        return next_observations

    observations = next_observations.clone()
    for index in np.flatnonzero(infos["_final_obs"]):
        final_observation = flatten_tensorized_space(
            tensorize_space(
                observation_space, infos["final_obs"][index], device=observations.device
            )
        )
        observations[index] = final_observation.reshape_as(observations[index])
    return observations
