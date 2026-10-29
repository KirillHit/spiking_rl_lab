"""Observation handling for same-step autoreset (Gymnasium and Isaac Lab)."""

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
    """Restore terminal observations replaced by same-step autoreset.

    Gymnasium vector envs expose ``infos["final_obs"]`` with a boolean
    ``infos["_final_obs"]`` mask. Isaac Lab (with ``compute_final_obs``) stores a
    dict under ``infos["final_obs"]`` whose ``policy`` tensor is the pre-reset
    observation for every env; callers should only consume rows that timed out.
    """
    if not isinstance(infos, dict) or "final_obs" not in infos:
        return next_observations

    final_obs = infos["final_obs"]
    device = next_observations.device

    # Isaac Lab: pre-reset observation groups, e.g. {"policy": Tensor[n_envs, ...]}.
    if isinstance(final_obs, dict) and "policy" in final_obs:
        final_observation = flatten_tensorized_space(
            tensorize_space(observation_space, final_obs["policy"], device=device)
        )
        return final_observation.reshape_as(next_observations)

    # Gymnasium: per-env final observations with an explicit boolean mask.
    if "_final_obs" not in infos:
        return next_observations

    observations = next_observations.clone()
    for index in np.flatnonzero(infos["_final_obs"]):
        final_observation = flatten_tensorized_space(
            tensorize_space(observation_space, final_obs[index], device=device)
        )
        observations[index] = final_observation.reshape_as(observations[index])
    return observations
