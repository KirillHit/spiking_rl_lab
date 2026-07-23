"""Operations on explicit network states."""

from __future__ import annotations

import torch

type ListState = list[object | None | ListState]


def detach_state[StateT](state: StateT) -> StateT:
    """Detach every tensor in a nested network state from its autograd graph."""
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, list):
        return [detach_state(item) for item in state]
    if isinstance(state, tuple):
        values = tuple(detach_state(item) for item in state)
        return type(state)(*values) if hasattr(state, "_fields") else values
    if isinstance(state, dict):
        return {key: detach_state(value) for key, value in state.items()}
    return state
