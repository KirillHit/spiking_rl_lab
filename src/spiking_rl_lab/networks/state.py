"""Operations on explicit network states."""

from __future__ import annotations

import torch

type ListState = list[object | ListState | None]


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


def concatenate_states[StateT](states: list[StateT]) -> StateT:
    """Concatenate matching network states along their batch dimension."""
    first = states[0]
    if isinstance(first, torch.Tensor):
        reference = next((state for state in states if state.ndim), None)
        if reference is None:
            if len(states) > 1:
                msg = "Cannot concatenate multiple scalar network states without a batch shape"
                raise ValueError(msg)
            return first.detach()
        return torch.cat(
            [state.to(reference).expand_as(reference) if not state.ndim else state for state in states]
        )
    if isinstance(first, list):
        return [
            concatenate_states([state[index] for state in states]) for index in range(len(first))
        ]
    if isinstance(first, tuple):
        values = tuple(
            concatenate_states([state[index] for state in states]) for index in range(len(first))
        )
        return type(first)(*values) if hasattr(first, "_fields") else values
    if isinstance(first, dict):
        return {key: concatenate_states([state[key] for state in states]) for key in first}
    return first


def select_state[StateT](state: StateT, indices: torch.Tensor) -> StateT:
    """Select sequences from tensors in a batched network state."""
    if isinstance(state, torch.Tensor):
        return state[indices].detach() if state.ndim else state.detach()
    if isinstance(state, list):
        return [select_state(item, indices) for item in state]
    if isinstance(state, tuple):
        values = tuple(select_state(item, indices) for item in state)
        return type(state)(*values) if hasattr(state, "_fields") else values
    if isinstance(state, dict):
        return {key: select_state(value, indices) for key, value in state.items()}
    return state
