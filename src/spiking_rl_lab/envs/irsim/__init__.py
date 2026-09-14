"""Gymnasium environments backed by IR-SIM."""

from gymnasium.envs.registration import register

from spiking_rl_lab.envs.irsim.dynamic_avoidance import IRSimDynamicAvoidance

register(
    id="IRSimDynamicAvoidance-v0",
    entry_point="spiking_rl_lab.envs.irsim:IRSimDynamicAvoidance",
)

__all__ = ["IRSimDynamicAvoidance"]
