"""Gymnasium environment for NeuroDrive Lab.

Importing this package registers ``NeuroDrive-v0`` with Gymnasium, so
``gymnasium.make("NeuroDrive-v0")`` works.
"""

import gymnasium as gym

from neurodrive.env.config import EnvConfig
from neurodrive.env.driving_env import DrivingEnv

__all__ = ["DrivingEnv", "EnvConfig"]

if "NeuroDrive-v0" not in gym.registry:
    gym.register(id="NeuroDrive-v0", entry_point="neurodrive.env.driving_env:DrivingEnv")
