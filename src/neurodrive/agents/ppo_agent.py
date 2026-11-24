"""A trained PPO policy behind the common Controller interface.

Saving and metadata conventions are shared with NEAT: see :mod:`neurodrive.agents.model_io`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from neurodrive.agents.model_io import (  # re-exported for existing imports
    ModelError,
    check_compatible,
    meta_path,
    read_model_meta,
    write_model_meta,
)

__all__ = [
    "ModelError",
    "PPOController",
    "check_compatible",
    "meta_path",
    "read_model_meta",
    "write_model_meta",
]


class PPOController:
    kind = "ppo"

    def __init__(self, model: BaseAlgorithm, name: str = "ppo", meta: dict[str, Any] | None = None):
        self.model = model
        self.name = name
        self.meta = meta or {}

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> PPOController:
        path = Path(path)
        if not path.is_file():
            raise ModelError(f"model file not found: {path}")
        meta = read_model_meta(path)
        if meta.get("algorithm") != "ppo":
            raise ModelError(f"{path.name} is not a PPO model (algorithm={meta.get('algorithm')})")
        model = PPO.load(path, device=device)
        label = meta.get("label", path.stem)
        return cls(model, name=f"ppo {label}", meta=meta)

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        # deterministic=True: use the policy's mean action, no exploration noise. Evaluating
        # with noise would be unfair to PPO next to deterministic drivers.
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float64)
