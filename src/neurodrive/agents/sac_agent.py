"""A trained SAC policy behind the common Controller interface (same conventions as PPO)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from neurodrive.agents.model_io import ModelError, read_model_meta


class SACController:
    kind = "sac"

    def __init__(self, model: BaseAlgorithm, name: str = "sac", meta: dict[str, Any] | None = None):
        self.model = model
        self.name = name
        self.meta = meta or {}

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> SACController:
        path = Path(path)
        if not path.is_file():
            raise ModelError(f"model file not found: {path}")
        meta = read_model_meta(path)
        if meta.get("algorithm") != "sac":
            raise ModelError(f"{path.name} is not a SAC model (algorithm={meta.get('algorithm')})")
        model = SAC.load(path, device=device)
        return cls(model, name=f"sac {meta.get('label', path.stem)}", meta=meta)

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        # deterministic=True: the policy's mean action (tanh-squashed), no exploration noise.
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float64)
