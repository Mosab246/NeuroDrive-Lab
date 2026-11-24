"""A trained PPO policy behind the common Controller interface.

Every saved model ``X.zip`` has a sidecar ``X.meta.json`` describing what it was trained
on (observation spec, environment config, training step). Loading checks the sidecar
exists, and :func:`check_compatible` refuses to run a model in an environment whose
observation vector differs. That stops the classic silent bug of feeding a model 11
inputs it was not trained on, or the right number of inputs in a different order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

if TYPE_CHECKING:
    from neurodrive.env.driving_env import DrivingEnv


class ModelError(ValueError):
    """A saved model is missing, has no metadata, or doesn't fit the environment."""


def meta_path(model_path: Path) -> Path:
    model_path = Path(model_path)
    stem = model_path.name[:-4] if model_path.name.endswith(".zip") else model_path.name
    return model_path.with_name(stem + ".meta.json")


def write_model_meta(model_path: Path, meta: dict[str, Any]) -> None:
    meta_path(model_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_model_meta(model_path: Path) -> dict[str, Any]:
    p = meta_path(model_path)
    if not p.is_file():
        raise ModelError(
            f"no metadata next to the model (expected {p.name}); models must be saved by "
            "`neurodrive train` so their training setup is known"
        )
    meta: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return meta


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


def check_compatible(meta: dict[str, Any], env: DrivingEnv) -> None:
    """Raise if the environment's observation vector differs from the model's."""
    trained = meta.get("observation", {})
    current = env.obs_builder.spec.to_dict()
    if trained.get("names") != current["names"]:
        raise ModelError(
            "observation mismatch: model was trained on "
            f"{trained.get('version')} {trained.get('names')}, environment produces "
            f"{current['version']} {current['names']}"
        )
