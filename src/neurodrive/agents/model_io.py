"""Saving, describing and loading trained models, for every algorithm.

Every saved model file (``X.zip`` for PPO, ``X.pkl`` for NEAT) has a sidecar
``X.meta.json`` describing what it was trained on: algorithm, observation spec,
environment config, training step. Loading requires the sidecar, and
:func:`check_compatible` refuses to run a model in an environment whose observation vector
differs. That stops the classic silent bug of feeding a model inputs it was not trained on,
or the right number of inputs in a different order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neurodrive.agents.base import Controller
    from neurodrive.env.driving_env import DrivingEnv


class ModelError(ValueError):
    """A saved model is missing, has no metadata, or doesn't fit the environment."""


def meta_path(model_path: Path) -> Path:
    return Path(model_path).with_suffix(".meta.json")


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


def load_controller(path: str | Path) -> Controller:
    """Load any saved model (PPO or NEAT) as a Controller, based on its metadata."""
    path = Path(path)
    if not path.is_file():
        raise ModelError(f"model file not found: {path}")
    algorithm = read_model_meta(path).get("algorithm")
    if algorithm == "ppo":
        from neurodrive.agents.ppo_agent import PPOController

        return PPOController.load(path)
    if algorithm == "neat":
        from neurodrive.agents.neat_agent import NEATController

        return NEATController.load(path)
    raise ModelError(f"{path.name}: unknown algorithm {algorithm!r} in its metadata")
