"""A trained model (PPO or NEAT) driving an opponent car in traffic.

Used for self-play: an agent is trained against earlier learned drivers instead of only
scripted ones. The opponent sees exactly what it saw in its own training (its sensors, with
the other cars in them) and drives its own car in the shared race world.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from neurodrive.agents.model_io import load_controller, read_model_meta
from neurodrive.env.config import EnvConfig
from neurodrive.env.observations import ObservationBuilder
from neurodrive.sim.simulation import Simulation


@cache
def _load(path: str) -> tuple[Any, EnvConfig]:
    """One copy of each opponent model per process (environments run in subprocesses)."""
    try:
        import torch

        torch.set_num_threads(1)  # many environments per machine: no thread oversubscription
    except ImportError:  # pragma: no cover - torch is a dependency
        pass
    meta = read_model_meta(Path(path))
    return load_controller(path), EnvConfig.model_validate(meta["env_config"])


def check_opponent(path: str, env: EnvConfig) -> None:
    """Opponents must have been trained with the same physics as the environment they join."""
    _, own = _load(path)
    if own.physics.model != env.physics.model:
        raise ValueError(
            f"opponent {path} was trained with {own.physics.model} physics, "
            f"this environment uses {env.physics.model}"
        )


class _NumpyPolicy:
    """A PPO policy's mean action in plain numpy: the same numbers as
    ``model.predict(obs, deterministic=True)`` (see the web-port parity test), ~20x faster
    for one observation at a time, which is what an opponent car needs."""

    def __init__(self, controller: Any) -> None:
        from neurodrive.web_export import controller_data

        data = controller_data(controller, "opponent")
        if data["activation"] not in ("Tanh", "ReLU"):
            raise ValueError(f"unsupported activation {data['activation']}")
        self.relu = data["activation"] == "ReLU"
        self.layers = [(np.array(L["w"]), np.array(L["b"])) for L in data["layers"]]

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        for w, b in self.layers[:-1]:
            x = w @ x + b
            x = np.maximum(x, 0.0) if self.relu else np.tanh(x)
        w, b = self.layers[-1]
        return np.clip(w @ x + b, -1.0, 1.0)


class LearnedDriver:
    """Drives ``sim`` with a saved model, like :class:`PurePursuitDriver` drives with rules."""

    def __init__(self, path: str, sim: Simulation) -> None:
        self.path = path
        self.controller, cfg = _load(path)
        self.policy = _NumpyPolicy(self.controller) if self.controller.kind == "ppo" else None
        self.builder = ObservationBuilder(sim, cfg.sensors, cfg.observation)
        self.controller.reset(0)

    def act(self, sim: Simulation, others: list[np.ndarray] | None = None) -> np.ndarray:
        assert sim is self.builder.sim
        obs = self.builder.build(others)
        if self.policy is not None:
            return self.policy.act(obs)
        return np.asarray(self.controller.act(obs), dtype=np.float64)
