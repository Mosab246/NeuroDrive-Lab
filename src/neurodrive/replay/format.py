"""The replay file: one recorded episode, everything the viewer needs and nothing else.

A replay is a single ``.npz`` file holding

* per-step arrays (step 0 is the state right after reset, so arrays have ``steps + 1``
  rows, except ``actions`` which has ``steps``): car pose, speed, steering angle, lap
  progress, and what the distance sensors saw;
* a JSON ``meta`` string: who drove (name, kind, model), where (track, scenario seed,
  start mode), the full environment config, and how it ended.

The viewer only *plays back* these arrays: it never loads a model, so a race between eight
cars runs at a steady frame rate. Because the simulator is deterministic, the stored
actions are enough to re-simulate the episode, and :func:`verify_replay` checks that doing
so reproduces every stored state exactly.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "neurodrive-replay-v1"
ARRAYS = (
    "x",
    "y",
    "heading",
    "speed",
    "steer",
    "lap_fraction",
    "laps",
    "actions",
    "ray_origin",
    "ray_dist",
    "ray_dirs",
)


class ReplayError(ValueError):
    pass


@dataclass
class Replay:
    meta: dict[str, Any]
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    steer: np.ndarray
    lap_fraction: np.ndarray
    laps: np.ndarray
    actions: np.ndarray
    ray_origin: np.ndarray
    ray_dist: np.ndarray
    ray_dirs: np.ndarray
    network_png: bytes | None = field(default=None, repr=False)
    #: NEAT only: every network node's value after each action, shape (steps, nodes), in
    #: the order of ``meta["network"]["graph"]["order"]``; drives the live network view.
    node_values: np.ndarray | None = field(default=None, repr=False)

    @property
    def steps(self) -> int:
        return len(self.actions)

    @property
    def dt(self) -> float:
        return float(self.meta["dt"])

    @property
    def duration(self) -> float:
        return self.steps * self.dt

    @property
    def label(self) -> str:
        return str(self.meta.get("label") or self.meta.get("controller_name", "?"))

    @property
    def track(self) -> str:
        return str(self.meta["track"])

    def index_at(self, t: float) -> int:
        """Row to show at playback time ``t`` seconds (holds the last row afterwards)."""
        return int(min(max(0, int(t / self.dt + 1e-9)), self.steps))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: getattr(self, name) for name in ARRAYS}
        if self.network_png is not None:
            arrays["network_png"] = np.frombuffer(self.network_png, dtype=np.uint8)
        if self.node_values is not None:
            arrays["node_values"] = self.node_values
        buf = io.BytesIO()
        np.savez_compressed(buf, meta=np.array(json.dumps(self.meta)), **arrays)
        path.write_bytes(buf.getvalue())
        return path


def load_replay(path: str | Path) -> Replay:
    path = Path(path)
    if not path.is_file():
        raise ReplayError(f"replay file not found: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("format") != FORMAT:
                raise ReplayError(f"{path.name}: not a {FORMAT} file")
            arrays = {name: data[name] for name in ARRAYS}
            png = bytes(data["network_png"]) if "network_png" in data.files else None
            nodes = data["node_values"] if "node_values" in data.files else None
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, ReplayError):
            raise
        raise ReplayError(f"{path.name}: unreadable replay ({exc})") from exc
    replay = Replay(meta=meta, network_png=png, node_values=nodes, **arrays)
    n = replay.steps
    for name in ARRAYS:
        expected = n if name == "actions" else n + 1
        if len(arrays[name]) != expected:
            raise ReplayError(f"{path.name}: array {name!r} has the wrong length")
    return replay
