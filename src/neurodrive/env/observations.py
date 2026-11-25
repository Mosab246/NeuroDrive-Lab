"""Turns simulator state into the fixed-size vector a learning agent receives.

Default vector (``obs-v1``, 11 numbers, all from the car's own point of view):

====  ======================  =================  ========
idx   feature                 normalization      range
====  ======================  =================  ========
0-8   ray distances           d / max_range      [0, 1]
9     speed                   v / max_speed      [0, 1]
10    current wheel angle     steer / max_steer  [-1, 1]
====  ======================  =================  ========

Rays are ordered by angle, left to right (-90 deg first). With ``route_features`` three
**privileged** features follow (computed from the track's centerline, which a real car's
sensors would not give it):

====  =========================  ==================  ========
11    sin(heading error)         --                  [-1, 1]
12    cos(heading error)         --                  [-1, 1]
13    lateral offset             lateral / half_w    [-1, 1]
====  =========================  ==================  ========

Heading error is the car's heading minus the local track direction (positive = pointing
left of the track). Lateral offset is positive left of the centerline.

Why normalize: neural networks learn best when inputs have similar, small ranges.
A raw speed of 35 next to a ray value of 0.4 would make the network's early learning
dominated by the speed input just because its numbers are bigger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from neurodrive.env.config import ObservationConfig, SensorConfig
from neurodrive.sim.sensors import RayHits, RaySensor
from neurodrive.sim.simulation import Simulation
from neurodrive.sim.vehicle import body_center

OBS_VERSION = "obs-v1"
LATERAL_SPEED_SCALE = 10.0  # m/s mapped to 1.0 (motion features)
YAW_RATE_SCALE = 3.0  # rad/s mapped to 1.0


@dataclass(frozen=True)
class ObservationSpec:
    version: str
    names: tuple[str, ...]
    low: np.ndarray
    high: np.ndarray
    privileged: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.names)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "names": list(self.names),
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "privileged": list(self.privileged),
        }


class ObservationBuilder:
    def __init__(self, sim: Simulation, sensors: SensorConfig, config: ObservationConfig):
        self.sim = sim
        self.config = config
        self.sensor = RaySensor(sim.track, sensors.angles_deg, sensors.max_range)
        names = [f"ray_{a:+g}" for a in sensors.angles_deg] + ["speed", "steer"]
        low = [0.0] * len(sensors.angles_deg) + [0.0, -1.0]
        high = [1.0] * (len(sensors.angles_deg) + 2)
        privileged: list[str] = []
        if config.route_features:
            extra = ["heading_error_sin", "heading_error_cos", "lateral_offset"]
            names += extra
            privileged += extra
            low += [-1.0, -1.0, -1.0]
            high += [1.0, 1.0, 1.0]
        if config.motion_features:
            names += ["lateral_speed", "yaw_rate"]
            low += [-1.0, -1.0]
            high += [1.0, 1.0]
        suffix = ("+route" if config.route_features else "") + (
            "+motion" if config.motion_features else ""
        )
        self.spec = ObservationSpec(
            version=OBS_VERSION + suffix,
            names=tuple(names),
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            privileged=tuple(privileged),
        )
        self.last_hits: RayHits | None = None

    def build(self, obstacles: list[np.ndarray] | None = None) -> np.ndarray:
        """The observation for the car's current state. ``obstacles``: other cars' corner
        arrays, which the distance sensors then see (used in races)."""
        sim = self.sim
        state, params = sim.state, sim.params
        origin = np.array(body_center(state, params))
        hits = self.sensor.cast(origin, state.heading, sim.progress.segment, obstacles)
        self.last_hits = hits
        parts = [
            hits.distances / self.sensor.max_range,
            [state.speed / params.max_speed, state.steer / params.max_steer],
        ]
        if self.config.route_features:
            tangent = sim.track.tangents[sim.progress.segment]
            err = state.heading - math.atan2(tangent[1], tangent[0])
            half_width = sim.track.half_width[sim.progress.segment]
            parts.append([math.sin(err), math.cos(err), sim.progress.lateral / half_width])
        if self.config.motion_features:
            parts.append(
                [state.lateral_speed / LATERAL_SPEED_SCALE, state.yaw_rate / YAW_RATE_SCALE]
            )
        obs = np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])
        np.clip(obs, self.spec.low, self.spec.high, out=obs)
        out = obs.astype(np.float32)
        if not np.all(np.isfinite(out)):  # never silently hand NaN to a learner
            raise RuntimeError(f"non-finite observation: {out.tolist()}")
        return out
