"""Raycast distance sensors: what the car can "see".

Each ray starts at the center of the car body and points at a fixed angle relative to the
car's heading. It reports the distance to the first track edge it hits, capped at
``max_range``. Like the steering action, **positive angles point right** (clockwise):
``-90`` is straight out of the left side, ``0`` straight ahead, ``+90`` the right side.

Rays are cast against both track edges, but only against edge segments within an
arc-length window around the car. A ray leaving the car's own corridor has to cross one of
the nearby edges first, so distant parts of the track can never be "seen through" a wall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from neurodrive.sim.geometry import FloatArray
from neurodrive.sim.track import Track

DEFAULT_RAY_ANGLES_DEG: tuple[float, ...] = (-90, -60, -40, -20, 0, 20, 40, 60, 90)


@dataclass(frozen=True)
class RayHits:
    distances: FloatArray  # (R,) meters, capped at max_range
    directions: FloatArray  # (R, 2) unit vectors in world coordinates
    origin: FloatArray  # (2,)

    def endpoints(self) -> FloatArray:
        return self.origin + self.directions * self.distances[:, None]


class RaySensor:
    def __init__(
        self,
        track: Track,
        angles_deg: tuple[float, ...] = DEFAULT_RAY_ANGLES_DEG,
        max_range: float = 50.0,
    ) -> None:
        if max_range <= 0:
            raise ValueError("max_range must be positive")
        if not angles_deg:
            raise ValueError("at least one ray angle is required")
        self.track = track
        self.max_range = float(max_range)
        # Positive angle = clockwise (right), so the world-frame offset is -angle.
        self.offsets = -np.radians(np.asarray(angles_deg, dtype=np.float64))
        n = track.num_points
        spacing = track.length / n
        self.window = int(min(n // 2, math.ceil(1.5 * self.max_range / spacing) + 10))
        w = self.window
        pad = (np.r_[np.arange(n - w, n), np.arange(n), np.arange(w)] % n).astype(np.int64)
        starts, vecs = [], []
        for edge in (track.left_edge, track.right_edge):
            d = np.roll(edge, -1, axis=0) - edge
            starts.append(edge[pad])
            vecs.append(d[pad])
        # Shape (2, n + 2w, 2): edge, padded segment index, xy.
        self._a = np.stack(starts)
        self._e = np.stack(vecs)
        self._full = 2 * w + 1 >= n

    @property
    def num_rays(self) -> int:
        return len(self.offsets)

    def cast(self, origin: FloatArray, heading: float, segment: int) -> RayHits:
        """Cast all rays from ``origin``. ``segment`` is the car's current centerline segment
        (from the progress tracker), used to pick nearby edge segments."""
        n, w = self.track.num_points, self.window
        if self._full:
            sl = slice(w, w + n)
        else:
            start = int(segment) % n
            sl = slice(start, start + 2 * w + 1)
        a = self._a[:, sl].reshape(-1, 2)  # (S, 2)
        e = self._e[:, sl].reshape(-1, 2)
        ang = heading + self.offsets
        d = np.column_stack([np.cos(ang), np.sin(ang)])  # (R, 2)

        ao = a - origin  # (S, 2)
        # Solve origin + t*d = a + u*e  for t (along ray) and u (along segment).
        denom = d[:, 0:1] * e[None, :, 1] - d[:, 1:2] * e[None, :, 0]  # (R, S)
        t_num = ao[None, :, 0] * e[None, :, 1] - ao[None, :, 1] * e[None, :, 0]
        u_num = ao[None, :, 0] * d[:, 1:2] - ao[None, :, 1] * d[:, 0:1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = t_num / denom
            u = u_num / denom
        valid = (np.abs(denom) > 1e-12) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
        t = np.where(valid, t, np.inf)
        dist = np.minimum(t.min(axis=1), self.max_range)
        return RayHits(distances=dist, directions=d, origin=np.asarray(origin, dtype=float))
