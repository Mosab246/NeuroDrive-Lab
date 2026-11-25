"""A scripted, *privileged* driver: pure-pursuit steering along the known centerline.

This is NOT machine learning and NOT a fair baseline: it reads the track geometry
directly, which no learning agent will be given. Its jobs are to prove a track is
drivable, to exercise the simulator in tests and benchmarks, and later to provide a
rough reference lap time. It is always labelled "scripted (privileged)".

Pure pursuit: pick a point on the centerline a short distance ahead, and steer along
the circular arc that would pass through it. Speed: look at the tightest curve ahead
and stay below the speed at which the tyres would lose grip there. Given the other cars
on track, it also keeps a time gap to any car directly ahead instead of driving into it.
"""

from __future__ import annotations

import math

import numpy as np

from neurodrive.sim.simulation import Simulation
from neurodrive.sim.vehicle import GRAVITY

DYNAMIC_GRIP_SHARE = 0.5


class PurePursuitDriver:
    name = "pure_pursuit"
    kind = "scripted (privileged)"

    def __init__(
        self,
        lookahead_base: float = 6.0,
        lookahead_per_speed: float = 0.4,
        speed_margin: float = 0.8,
        speed_gain: float = 0.5,
        preview: float = 40.0,
        dynamic_grip_share: float = DYNAMIC_GRIP_SHARE,
    ) -> None:
        self.lookahead_base = lookahead_base
        self.lookahead_per_speed = lookahead_per_speed
        self.speed_margin = speed_margin
        self.speed_gain = speed_gain
        self.preview = preview
        self.dynamic_grip_share = dynamic_grip_share

    def act(self, sim: Simulation, others: list[np.ndarray] | None = None) -> np.ndarray:
        """``others``: corners of other cars on track (races and traffic)."""
        state, params, track = sim.state, sim.params, sim.track
        s = sim.progress.s
        lookahead = self.lookahead_base + self.lookahead_per_speed * state.speed
        target, _ = track.point_at(s + lookahead)
        dx, dy = target[0] - state.x, target[1] - state.y
        alpha = math.atan2(dy, dx) - state.heading
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        dist = max(math.hypot(dx, dy), 1e-6)
        wheel = math.atan2(2.0 * params.wheelbase * math.sin(alpha), dist)  # + = left
        steering = -wheel / params.max_steer  # action convention: + = right

        # Speed target from the tightest curvature in the preview window.
        kmax = track.max_abs_curvature(s + np.arange(0.0, self.preview, 2.0))
        if params.model == "dynamic":
            # Tyres can slide: plan corners at half the friction limit (mu * g).
            grip = self.dynamic_grip_share * params.tire_friction * GRAVITY
        else:
            grip = self.speed_margin * params.max_lateral_accel
        v_target = math.sqrt(grip / kmax) if grip > 0 and kmax > 1e-06 else params.max_speed
        v_target = min(v_target, params.max_speed)
        if others:
            v_target = min(v_target, self._follow_speed(sim, others))
        longitudinal = self.speed_gain * (v_target - state.speed)
        return np.clip(np.array([steering, longitudinal]), -1.0, 1.0)

    #: Car-following: slow down for a car in this lane ahead, keeping this time gap.
    FOLLOW_GAP_S = 0.8
    FOLLOW_STANDSTILL_M = 6.0  # center-to-center distance at which it stops
    FOLLOW_LANE_M = 2.4  # how far sideways a car counts as "in my lane"
    FOLLOW_RANGE_M = 40.0

    def _follow_speed(self, sim: Simulation, others: list[np.ndarray]) -> float:
        me = sim.corners().mean(axis=0)
        c, s = math.cos(sim.state.heading), math.sin(sim.state.heading)
        limit = float("inf")
        for corners in others:
            dx, dy = corners.mean(axis=0) - me
            ahead, side = dx * c + dy * s, -dx * s + dy * c
            if 0.0 < ahead < self.FOLLOW_RANGE_M and abs(side) < self.FOLLOW_LANE_M:
                limit = min(limit, max(0.0, (ahead - self.FOLLOW_STANDSTILL_M) / self.FOLLOW_GAP_S))
        return limit
