"""Kinematic bicycle model of a car.

Picture the car as a bicycle: one rear wheel and one steerable front wheel, a
``wheelbase`` apart. The reference point is the middle of the rear axle. Each step:

1. The wheels turn toward the commanded angle, but no faster than ``steer_rate``.
2. Throttle or brake changes speed; drag (``drag * v^2``) and rolling resistance slow it.
   Speed stays in ``[0, max_speed]``: there is no reverse gear.
3. The heading changes at ``v / wheelbase * tan(steer)``. This is proportional to speed,
   so a stationary car cannot rotate on the spot.
4. A grip limit caps how sharply the car can turn at speed: lateral acceleration
   ``v^2 / turn_radius`` may not exceed ``max_lateral_accel``. Fast cars must slow down
   for corners, which is the core skill a driver (human or AI) has to learn.

Sign conventions: the world is y-up, heading is counter-clockwise from +x, and a
**positive steering command turns right** (clockwise), matching the action interface in
the spec (+1 = full right).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VehicleParams:
    """Physical parameters. Units: meters, seconds, radians."""

    wheelbase: float = 2.6
    length: float = 4.4
    width: float = 1.9
    max_speed: float = 35.0  # m/s (126 km/h)
    max_accel: float = 8.0  # m/s^2 at full throttle
    max_brake: float = 16.0  # m/s^2 at full brake
    drag: float = 0.003  # 1/m; deceleration = drag * v^2
    rolling_resistance: float = 0.4  # m/s^2 whenever moving
    max_steer: float = 0.5  # rad, ~29 degrees at the front wheel
    steer_rate: float = 2.5  # rad/s the wheel can turn
    max_lateral_accel: float = 12.0  # m/s^2, ~1.2 g; set <= 0 to disable the grip limit

    def __post_init__(self) -> None:
        for name in ("wheelbase", "length", "width", "max_speed", "max_steer", "steer_rate"):
            if getattr(self, name) <= 0:
                raise ValueError(f"VehicleParams.{name} must be positive")
        for name in ("max_accel", "max_brake", "drag", "rolling_resistance"):
            if getattr(self, name) < 0:
                raise ValueError(f"VehicleParams.{name} must not be negative")


@dataclass(slots=True)
class VehicleState:
    """Mutable car state. ``x, y`` is the middle of the rear axle."""

    x: float
    y: float
    heading: float
    speed: float = 0.0
    steer: float = 0.0  # actual front-wheel angle, rad; positive = right

    def copy(self) -> VehicleState:
        return VehicleState(self.x, self.y, self.heading, self.speed, self.steer)


def clamp_action(action: np.ndarray | tuple[float, float]) -> tuple[float, float]:
    """Validate an action ``[steering, longitudinal]`` and clamp it into ``[-1, 1]``.

    Raises ``ValueError`` for the wrong shape or non-finite values instead of guessing.
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape != (2,):
        raise ValueError(f"action must have 2 values [steering, throttle/brake], got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"action contains NaN or infinity: {a.tolist()}")
    return float(min(1.0, max(-1.0, a[0]))), float(min(1.0, max(-1.0, a[1])))


def grip_steer_limit(params: VehicleParams, speed: float) -> float:
    """Largest usable wheel angle at ``speed`` before the tyres would lose grip."""
    if params.max_lateral_accel <= 0 or speed < 1e-6:
        return params.max_steer
    # turn radius R = wheelbase / tan(steer); lateral accel = v^2 / R  <=  a_max
    limit = math.atan(params.wheelbase * params.max_lateral_accel / (speed * speed))
    return min(params.max_steer, limit)


def step_vehicle(
    state: VehicleState, steering: float, longitudinal: float, params: VehicleParams, dt: float
) -> None:
    """Advance ``state`` in place by ``dt`` seconds (semi-implicit Euler).

    ``steering`` and ``longitudinal`` must already be clamped to ``[-1, 1]``.
    """
    # 1. Steering actuator with a rate limit.
    target = steering * params.max_steer
    max_change = params.steer_rate * dt
    state.steer += min(max_change, max(-max_change, target - state.steer))

    # 2. Longitudinal dynamics.
    if longitudinal >= 0.0:
        accel = longitudinal * params.max_accel
    else:
        accel = longitudinal * params.max_brake  # negative
    v = state.speed
    if v > 0.0:
        accel -= params.drag * v * v + params.rolling_resistance
    v = min(params.max_speed, max(0.0, v + accel * dt))
    state.speed = v

    # 3-4. Heading change, limited by grip. Positive steer turns clockwise (right).
    limit = grip_steer_limit(params, v)
    effective = min(limit, max(-limit, state.steer))
    state.heading -= v / params.wheelbase * math.tan(effective) * dt
    state.heading = (state.heading + math.pi) % (2.0 * math.pi) - math.pi

    # Position update uses the new speed and heading (semi-implicit Euler).
    state.x += v * math.cos(state.heading) * dt
    state.y += v * math.sin(state.heading) * dt


def body_center(state: VehicleState, params: VehicleParams) -> tuple[float, float]:
    """Geometric center of the car body (half a wheelbase ahead of the rear axle)."""
    half = 0.5 * params.wheelbase
    return state.x + half * math.cos(state.heading), state.y + half * math.sin(state.heading)


def body_corners(state: VehicleState, params: VehicleParams) -> np.ndarray:
    """The car rectangle's corners, shape ``(4, 2)``: front-left, front-right, rear-right,
    rear-left."""
    cx, cy = body_center(state, params)
    c, s = math.cos(state.heading), math.sin(state.heading)
    hl, hw = 0.5 * params.length, 0.5 * params.width
    fwd = np.array([c, s])
    left = np.array([-s, c])
    center = np.array([cx, cy])
    return np.array(
        [
            center + hl * fwd + hw * left,
            center + hl * fwd - hw * left,
            center - hl * fwd - hw * left,
            center - hl * fwd + hw * left,
        ]
    )
