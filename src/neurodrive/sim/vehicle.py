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

With ``model="dynamic"`` the car instead follows a **dynamic bicycle model with tyre
slip** (see :func:`step_dynamic`): the tyres produce sideways force from their slip
angle up to a friction limit, the rear wheels drive the car, and throttle or brake use up
grip that cornering then lacks. The car can understeer, oversteer, slide and spin. This
is the "hard" physics; the kinematic model above stays the default.

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
    # --- dynamic model only (model="dynamic") ---
    model: str = "kinematic"  # "kinematic" (default) or "dynamic" (tyre slip)
    mass: float = 1200.0  # kg
    yaw_inertia: float = 1600.0  # kg m^2
    cg_to_front: float = 1.2  # m from the centre of mass to the front axle
    cornering_front: float = 70000.0  # N/rad, tyre sideways stiffness (front axle)
    cornering_rear: float = 80000.0  # N/rad (rear axle)
    tire_friction: float = 1.1  # friction coefficient mu; total tyre force <= mu * load
    brake_front_share: float = 0.6  # fraction of braking force on the front axle
    # Traction control / ABS: driving and braking may use at most this share of a tyre's
    # grip, so some sideways grip is always left (1.0 = none, wheels can fully saturate).
    max_tire_usage: float = 0.85

    def __post_init__(self) -> None:
        if self.model not in ("kinematic", "dynamic"):
            raise ValueError("VehicleParams.model must be 'kinematic' or 'dynamic'")
        for name in ("mass", "yaw_inertia", "cornering_front", "cornering_rear"):
            if getattr(self, name) <= 0:
                raise ValueError(f"VehicleParams.{name} must be positive")
        if not 0 < self.cg_to_front < self.wheelbase:
            raise ValueError("VehicleParams.cg_to_front must be between 0 and the wheelbase")
        if not 0 < self.max_tire_usage <= 1:
            raise ValueError("VehicleParams.max_tire_usage must be in (0, 1]")
        if self.tire_friction <= 0 or not 0 <= self.brake_front_share <= 1:
            raise ValueError("tire_friction must be positive, brake_front_share in [0, 1]")
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
    lateral_speed: float = 0.0  # m/s sideways at the centre of mass, + = left (dynamic only)
    yaw_rate: float = 0.0  # rad/s, + = counter-clockwise (dynamic model only)

    def copy(self) -> VehicleState:
        return VehicleState(
            self.x, self.y, self.heading, self.speed, self.steer, self.lateral_speed, self.yaw_rate
        )


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
    if params.model == "dynamic":
        step_dynamic(state, longitudinal, params, dt)
        return

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


#: Below this forward speed the dynamic model hands over to kinematic rolling: slip angles
#: are undefined at standstill and the tyre equations become numerically stiff.
DYNAMIC_MIN_SPEED = 2.0
DYNAMIC_ITERATIONS = 4  # internal integration steps per call, for numerical stability
GRAVITY = 9.81


def _clip(v: float, limit: float) -> float:
    return min(limit, max(-limit, v))


def step_dynamic(state: VehicleState, longitudinal: float, p: VehicleParams, dt: float) -> None:
    """Dynamic bicycle model with tyre slip (body frame, rear-wheel drive).

    * Each axle's tyres push sideways in proportion to their **slip angle** (the angle
      between where the wheel points and where it is actually moving), up to a friction
      limit ``mu * load``.
    * Throttle drives the rear wheels; braking is shared between the axles. Force spent
      speeding up or slowing down comes out of the same friction budget ("friction
      circle"), so hard throttle in a corner makes the rear slide out (oversteer) and hard
      braking in a corner makes the front push wide (understeer). Like traction control
      and ABS, ``max_tire_usage`` keeps part of the grip for steering.
    * ``state.speed`` is the forward speed; ``lateral_speed`` and ``yaw_rate`` are the
      sideways slide and rotation rate. The steering angle is already updated.
    """
    lf = p.cg_to_front
    lr = p.wheelbase - lf
    m = p.mass
    load_f = m * GRAVITY * lr / p.wheelbase
    load_r = m * GRAVITY * lf / p.wheelbase
    cap_f, cap_r = p.tire_friction * load_f, p.tire_friction * load_r
    use_f, use_r = p.max_tire_usage * cap_f, p.max_tire_usage * cap_r
    delta = -state.steer  # standard convention: positive = left
    cos_d, sin_d = math.cos(delta), math.sin(delta)
    h = dt / DYNAMIC_ITERATIONS
    for _ in range(DYNAMIC_ITERATIONS):
        vx, vy, r = state.speed, state.lateral_speed, state.yaw_rate
        # Longitudinal tyre forces, each within its axle's friction limit.
        if longitudinal >= 0.0:
            fx_f, fx_r = 0.0, min(longitudinal * p.max_accel * m, use_r)
        elif vx > 0.0:
            brake = -longitudinal * p.max_brake * m
            fx_f = -min(brake * p.brake_front_share, use_f)
            fx_r = -min(brake * (1.0 - p.brake_front_share), use_r)
        else:
            fx_f = fx_r = 0.0
        resist = (p.drag * vx * vx + p.rolling_resistance) * m if vx > 0.0 else 0.0
        if vx < DYNAMIC_MIN_SPEED:
            # Kinematic rolling at low speed: no slip, sideways motion follows the steering.
            ax = (fx_f + fx_r - resist) / m
            vx = min(p.max_speed, max(0.0, vx + ax * h))
            r = vx * math.tan(delta) / p.wheelbase
            vy = r * lr
        else:
            slip_f = math.atan2(vy + lf * r, vx) - delta
            slip_r = math.atan2(vy - lr * r, vx)
            room_f = math.sqrt(max(cap_f * cap_f - fx_f * fx_f, 0.0))
            room_r = math.sqrt(max(cap_r * cap_r - fx_r * fx_r, 0.0))
            fy_f = _clip(-p.cornering_front * slip_f, room_f)
            fy_r = _clip(-p.cornering_rear * slip_r, room_r)
            ax = (fx_r + fx_f * cos_d - fy_f * sin_d - resist) / m + vy * r
            ay = (fy_r + fx_f * sin_d + fy_f * cos_d) / m - vx * r
            yaw_acc = (lf * (fy_f * cos_d + fx_f * sin_d) - lr * fy_r) / p.yaw_inertia
            vx = min(p.max_speed, max(0.0, vx + ax * h))
            vy += ay * h
            r += yaw_acc * h
            if vx == 0.0:  # stopped: no sliding on the spot
                vy = r = 0.0
        state.speed, state.lateral_speed, state.yaw_rate = vx, vy, r
        state.heading += r * h
        state.heading = (state.heading + math.pi) % (2.0 * math.pi) - math.pi
        # Move the rear-axle reference point with its own velocity (body -> world frame).
        bvx, bvy = vx, vy - lr * r
        c, s = math.cos(state.heading), math.sin(state.heading)
        state.x += (bvx * c - bvy * s) * h
        state.y += (bvx * s + bvy * c) * h


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
