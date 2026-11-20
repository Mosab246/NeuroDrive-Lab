import math

import numpy as np
import pytest

from neurodrive.sim.vehicle import (
    VehicleParams,
    VehicleState,
    body_corners,
    clamp_action,
    grip_steer_limit,
    step_vehicle,
)

P = VehicleParams()
DT = 1 / 60


def run(state: VehicleState, steering: float, longitudinal: float, seconds: float) -> None:
    for _ in range(round(seconds / DT)):
        step_vehicle(state, steering, longitudinal, P, DT)


def test_stationary_car_cannot_rotate_in_place():
    s = VehicleState(0, 0, 0.3)
    run(s, 1.0, 0.0, 2.0)
    assert s.heading == pytest.approx(0.3)
    assert (s.x, s.y) == (0, 0)
    assert s.steer == pytest.approx(P.max_steer)  # wheels turned, car did not


def test_throttle_accelerates_up_to_max_speed_and_no_further():
    s = VehicleState(0, 0, 0)
    run(s, 0.0, 1.0, 1.0)
    assert 0 < s.speed < P.max_speed
    run(s, 0.0, 1.0, 30.0)
    assert s.speed == pytest.approx(P.max_speed)
    assert s.x > 0 and s.y == pytest.approx(0.0)  # drives straight along heading


def test_braking_never_produces_negative_speed():
    s = VehicleState(0, 0, 0, speed=5.0)
    run(s, 0.0, -1.0, 3.0)
    assert s.speed == 0.0


def test_coasting_slows_down():
    s = VehicleState(0, 0, 0, speed=20.0)
    run(s, 0.0, 0.0, 2.0)
    assert 0 < s.speed < 20.0


def test_positive_steering_turns_right_clockwise():
    s = VehicleState(0, 0, 0, speed=10.0)
    run(s, 1.0, 0.0, 0.5)
    assert s.heading < 0  # clockwise in a y-up world
    assert s.y < 0  # heading east and turning right means drifting south


def test_negative_steering_turns_left():
    s = VehicleState(0, 0, 0, speed=10.0)
    run(s, -1.0, 0.0, 0.5)
    assert s.heading > 0 and s.y > 0


def test_steering_rate_limit():
    s = VehicleState(0, 0, 0)
    step_vehicle(s, 1.0, 0.0, P, DT)
    assert s.steer == pytest.approx(P.steer_rate * DT)
    run(s, 1.0, 0.0, 1.0)
    assert s.steer == pytest.approx(P.max_steer)


def test_grip_limit_widens_turn_radius_at_speed():
    assert grip_steer_limit(P, 0.0) == P.max_steer
    fast = grip_steer_limit(P, 30.0)
    assert fast < P.max_steer
    radius = P.wheelbase / math.tan(fast)
    assert 30.0**2 / radius == pytest.approx(P.max_lateral_accel)


def test_grip_limit_can_be_disabled():
    p = VehicleParams(max_lateral_accel=0.0)
    assert grip_steer_limit(p, 30.0) == p.max_steer


@pytest.mark.parametrize(
    ("raw", "expected"), [((2.0, -3.0), (1.0, -1.0)), ((0.25, 0.5), (0.25, 0.5))]
)
def test_actions_are_clamped(raw, expected):
    assert clamp_action(raw) == expected


@pytest.mark.parametrize("bad", [(float("nan"), 0.0), (0.0, float("inf"))])
def test_non_finite_actions_are_rejected(bad):
    with pytest.raises(ValueError, match="NaN or infinity"):
        clamp_action(bad)


def test_wrong_action_shape_is_rejected():
    with pytest.raises(ValueError, match="2 values"):
        clamp_action(np.zeros(3))


def test_body_corners_form_the_car_rectangle():
    c = body_corners(VehicleState(0, 0, 0), P)
    xs, ys = c[:, 0], c[:, 1]
    assert xs.max() - xs.min() == pytest.approx(P.length)
    assert ys.max() - ys.min() == pytest.approx(P.width)
    # Body center is half a wheelbase ahead of the rear axle.
    assert xs.mean() == pytest.approx(P.wheelbase / 2)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        VehicleParams(wheelbase=0)
