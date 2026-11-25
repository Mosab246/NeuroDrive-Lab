"""The dynamic (tyre-slip) vehicle model: behaviour a driver would recognise."""

import math

import pytest

from neurodrive.sim.vehicle import VehicleParams, VehicleState, step_vehicle

DYN = VehicleParams(model="dynamic")
DT = 1 / 60


def drive(state, steer, pedal, seconds, params=DYN):
    for _ in range(round(seconds / DT)):
        step_vehicle(state, steer, pedal, params, DT)
    return state


def slip_deg(s):
    return math.degrees(math.atan2(s.lateral_speed, max(s.speed, 1e-6)))


def test_straight_line_stays_straight():
    s = drive(VehicleState(0, 0, 0), 0.0, 1.0, 5)
    assert s.speed > 15 and abs(s.y) < 1e-9 and s.yaw_rate == 0.0


def test_positive_steering_turns_right():
    s = drive(VehicleState(0, 0, 0, speed=15), 0.2, 0.2, 1)
    assert s.yaw_rate < 0  # clockwise
    assert s.heading < 0


def test_gentle_corner_has_little_slip_and_some_understeer():
    s = drive(VehicleState(0, 0, 0, speed=15), 0.2, 0.2, 3)
    kinematic_rate = s.speed * math.tan(0.2 * DYN.max_steer) / DYN.wheelbase
    assert abs(slip_deg(s)) < 3
    assert 0.6 * kinematic_rate < abs(s.yaw_rate) < kinematic_rate  # turns a bit less


def test_too_fast_with_throttle_spins_out():
    s = VehicleState(0, 0, 0, speed=25)
    peak = 0.0
    for _ in range(60):
        drive(s, 0.6, 1.0, DT)
        peak = max(peak, abs(slip_deg(s)))
    assert peak > 20  # sliding sideways: the rear has let go


def test_braking_stops_without_moving_backwards():
    s = drive(VehicleState(0, 0, 0, speed=25), 0.0, -1.0, 4)
    assert s.speed == 0.0 and s.lateral_speed == 0.0 and s.yaw_rate == 0.0


def test_braking_is_limited_by_tyre_grip():
    s = drive(VehicleState(0, 0, 0, speed=30), 0.0, -1.0, 1)
    decel = 30 - s.speed  # m/s lost in 1 s
    assert decel < DYN.max_brake  # the kinematic model would brake at 16 m/s^2
    assert decel < DYN.tire_friction * 9.81 + 1.0


def test_less_friction_slides_more():
    icy = VehicleParams(model="dynamic", tire_friction=0.5)
    a = drive(VehicleState(0, 0, 0, speed=18), 0.4, 0.3, 1.5)
    b = drive(VehicleState(0, 0, 0, speed=18), 0.4, 0.3, 1.5, icy)
    assert abs(slip_deg(b)) > abs(slip_deg(a))


def test_deterministic():
    a = drive(VehicleState(0, 0, 0, speed=20), 0.5, 0.8, 3)
    b = drive(VehicleState(0, 0, 0, speed=20), 0.5, 0.8, 3)
    assert a == b


def test_kinematic_default_ignores_the_new_state():
    s = drive(VehicleState(0, 0, 0, speed=15), 0.3, 0.5, 2, VehicleParams())
    assert s.lateral_speed == 0.0 and s.yaw_rate == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "drift"},
        {"model": "dynamic", "mass": 0},
        {"model": "dynamic", "cg_to_front": 3.0},
        {"model": "dynamic", "max_tire_usage": 0},
        {"model": "dynamic", "tire_friction": -1},
    ],
)
def test_bad_parameters_are_refused(kwargs):
    with pytest.raises(ValueError):
        VehicleParams(**kwargs)
