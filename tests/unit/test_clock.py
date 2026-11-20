import pytest

from neurodrive.ui.clock import SPEEDS, FixedStepClock

DT = 1 / 30


@pytest.mark.parametrize("speed", SPEEDS)
def test_total_steps_follow_simulated_time_not_frame_rate(speed):
    clock = FixedStepClock(DT)
    clock.set_speed(speed)
    steps = sum(clock.advance(1 / 60) for _ in range(600))  # 10 real seconds at 60 fps
    assert steps == pytest.approx(10 * speed / DT, abs=1)


def test_uneven_frames_give_the_same_step_count():
    a, b = FixedStepClock(DT), FixedStepClock(DT)
    even = sum(a.advance(0.02) for _ in range(500))
    uneven = sum(b.advance(t) for t in [0.005, 0.035] * 250)
    assert even == uneven == 300


def test_backlog_is_capped():
    clock = FixedStepClock(DT, max_steps_per_frame=10)
    assert clock.advance(5.0) == 10
    assert clock.accumulator == 0.0


def test_speed_bounds():
    clock = FixedStepClock(DT)
    for _ in range(20):
        clock.faster()
    assert clock.speed == max(SPEEDS)
    for _ in range(20):
        clock.slower()
    assert clock.speed == min(SPEEDS)
