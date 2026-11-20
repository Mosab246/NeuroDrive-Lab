"""The scripted pure-pursuit driver proves the oval is drivable and laps are counted."""

import numpy as np
import pytest

from neurodrive.agents.pure_pursuit import PurePursuitDriver
from neurodrive.sim.simulation import SimConfig, Simulation, Termination
from neurodrive.sim.track import load_track


@pytest.fixture(scope="module")
def oval():
    return load_track("beginner_oval")


def test_autopilot_completes_two_clean_laps(oval):
    sim = Simulation(oval, config=SimConfig(laps_to_finish=2, max_time=300.0))
    driver = PurePursuitDriver()
    while not sim.done:
        sim.step(driver.act(sim))
    assert sim.termination is Termination.LAPS_COMPLETE
    assert sim.progress.laps == 2
    assert len(sim.lap_times) == 2
    assert all(t > 0 for t in sim.lap_times)
    # The second lap has a flying start, so it should not be slower than the first.
    assert sim.lap_times[1] <= sim.lap_times[0]
    assert sum(sim.lap_times) == pytest.approx(sim.time)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_autopilot_handles_perturbed_starts(oval, seed):
    sim = Simulation(oval, config=SimConfig(laps_to_finish=1, max_time=120.0))
    sim.reset(seed=seed, perturb=True)
    driver = PurePursuitDriver()
    while not sim.done:
        sim.step(driver.act(sim))
    assert sim.termination is Termination.LAPS_COMPLETE


def test_ten_thousand_headless_steps(oval):
    """Runs with no display at all; resets whenever a run ends."""
    sim = Simulation(oval, config=SimConfig(laps_to_finish=3))
    driver = PurePursuitDriver()
    for _ in range(10_000):
        if sim.done:
            sim.reset()
        sim.step(driver.act(sim))
    assert sim.step_count > 0


def test_keyboard_style_digital_inputs_can_complete_a_lap(oval):
    """A human on a keyboard can only send -1, 0 or +1 on each axis. Quantise the
    autopilot's commands the same way: if this laps cleanly, the car is drivable with
    keys (the steering rate limit smooths the on/off inputs)."""
    sim = Simulation(oval, config=SimConfig(laps_to_finish=1, max_time=120.0))
    driver = PurePursuitDriver()

    def key(value: float, deadband: float) -> float:
        return 0.0 if abs(value) < deadband else float(np.sign(value))

    # The oval's 40 m curves need only ~13% steering, so the steering deadband must be
    # small: like a human, the controller "taps" the key and the wheel angle averages out.
    while not sim.done:
        steer, pedal = driver.act(sim)
        sim.step((key(steer, 0.05), key(pedal, 0.3)))
    assert sim.termination is Termination.LAPS_COMPLETE
