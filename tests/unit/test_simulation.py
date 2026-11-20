import math

import numpy as np
import pytest

from neurodrive.sim.simulation import SimConfig, Simulation, Termination
from neurodrive.sim.track import load_track


@pytest.fixture(scope="module")
def oval():
    return load_track("beginner_oval")


def trajectory(sim: Simulation, actions: np.ndarray) -> np.ndarray:
    rows = []
    for a in actions:
        if sim.done:
            break
        r = sim.step(a)
        s = sim.state
        rows.append([s.x, s.y, s.heading, s.speed, s.steer, sim.progress.distance, r.time])
    return np.array(rows)


def test_same_seed_and_actions_give_identical_trajectories(oval):
    actions = np.random.default_rng(123).uniform(-1, 1, size=(600, 2))
    actions[:, 1] = np.abs(actions[:, 1]) * 0.6  # mostly forward so the run lasts a while
    runs = []
    for _ in range(2):
        sim = Simulation(oval, seed=7)
        sim.reset(seed=7, perturb=True)
        runs.append(trajectory(sim, actions))
    assert runs[0].shape == runs[1].shape and len(runs[0]) > 10
    assert np.array_equal(runs[0], runs[1])  # exact, not approximate


def test_reset_restores_the_initial_state(oval):
    sim = Simulation(oval)
    start = sim.state.copy()
    for _ in range(50):
        sim.step((0.0, 1.0))
    sim.reset()
    assert sim.state == start
    assert sim.time == 0.0 and sim.step_count == 0 and sim.lap_times == []
    assert sim.progress.distance == pytest.approx(0.0, abs=1e-9)


def test_unperturbed_start_puts_the_body_center_on_the_start_line(oval):
    sim = Simulation(oval)
    assert sim.progress.s == pytest.approx(0.0, abs=1e-9) or sim.progress.s == pytest.approx(
        oval.length, abs=1e-9
    )
    assert abs(sim.progress.lateral) < 1e-9


def test_perturbation_is_seeded_and_bounded(oval):
    sim = Simulation(oval)
    sim.reset(seed=1, perturb=True)
    a = sim.state.copy()
    sim.reset(seed=1, perturb=True)
    assert sim.state == a
    sim.reset(seed=2, perturb=True)
    assert sim.state != a
    assert abs(sim.progress.lateral) <= sim.config.start_lateral_jitter + 1e-9
    assert abs(sim.state.heading) <= sim.config.start_heading_jitter + 1e-9


def test_steering_into_the_outer_wall_is_a_collision(oval):
    sim = Simulation(oval)
    result = None
    for _ in range(600):
        result = sim.step((1.0, 1.0))  # full right on a counter-clockwise track
        if sim.done:
            break
    assert result is not None and result.collided
    assert sim.termination is Termination.COLLISION
    assert sim.progress.laps == 0


def test_cutting_into_the_infield_is_a_collision(oval):
    sim = Simulation(oval)
    for _ in range(600):
        sim.step((-1.0, 1.0))  # full left = toward the infield
        if sim.done:
            break
    assert sim.termination is Termination.COLLISION


def test_step_after_termination_raises(oval):
    sim = Simulation(oval)
    while not sim.done:
        sim.step((1.0, 1.0))
    with pytest.raises(RuntimeError, match="reset"):
        sim.step((0.0, 0.0))


def test_time_limit_terminates(oval):
    sim = Simulation(oval, config=SimConfig(max_time=1.0))
    steps = 0
    while not sim.done:
        sim.step((0.0, 0.0))
        steps += 1
    assert sim.termination is Termination.TIME_LIMIT
    assert steps == 30
    assert sim.time == pytest.approx(1.0)


def test_parked_car_stays_parked(oval):
    sim = Simulation(oval, config=SimConfig(max_time=10.0))
    start = sim.state.copy()
    while not sim.done:
        sim.step((0.5, 0.0))
    assert (sim.state.x, sim.state.y, sim.state.heading) == (start.x, start.y, start.heading)
    assert sim.progress.laps == 0


def test_time_comes_from_step_count(oval):
    sim = Simulation(oval)
    for _ in range(300):
        sim.step((0.0, 0.2))
    assert sim.time == 300 * sim.config.dt
    assert math.isclose(sim.time, 10.0)


def test_invalid_action_raises_instead_of_being_guessed(oval):
    sim = Simulation(oval)
    with pytest.raises(ValueError):
        sim.step((float("nan"), 0.0))


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        SimConfig(substeps=0)
    with pytest.raises(ValueError):
        SimConfig(laps_to_finish=0)
