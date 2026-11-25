"""Real races: several cars on one track that see and touch each other."""

import numpy as np
import pytest

from neurodrive.agents.pure_pursuit import PurePursuitDriver
from neurodrive.env.config import EnvConfig
from neurodrive.env.driving_env import DrivingEnv
from neurodrive.replay.format import ReplayError
from neurodrive.replay.lab import driver_entry
from neurodrive.replay.race import record_race
from neurodrive.replay.recorder import verify_replay
from neurodrive.sim.race import RaceWorld, rectangle_overlap
from neurodrive.sim.sensors import RaySensor
from neurodrive.sim.simulation import SimConfig, Simulation
from neurodrive.sim.track import load_track


def square(cx, cy, half=1.0):
    return np.array([[cx - half, cy - half], [cx + half, cy - half],
                     [cx + half, cy + half], [cx - half, cy + half]])  # fmt: skip


def test_rectangle_overlap():
    assert rectangle_overlap(square(0, 0), square(3, 0)) is None
    push = rectangle_overlap(square(0, 0), square(1.5, 0))
    np.testing.assert_allclose(push, [-0.5, 0.0])  # move the first square left by 0.5


@pytest.fixture(scope="module")
def oval():
    return load_track("beginner_oval")


def make_world(oval, n, laps=1):
    cfg = SimConfig(laps_to_finish=laps, max_time=120.0)
    return RaceWorld([Simulation(oval, config=cfg) for _ in range(n)])


def test_grid_is_behind_the_line_and_cars_do_not_overlap(oval):
    world = make_world(oval, 6)
    world.reset()
    assert all(sim.progress.distance < 0 for sim in world.cars)
    corners = [sim.corners() for sim in world.cars]
    for i in range(6):
        for j in range(i + 1, 6):
            assert rectangle_overlap(corners[i], corners[j]) is None


def test_sensors_see_other_cars(oval):
    sensor = RaySensor(oval)
    sim = Simulation(oval)
    sim.reset()
    origin = sim.corners().mean(axis=0)
    alone = sensor.cast(origin, sim.state.heading, sim.progress.segment).distances
    ahead = origin + 10.0 * np.array([np.cos(sim.state.heading), np.sin(sim.state.heading)])
    blocked = sensor.cast(origin, sim.state.heading, sim.progress.segment, [square(*ahead)])
    middle = len(alone) // 2  # the straight-ahead ray
    assert blocked.distances[middle] == pytest.approx(9.0, abs=0.05)
    assert alone[middle] > 30


def test_contact_pushes_apart_slows_down_and_counts_once(oval):
    world = make_world(oval, 2)
    world.reset()
    a, b = world.cars
    b.state.x, b.state.y, b.state.heading = a.state.x + 1.0, a.state.y, a.state.heading
    a.state.speed = b.state.speed = 10.0
    world.step([np.zeros(2), np.zeros(2)])
    assert world.status[0].contacts == world.status[1].contacts == 1
    assert rectangle_overlap(a.corners(), b.corners()) is None
    assert a.state.speed < 10.0 and b.state.speed < 10.0


def test_autopilot_race_finishes_in_order(oval):
    world = make_world(oval, 2, laps=1)
    world.reset()
    drivers = [PurePursuitDriver(speed_margin=0.8), PurePursuitDriver(speed_margin=0.4)]
    while not world.done:
        cars = zip(drivers, world.cars, world.status, strict=True)
        world.step([d.act(s) if st.racing else None for d, s, st in cars])
    times = [st.finish_time for st in world.status]
    assert all(t is not None for t in times)
    assert world.standings()[0] == int(np.argmin(times))


def test_mixed_physics_is_refused(oval):
    from neurodrive.sim.vehicle import VehicleParams

    cars = [Simulation(oval), Simulation(oval, vehicle=VehicleParams(model="dynamic"))]
    with pytest.raises(ValueError, match="physics"):
        RaceWorld(cars)


def test_recorded_race_has_positions_and_cannot_be_verified_alone():
    entries = [driver_entry("autopilot"), driver_entry("heuristic")]
    reps = record_race(entries, "beginner_oval", laps=1, log=lambda _: None)
    positions = sorted(r.meta["race"]["position"] for r in reps)
    assert positions == [1, 2]
    assert all(r.meta["multi_car"] for r in reps)
    with pytest.raises(ReplayError, match="race replay"):
        verify_replay(reps[0])


# ------------------------------------------------------------------ traffic in training
TRAFFIC = EnvConfig.model_validate({"traffic": {"cars": 3}})


def test_traffic_env_passes_the_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    check_env(DrivingEnv(TRAFFIC), skip_render_check=True)


def test_traffic_is_deterministic_and_seen_by_the_sensors():
    def run(seed):
        env = DrivingEnv(TRAFFIC)
        obs, _ = env.reset(seed=seed)
        out = [obs]
        for _ in range(60):
            obs, *_ = env.step(np.array([0.0, 1.0]))
            out.append(obs)
        return np.array(out), env

    a, env = run(4)
    b, _ = run(4)
    np.testing.assert_array_equal(a, b)
    assert env.race is not None and len(env.race.cars) == 4
    alone = DrivingEnv()
    alone.reset(seed=4)
    assert alone.race is None


def test_hitting_a_car_costs_reward():
    from neurodrive.agents.baselines import HeuristicController

    # Very slow opponents: the sensor-only heuristic (which ignores cars) will hit them.
    env = DrivingEnv(EnvConfig.model_validate({"traffic": {"cars": 3, "speed_range": [0.2, 0.25]}}))
    driver = HeuristicController()
    obs, _ = env.reset(seed=0)
    total = 0.0
    while True:
        obs, _, term, trunc, info = env.step(driver.act(obs))
        total += info["reward_parts"]["contact"]
        if term or trunc:
            break
    assert info["contacts"] >= 1
    assert total == pytest.approx(-env.config.reward.contact_penalty * info["contacts"])
