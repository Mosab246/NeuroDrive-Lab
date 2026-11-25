"""The hard (dynamic) physics through the whole stack: env config, observation, drivability."""

import numpy as np
import pytest

from neurodrive.agents.pure_pursuit import PurePursuitDriver
from neurodrive.env.config import EnvConfig
from neurodrive.env.driving_env import DrivingEnv
from neurodrive.sim.simulation import SimConfig, Simulation, Termination
from neurodrive.sim.track import builtin_track_ids, load_track
from neurodrive.sim.vehicle import VehicleParams

HARD = EnvConfig.model_validate(
    {"physics": {"model": "dynamic"}, "observation": {"motion_features": True}}
)


@pytest.mark.parametrize("track", builtin_track_ids())
def test_every_builtin_track_is_drivable_with_slip(track):
    sim = Simulation(
        load_track(track),
        vehicle=VehicleParams(model="dynamic"),
        config=SimConfig(laps_to_finish=1, max_time=120.0),
    )
    sim.reset(seed=0)
    driver = PurePursuitDriver()
    while not sim.done:
        sim.step(driver.act(sim))
    assert sim.termination is Termination.LAPS_COMPLETE


def test_env_uses_the_configured_physics():
    assert DrivingEnv().sim.params.model == "kinematic"
    env = DrivingEnv(HARD)
    assert env.sim.params.model == "dynamic"
    assert env.observation_space.shape == (13,)
    assert env.obs_builder.spec.names[-2:] == ("lateral_speed", "yaw_rate")
    assert env.obs_builder.spec.version == "obs-v1+motion"


def test_motion_features_show_a_slide():
    env = DrivingEnv(HARD)
    obs, _ = env.reset(seed=0)
    assert obs[-1] == 0.0 and obs[-2] == 0.0
    for _ in range(90):  # floor it, then steer hard right
        obs, *_ = env.step(np.array([0.0, 1.0]))
    for _ in range(10):
        obs, _, term, *_ = env.step(np.array([1.0, 1.0]))
        if term:
            break
    assert obs[-1] < 0  # rotating clockwise
    assert abs(obs[-2]) > 0.01  # sliding sideways


def test_old_configs_without_physics_still_load():
    cfg = EnvConfig.model_validate({"track": "beginner_oval"})
    assert cfg.physics.model == "kinematic"
    assert cfg.observation.motion_features is False
