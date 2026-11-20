"""The Gymnasium environment: API compliance, validity, and reward-exploit tests."""

import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

import neurodrive.env  # noqa: F401  (registers NeuroDrive-v0)
from neurodrive.agents.baselines import PrivilegedAutopilot
from neurodrive.env.config import EnvConfig, ObservationConfig, RewardConfig
from neurodrive.env.driving_env import DrivingEnv


@pytest.fixture
def env():
    e = DrivingEnv()
    yield e
    e.close()


def rollout(env, policy, seed=0, max_steps=10_000, options=None):
    obs, info = env.reset(seed=seed, options=options)
    rewards, infos, observations = [], [], [obs]
    for _ in range(max_steps):
        obs, r, term, trunc, info = env.step(policy(obs))
        rewards.append(r)
        infos.append(info)
        observations.append(obs)
        if term or trunc:
            return np.array(rewards), infos, np.array(observations), term, trunc
    raise AssertionError("episode did not end")


# --------------------------------------------------------------------------- API
def test_gymnasium_env_checker_passes():
    e = DrivingEnv()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any checker warning fails the test
        gym_check_env(e, skip_render_check=True)


def test_sb3_env_checker_passes():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sb3_check_env(DrivingEnv(), warn=True)


def test_registered_with_gymnasium():
    e = gym.make("NeuroDrive-v0")
    obs, _ = e.reset(seed=0)
    assert obs.shape == (11,)
    e.close()


def test_spaces(env):
    assert env.observation_space.shape == (11,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (2,)
    assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)


def test_reset_with_same_seed_is_reproducible(env):
    a, _ = env.reset(seed=42)
    b, _ = env.reset(seed=42)
    c, _ = env.reset(seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_whole_episode_is_reproducible(env):
    actions = np.random.default_rng(5).uniform(-1, 1, (3000, 2))
    it = iter(actions)
    r1, _, o1, *_ = rollout(env, lambda _: next(it), seed=9)
    it = iter(actions)
    r2, _, o2, *_ = rollout(env, lambda _: next(it), seed=9)
    assert np.array_equal(r1, r2) and np.array_equal(o1, o2)


def test_observations_valid_over_many_random_episodes(env):
    rng = np.random.default_rng(0)
    for seed in range(40):
        _, _, obs, *_ = rollout(env, lambda _: rng.uniform(-1, 1, 2), seed=seed)
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))
        assert np.all(obs >= env.observation_space.low)
        assert np.all(obs <= env.observation_space.high)


def test_route_features_add_three_privileged_inputs():
    e = DrivingEnv(EnvConfig(observation=ObservationConfig(route_features=True)))
    obs, _ = e.reset(seed=0, options={"perturb": False})
    assert obs.shape == (14,)
    assert e.obs_builder.spec.privileged == (
        "heading_error_sin",
        "heading_error_cos",
        "lateral_offset",
    )
    # On the start line, aligned with the track: no heading error, centered.
    np.testing.assert_allclose(obs[11:], [0.0, 1.0, 0.0], atol=1e-6)


def test_observation_names_document_the_vector(env):
    names = env.obs_builder.spec.names
    assert names[0] == "ray_-90" and names[4] == "ray_+0" and names[8] == "ray_+90"
    assert names[9:] == ("speed", "steer")


# --------------------------------------------------------------------------- endings
def test_lap_completion_terminates_with_bonus(env):
    pilot = PrivilegedAutopilot(env)
    _rewards, infos, _, term, trunc = rollout(env, pilot.act)
    assert term and not trunc
    assert infos[-1]["termination_reason"] == "lap_complete"
    assert infos[-1]["reward_parts"]["lap_bonus"] == env.config.reward.lap_bonus
    assert sum(i["reward_parts"]["lap_bonus"] for i in infos) == env.config.reward.lap_bonus


def test_collision_terminates_with_penalty_once(env):
    _rewards, infos, _, term, trunc = rollout(env, lambda _: np.array([1.0, 1.0]))
    assert term and not trunc
    assert infos[-1]["termination_reason"] == "collision"
    crash = [i["reward_parts"]["crash"] for i in infos]
    assert crash.count(-env.config.reward.crash_penalty) == 1 and sum(crash) == -10.0


def test_standing_still_is_stalled_and_penalised():
    """Doing nothing must never be a safe strategy."""
    e = DrivingEnv()
    rewards, infos, _, term, trunc = rollout(e, lambda _: np.zeros(2))
    assert term and not trunc
    assert infos[-1]["termination_reason"] == "stalled"
    assert infos[-1]["time"] == pytest.approx(e.config.reward.stall_seconds, abs=0.05)
    assert rewards.sum() == pytest.approx(-e.config.reward.stall_penalty)


def test_time_limit_is_truncation_not_termination():
    """Creep forward slowly: never stalls, never finishes, so time runs out."""
    e = DrivingEnv(EnvConfig(max_episode_seconds=8.0, perturb_start=False))
    _rewards, infos, _, term, trunc = rollout(e, lambda o: np.array([0.0, 0.3 - 3 * o[9]]))
    assert trunc and not term
    assert infos[-1]["termination_reason"] == "time_limit"


# --------------------------------------------------------------------------- exploits
def test_progress_reward_equals_net_distance_for_any_behaviour(env):
    """Reward from progress depends only on where the car ended up, not the path taken.
    So driving back and forth, or in circles, earns exactly nothing extra."""
    k = env.config.reward.progress_per_meter
    rng = np.random.default_rng(1)
    for seed in range(10):
        _, info0 = env.reset(seed=seed)
        start = info0["distance"]
        total = 0.0
        while True:
            _obs, _, term, trunc, info = env.step(rng.uniform(-1, 1, 2))
            total += info["reward_parts"]["progress"]
            if term or trunc:
                break
        assert total == pytest.approx(k * (info["distance"] - start), abs=1e-9)


def test_driving_the_wrong_way_is_punished(env):
    """A car pointing backwards that drives cleanly the wrong way loses reward every meter
    and is ended as 'stalled' (no net forward progress), never rewarded.

    The 12 m track is too narrow for this car to U-turn at speed (it hits the wall), so
    the test turns it around directly instead of trying to drive a U-turn."""
    import math

    env.reset(seed=0, options={"perturb": False})
    s = env.sim.state
    s.heading = math.pi  # face west on the bottom straight: the wrong way
    # The state is the rear axle, half a wheelbase behind the body center. Flipping the
    # heading moves the rear axle to the other side of the center: +1 wheelbase in x.
    s.x += env.sim.params.wheelbase
    env.sim.progress.reset(np.array([0.0, -40.0]))
    env._history.clear()
    env._history.append(env.sim.progress.distance)

    rewards, infos = [], []
    while True:
        _, r, term, trunc, info = env.step(np.array([0.0, 0.5]))
        rewards.append(r)
        infos.append(info)
        if term or trunc:
            break
    assert infos[-1]["termination_reason"] == "stalled"
    assert infos[-1]["distance"] < -10  # it really did drive backwards along the track
    assert all(i["reward_parts"]["progress"] <= 0 for i in infos)
    assert sum(rewards) < -env.config.reward.stall_penalty


def test_reward_scale_of_a_clean_lap(env):
    """A clean lap is worth progress (0.1/m * ~491 m) + lap bonus: ~59. Pins the scale so a
    change to reward coefficients is a deliberate, visible decision."""
    pilot = PrivilegedAutopilot(env)
    rewards, _infos, *_ = rollout(env, pilot.act, options={"perturb": False})
    expected = 0.1 * env.track.length + 10.0
    assert rewards.sum() == pytest.approx(expected, abs=0.2)


def test_custom_reward_coefficients_are_used():
    cfg = EnvConfig(reward=RewardConfig(crash_penalty=3.0))
    e = DrivingEnv(cfg)
    _, infos, *_ = rollout(e, lambda _: np.array([1.0, 1.0]))
    assert infos[-1]["reward_parts"]["crash"] == -3.0


# --------------------------------------------------------------------------- render
def test_rgb_array_render(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    e = DrivingEnv(render_mode="rgb_array")
    e.reset(seed=0)
    e.step(np.array([0.0, 1.0]))
    frame = e.render()
    assert frame is not None and frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8 and frame.std() > 0
    e.close()
