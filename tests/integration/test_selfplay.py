"""Self-play: saved PPO and NEAT models driving opponent cars in traffic."""

import pickle

import neat
import numpy as np
import pytest

from neurodrive.agents.learned_opponent import LearnedDriver, _load, _NumpyPolicy
from neurodrive.agents.model_io import write_model_meta
from neurodrive.agents.neat_agent import NEATController
from neurodrive.cli import _model_list
from neurodrive.env.config import EnvConfig
from neurodrive.env.driving_env import DrivingEnv


def neat_driver() -> NEATController:
    """A hand-wired NEAT network that actually drives: steer towards the side with more
    room, accelerate when the road ahead is clear, ease off at high speed."""
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "configs/neat/default.ini",
    )
    genome = next(iter(neat.Population(config).population.values()))
    # Inputs -1..-9: rays at -90,-60,-40,-20,0,20,40,60,90 deg; -10 speed; -11 steer.
    # Outputs: 0 steering (+ = right), 1 throttle/brake.
    weights = {(-7, 0): 1.5, (-4, 0): -1.5, (-8, 0): 1.0, (-3, 0): -1.0,
               (-5, 1): 1.5, (-10, 1): -0.9}  # fmt: skip
    for key, gene in genome.connections.items():
        gene.weight = weights.get(key, 0.0)
        gene.enabled = True
    for (a, b), _w in weights.items():
        assert (a, b) in genome.connections, "default.ini must start fully connected"
    for k in (0, 1):
        genome.nodes[k].bias = 0.1 if k == 1 else 0.0
        genome.nodes[k].response = 1.0
    return NEATController(genome, config)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    from stable_baselines3 import PPO

    tmp = tmp_path_factory.mktemp("league")
    ppo = PPO("MlpPolicy", DrivingEnv(), device="cpu", seed=0)
    ppo_path = tmp / "ppo.zip"
    ppo.save(ppo_path)
    env_config = EnvConfig().model_dump(mode="json")
    write_model_meta(ppo_path, {"algorithm": "ppo", "env_config": env_config, "label": "t"})
    driver = neat_driver()  # hand-wired network that really drives
    neat_path = tmp / "neat.pkl"
    with neat_path.open("wb") as fh:
        pickle.dump((driver.genome, driver.config), fh)
    write_model_meta(neat_path, {"algorithm": "neat", "env_config": env_config, "label": "t"})
    return tmp, str(ppo_path), str(neat_path)


def test_numpy_policy_gives_the_same_actions_as_predict(models):
    _, ppo_path, _ = models
    controller, _ = _load(ppo_path)
    policy = _NumpyPolicy(controller)
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.uniform(0.0, 1.0, 11).astype(np.float32)
        np.testing.assert_allclose(policy.act(obs), controller.act(obs), atol=1e-5)


def traffic_env(paths, share=1.0, physics="kinematic"):
    cfg = EnvConfig.model_validate(
        {
            "track": "curved_circuit",
            "physics": {"model": physics},
            "traffic": {"cars": 3, "learned_opponents": paths, "learned_share": share},
        }
    )
    return DrivingEnv(cfg)


def test_learned_opponents_drive_their_own_cars(models):
    _, ppo_path, neat_path = models
    env = traffic_env([ppo_path, neat_path])
    env.reset(seed=3)
    if not any(d.path == neat_path for d in env._opponent_drivers):
        env.reset(seed=4)
    drivers = env._opponent_drivers
    assert all(isinstance(d, LearnedDriver) for d in drivers)
    assert {d.path for d in drivers} <= {ppo_path, neat_path}
    for k, d in enumerate(drivers, start=1):
        assert d.builder.sim is env.race.cars[k]  # each sees from its own car
    neat_cars = [k for k, d in enumerate(drivers, start=1) if d.path == neat_path]
    start = {k: env.race.cars[k].progress.distance for k in neat_cars}
    for _ in range(90):
        env.step(np.array([0.0, 0.5], dtype=np.float32))
    for k in neat_cars:  # the hand-wired NEAT driver really drives away
        assert env.race.cars[k].progress.distance > start[k] + 5.0


def test_share_zero_keeps_scripted_opponents(models):
    _, ppo_path, _ = models
    env = traffic_env([ppo_path], share=0.0)
    env.reset(seed=0)
    assert not any(isinstance(d, LearnedDriver) for d in env._opponent_drivers)


def test_opponent_must_match_physics(models):
    _, ppo_path, _ = models
    env = traffic_env([ppo_path], physics="dynamic")
    with pytest.raises(ValueError, match="physics"):
        env.reset(seed=0)


def test_model_list_reads_files_and_folders(models):
    tmp, ppo_path, neat_path = models
    assert _model_list(str(tmp)) == sorted([ppo_path, neat_path])
    assert _model_list(None) == []
    with pytest.raises(ValueError, match="not found"):
        _model_list(str(tmp / "missing.zip"))
