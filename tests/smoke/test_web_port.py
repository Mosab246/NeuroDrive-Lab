"""The browser version's JavaScript simulator must agree with the Python simulator.

Runs both on the same track with the same trained-network driver and compares every
step. Needs Node.js (skipped without it).
"""

import json
import shutil
import subprocess
from pathlib import Path

import neat
import numpy as np
import pytest

from neurodrive.agents.neat_agent import NEATController
from neurodrive.env.config import ObservationConfig, SensorConfig
from neurodrive.env.observations import ObservationBuilder
from neurodrive.sim.simulation import SimConfig, Simulation
from neurodrive.sim.track import load_any_track
from neurodrive.web_export import controller_data, export_web

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")
RUNNER = Path("web/test/run.mjs").resolve()


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
def setup(tmp_path_factory):
    from stable_baselines3 import PPO

    from neurodrive.agents.ppo_agent import PPOController
    from neurodrive.env.driving_env import DrivingEnv

    tmp = tmp_path_factory.mktemp("web")
    neat_c = neat_driver()
    ppo_c = PPOController(PPO("MlpPolicy", DrivingEnv(), device="cpu", seed=0))
    models = [controller_data(neat_c, "neat"), controller_data(ppo_c, "ppo")]
    data = export_web([], tmp / "data.js", tracks=("beginner_oval", "proc_4"), extra_models=models)
    return tmp, data, neat_c, ppo_c


def node(tmp: Path, request: dict) -> dict:
    path = tmp / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    out = subprocess.run([NODE, str(RUNNER), str(path)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def python_episode(track_id: str, controller, max_time: float):
    track = load_any_track(track_id)
    sim = Simulation(track, config=SimConfig(max_time=max_time, laps_to_finish=1))
    sim.reset(perturb=False)
    builder = ObservationBuilder(sim, SensorConfig(), ObservationConfig())
    states = []
    while not sim.done:
        sim.step(controller.act(builder.build()))
        s = sim.state
        states.append([s.x, s.y, s.heading, s.speed, s.steer])
    return np.array(states), str(sim.termination)


@pytest.mark.parametrize("track_id", ["beginner_oval", "proc_4"])
def test_same_episode_in_python_and_javascript(setup, track_id):
    tmp, data, neat_c, _ = setup
    expected, ending = python_episode(track_id, neat_c, max_time=20.0)
    assert len(expected) > 150 and expected[:, 3].max() > 10  # it really drives
    got = node(tmp, {"data": str(data), "track": track_id, "model": 0, "max_time": 20.0})
    assert got["termination"] == ending
    js = np.array(got["states"])
    assert js.shape == expected.shape
    np.testing.assert_allclose(js, expected, rtol=0, atol=1e-6)


def test_ppo_network_gives_the_same_actions(setup):
    tmp, data, _, ppo_c = setup
    rng = np.random.default_rng(0)
    obs = rng.uniform(0, 1, size=(50, 11)).astype(np.float32)
    obs[:, 10] = rng.uniform(-1, 1, size=50)
    got = np.array(
        node(tmp, {"data": str(data), "model": 1, "observations": obs.tolist()})["actions"]
    )
    expected = np.array([ppo_c.act(o) for o in obs])
    np.testing.assert_allclose(got, expected, atol=1e-5)
