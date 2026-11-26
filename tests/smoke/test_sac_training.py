"""A very short SAC run: the shared Stable-Baselines3 pipeline works for an off-policy
algorithm too (checkpoints, validation, best model, metadata, loading, benchmark report)."""

import json

import numpy as np
import pytest

from neurodrive.agents.model_io import load_controller, read_model_meta
from neurodrive.agents.sac_agent import SACController
from neurodrive.env.driving_env import DrivingEnv
from neurodrive.replay.recorder import describe_kind, is_learned, network_info
from neurodrive.training.config import PPOTraining, SACExperiment, SACHyperparams
from neurodrive.training.ppo import train_ppo

TINY = SACExperiment(
    name="tiny_sac",
    sac=SACHyperparams(learning_starts=200, batch_size=32, buffer_size=5000, net_arch=(32, 32)),
    training=PPOTraining(
        total_timesteps=800,
        n_envs=2,
        vec_env="dummy",
        eval_every_steps=400,
        eval_episodes=1,
        checkpoint_every_steps=400,
    ),
)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return train_ppo(TINY, runs_root=tmp_path_factory.mktemp("runs"), log=lambda _: None)


def test_run_directory_contents(run_dir):
    for name in ("best_model.zip", "final_model.zip", "eval_history.csv", "train_summary.json"):
        assert (run_dir / name).is_file(), name
    assert sorted(p.name for p in (run_dir / "checkpoints").glob("*.zip"))[0] == "step_00000000.zip"
    s = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    assert s["total_timesteps"] >= 800


def test_saved_model_is_a_sac_controller(run_dir):
    meta = read_model_meta(run_dir / "best_model.zip")
    assert meta["algorithm"] == "sac"
    assert meta["sac"]["net_arch"] == [32, 32]
    controller = load_controller(run_dir / "best_model.zip")
    assert isinstance(controller, SACController) and controller.kind == "sac"
    env = DrivingEnv()
    obs, _ = env.reset(seed=0)
    a = controller.act(obs)
    assert a.shape == (2,) and np.all(np.abs(a) <= 1.0)
    np.testing.assert_array_equal(a, controller.act(obs))  # deterministic


def test_viewer_knows_sac(run_dir):
    controller = load_controller(run_dir / "best_model.zip")
    assert is_learned("sac") and describe_kind("sac") == "learned (SAC)"
    info, png = network_info(controller)
    assert info is not None and png is None
    assert info["hidden"] == [32, 32] and info["inputs"] == 11 and info["outputs"] == 2
